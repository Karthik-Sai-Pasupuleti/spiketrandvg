"""Spiking cross-modal fusion: CMSF's cross-attention over event feature maps.

Wraps CMSF (`repositories/CMSF`, "Multimodal Spiking Neural Network for Image-Text
Retrieval") from the frozen fork. Nothing in it is edited -- `SCA_Block` and its
`SpikingCrossAttention` / `Spiking_GFNN` are used through their own constructors.

What this module adds is the adaptation CMSF does not do: CMSF fuses two *sequences*
(region features and caption tokens) for retrieval. Here one side is a dense event
feature map, so the map is flattened to a query sequence, fused, and folded back to
(T, B, C, H, W) for a detection neck.

Why the attention scales to a dense map
---------------------------------------
`SpikingCrossAttention` associates right-to-left: it forms `k^T @ v` (D x D) and then
`q @ (k^T v)`, so cost is O(L * D^2) -- LINEAR in the number of query tokens. A 60x80
map is 4800 queries, which softmax attention could not afford at T=5 but this can.
That property is why CMSF is usable here at all.

Text padding
------------
CMSF's attention takes no key-padding mask, and captions are padded to MAX_TEXT_LEN.
Padded keys would otherwise contribute to `k^T @ v` and silently pollute every query.
The mask is applied AFTER the spike coder -- zeroing spikes at padded positions removes
their contribution exactly, whereas masking before the coder's LayerNorm would not
(LayerNorm re-inflates zeros).

Spike discipline
----------------
`EventVisionEncoder` returns MEMBRANE potentials (analog), and `TextEmbedder`
returns an analog projection. CMSF's blocks expect spike trains on both sides -- its own
pipeline runs each modality through a spike coder first. Both sides are therefore coded
here: text with the fork's `RepeatTextEncoder` (B,L,D -> T,B,L,D), vision with
`SpikeMapTokens`, which is the same LayerNorm + dynamic-threshold LIF minus the repeat,
since the vision stream already carries a real time axis.

State
-----
Every LIF holds membrane state across calls. `forward` resets the whole module first, so
no state leaks between sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron as sj_neuron

from spiketrandvg.utils import forks

__all__ = ["SpikeMapTokens", "CrossModalFusion", "build_fusion"]


class SpikeMapTokens(nn.Module):
    """Analog (T, B, L, D) -> spikes, via LayerNorm + CMSF's dynamic-threshold LIF.

    The vision counterpart of CMSF's `RepeatTextEncoder`: identical body, without the
    repeat, because the event stream already has its own T axis.
    """

    def __init__(self, dim: int):
        super().__init__()
        cmsf = forks.load_cmsf()
        self.norm = nn.LayerNorm(dim)
        with forks.allow_cupy_construction():
            self.lif = cmsf.Dynamic_Threshold_LIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        forks.use_torch_backend(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(self.norm(x))


class CrossModalFusion(nn.Module):
    """Language-conditioned event features, one CMSF cross-attention stack per scale.

    forward(maps, text_tokens, text_mask)
        maps        {name: (T, B, C_name, H_s, W_s)}  membrane potentials from the encoder
        text_tokens (B, L, d_model)                   from TextEmbedder
        text_mask   (B, L) 1 for real tokens          the tokenizer's attention_mask
        -> {name: (T, B, d_model, H_s, W_s)}          all scales at a common width

    Scales listed in `fuse` get the full cross-attention; the rest are only projected to
    `d_model` and passed through, and pick language up later through the neck's top-down
    path. Restricting `fuse` is the main cost lever: the stride-4 tap is 120x160 = 19200
    query tokens, 4x the stride-8 map, for the features whose semantics are weakest.
    Default is to fuse every scale.

    Output convention: a fused scale returns the SCA_Block residual stream (spike-valued
    sums, small integers, sparse); an unfused scale returns the lateral conv's BatchNorm
    output (analog membrane). Both are legitimate inputs to the neck, whose every block
    opens with its own neuron -- the same convention `EventVisionEncoder` documents.
    """

    def __init__(
        self,
        in_channels: dict[str, int],
        d_model: int = 256,
        num_heads: int = 8,
        depth: int = 1,
        mlp_ratio: float = 2.0,
        fuse: tuple[str, ...] | None = None,
        attn_bn_gain: float = 3.0,
        T: int = 5,
    ):
        """
        Args:
            in_channels: {tap name: channels} as reported by the event encoder.
            d_model: common fusion width; must match the text encoder's d_model.
            num_heads: heads in CMSF's attention; must divide d_model.
            depth: SCA_Blocks per fused scale.
            mlp_ratio: hidden width of CMSF's gated MLP, as a multiple of d_model.
                Kept at 2 rather than CMSF's 4 because the MLP, not the attention, is
                what dominates memory on a dense map (its hidden tensor is
                T*B*L*2*mlp_ratio*d_model floats).
            fuse: which taps get cross-attention. Default: every tap except the finest.
            attn_bn_gain: initial gain of the q/k/v/proj BatchNorms inside CMSF's
                attention. See `_scale_attention_bn` -- at PyTorch's default of 1.0 the
                whole attention branch emits exactly zero and the model is text-blind.
            T: event-stream timesteps; the text coder repeats to this length.
        """
        super().__init__()
        if d_model % num_heads:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        cmsf = forks.load_cmsf()
        sy = forks.load_spikeyolo()

        self.names = tuple(in_channels)
        fuse = self.names if fuse is None else tuple(fuse)
        unknown = set(fuse) - set(self.names)
        if unknown:
            raise ValueError(f"fuse names {sorted(unknown)} not in {sorted(self.names)}")
        self.fuse = tuple(fuse)
        self.d_model = d_model
        self.T = T

        # Calibrate the tapped membranes into the I-LIF's firing window before the
        # lateral. MEASURED, not defensive: Meta-SpikeFormer's residual stream comes off
        # the taps at std ~0.02, and SpikeYOLO's neuron is round(clamp(x, 0, 4)) -- so
        # every value below 0.5 rounds to zero and the entire pyramid goes dead, silently
        # and at every scale. SpikeYOLO's own blocks never hit this because inside its
        # network each I-LIF is fed a BatchNorm output; this restores that precondition.
        # Note the consequence for eval: with untrained running stats BatchNorm is the
        # identity, so a randomly-initialised model must be run in train() mode (or have
        # its BN stats calibrated) before its eval path shows anything but zeros.
        self.tap_norm = nn.ModuleDict(
            {n: nn.BatchNorm2d(c) for n, c in in_channels.items()}
        )
        # 1x1 lateral: LIF -> conv -> BN, so the projection itself is spike-driven
        self.lateral = nn.ModuleDict(
            {n: sy.MS_StandardConv(c, d_model, k=1, s=1) for n, c in in_channels.items()}
        )

        with forks.allow_cupy_construction():
            # (B, L, D) -> (T, B, L, D) spikes; one coder shared by all fused scales,
            # since the caption representation is identical at every scale
            self.text_coder = cmsf.RepeatTextEncoder(T, d_model)
            self.vis_coder = nn.ModuleDict({n: SpikeMapTokens(d_model) for n in self.fuse})
            self.blocks = nn.ModuleDict(
                {
                    n: nn.ModuleList(
                        [
                            cmsf.SCA_Block(
                                dim=d_model,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                norm_layer=nn.LayerNorm,
                            )
                            for _ in range(depth)
                        ]
                    )
                    for n in self.fuse
                }
            )
        forks.use_torch_backend(self)
        self._scale_attention_bn(attn_bn_gain)

    def _scale_attention_bn(self, gain: float) -> None:
        """Start CMSF's attention in a live firing regime (threshold-dependent BN init).

        Not a tweak -- without it the cross-attention is dead on arrival, and the failure
        is silent. `SpikingCrossAttention` multiplies `q @ (k^T v)` by `scale = 0.125` and
        then thresholds at 0.5, so the raw spike-count product has to reach 4 before a
        single output spike is emitted. With q/k/v BatchNorms at PyTorch's default gain
        of 1.0 the q, k and v neurons fire at ~0.5%, the product never gets near 4, and
        the branch emits exactly zero -- `x + attn(x, y, y)` collapses to `x`, and the
        model detects objects while completely ignoring the caption.

        Measured on Talk2Event captions with BatchNorm statistics calibrated
        (160x224 input, T=5, ImageNet-initialised vision):

            gain   k firing   attention firing   caption sensitivity
            1.0     0.6%          0.0%              0.0000   <- text-blind
            3.0     2.6%         26.3%              1.0816
            6.0     4.0%         34.6%              1.0798
            10.0    7.1%         39.8%              1.0221

        3.0 is the default: the smallest gain that makes the branch live, which keeps the
        activation sparsity an SNN exists for. The gain is a plain BatchNorm weight, so
        training is free to move it.

        This is the standard threshold-dependent-BN argument (tdBN, Zheng et al. AAAI
        2021): a spiking layer's normaliser has to be scaled to its neuron's threshold,
        not left at the ANN default of 1.
        """
        for stack in self.blocks.values():
            for blk in stack:
                a = blk.attn
                for bn in (a.q_bn, a.k_bn, a.v_bn, a.proj_bn):
                    nn.init.constant_(bn.weight, gain)

    @property
    def out_channels(self) -> dict[str, int]:
        return {n: self.d_model for n in self.names}

    def reset(self) -> None:
        """Zero every LIF membrane in the fusion stack.

        Neurons are reset directly rather than via `sj_functional.reset_net(self)`:
        that helper calls `.reset()` on every submodule that has one, which includes
        THIS module, and recurses until the stack blows. Any module that owns
        spikingjelly neurons and also exposes a `reset` method has the same trap.
        """
        for m in self.modules():
            if isinstance(m, sj_neuron.MultiStepLIFNode):
                m.reset()

    def forward(
        self,
        maps: dict[str, torch.Tensor],
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        missing = set(self.names) - set(maps)
        if missing:
            raise KeyError(f"missing feature maps {sorted(missing)}")
        if text_tokens.dim() != 3:
            raise ValueError(f"expected (B, L, D) text tokens, got {tuple(text_tokens.shape)}")
        if text_tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"text width {text_tokens.shape[-1]} != fusion d_model {self.d_model}"
            )

        self.reset()

        # (B, L, D) -> (T, B, L, D) spikes, then kill the padded keys
        txt = self.text_coder(text_tokens)
        txt = txt * text_mask[None, :, :, None].to(txt.dtype)

        out: dict[str, torch.Tensor] = {}
        for name in self.names:
            m = maps[name]
            T_, B_, C_, H_, W_ = m.shape
            m = self.tap_norm[name](m.flatten(0, 1)).reshape(T_, B_, C_, H_, W_)
            x = self.lateral[name](m)                   # (T, B, d_model, H, W)
            if name not in self.fuse:
                out[name] = x
                continue

            T, B, D, H, W = x.shape
            q = x.flatten(3).permute(0, 1, 3, 2).contiguous()   # (T, B, H*W, D)
            q = self.vis_coder[name](q)
            for blk in self.blocks[name]:
                q = blk(q, txt)                                  # query=vision, kv=text
            out[name] = q.permute(0, 1, 3, 2).reshape(T, B, D, H, W).contiguous()
        return out


def build_fusion(in_channels: dict[str, int], **kwargs) -> CrossModalFusion:
    return CrossModalFusion(in_channels, **kwargs)
