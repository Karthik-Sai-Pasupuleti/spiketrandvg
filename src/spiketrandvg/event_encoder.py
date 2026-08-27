"""Language-conditioned spiking event encoder: Meta-SpikeFormer over event voxel cubes.

    forward(cube, gains) : (T,B,2,H,W) -> {tap: (T,B,C,H/s,W/s)}

Three things distinguish this from a plain SpikeFormer wrapper, all of them from the
Gate-2 design:

1. **T is a real temporal axis.** The cube carries T=9 genuine time bins of the event
   stream, and LIF membranes evolve across them inside the multi-step kernels. This is
   NOT nine stacked channels. The fork's own `forward()` would replicate one frame T
   times, so it is never called -- `forward_features` is invoked directly.

2. **I-LIF integer activations** (`--ilif`, default on). SpikeYOLO's `mem_update` is a
   genuine LIF -- it integrates `mem = (mem_old - spike)*decay + x[t]` across the T axis
   with a soft reset -- but its activation is `round(clamp(mem, 0, 4))` rather than a
   binary threshold. A neuron therefore emits 0-4 per step instead of 0/1, carrying
   ~2.3 bits instead of 1 while remaining multiplication-free downstream.

3. **Language modulates firing thresholds** at stages 2-4 (`gains`). See
   `ThresholdModulator` for why this is close to free at inference in a spiking network
   and is not in an ANN.

The accumulator
---------------
`accumulate()` sums each neuron's spikes over the T axis: a neuron that fired 7 times
out of 9 returns 7. **This is the hinge of the whole design.** Everything upstream is
binary or small-integer and pays the precision cost the Gate-1 study measured; everything
downstream of the accumulator is an ordinary real-valued tensor and an ordinary
regression/classification problem. Confining the precision anxiety to one side of this
line is the point.

Why the taps are multi-scale, and why the input is not downsampled
------------------------------------------------------------------
Talk2Event boxes are small: median target ~62x66 px, ~1.3% of the frame. Downsampling
throws that away before layer one, and a single stride-16 tap gives cells larger than the
objects. At native 480x640:

    stride  4 -> 120x160 (128 ch)    stride 16 -> 30x40 (512 ch, block3)
    stride  8 ->  60x80 (256 ch)     stride 16 -> 30x40 (640 ch, block4)

Taps are MEMBRANE potentials, not spikes -- each block returns its residual stream, which
is analog. Downstream consumers open with their own neuron.
"""

from __future__ import annotations

import contextlib
from functools import partial

import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron as sj_neuron
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from spiketrandvg import forks

__all__ = ["EventEncoder", "ThresholdModulator", "TAPS", "DEFAULT_TAPS",
           "MODULATED_STAGES"]

# tap name -> (attribute on the backbone, stride, channels)
TAPS: dict[str, tuple[str, int, int]] = {
    "s4": ("ConvBlock1_2", 4, 128),
    "s8": ("ConvBlock2_2", 8, 256),
    "s16": ("block3", 16, 512),
    "s16b": ("block4", 16, 640),
}
DEFAULT_TAPS = ("s8", "s16")

# Stages 2-4 take the language modulation. Stage 1 is deliberately excluded: it runs at
# stride 4 over the full 120x160 grid (the most expensive place to add anything) and
# encodes edges and polarity, which are not what a caption disambiguates.
MODULATED_STAGES: tuple[str, ...] = ("ConvBlock2_1", "ConvBlock2_2", "block3", "block4")

# Depth order, used to drop modulation of stages below the deepest tap (see __init__).
_STAGE_ORDER: dict[str, int] = {
    "ConvBlock1_1": 0, "ConvBlock1_2": 1,
    "ConvBlock2_1": 2, "ConvBlock2_2": 3,
    "block3": 4, "block4": 5,
}


class ThresholdModulator(nn.Module):
    """Four caption sub-queries -> per-channel firing-threshold gains for stages 2-4.

    Why threshold modulation rather than a feature-wise multiply
    -------------------------------------------------------------
    In a spiking neuron the threshold is a *parameter of the neuron*, not an operation:
    the neuron already compares its membrane against something, and changing what it
    compares against costs nothing extra at inference. Folded into the preceding
    convolution's weights it disappears entirely. The ANN equivalent -- FiLM -- is a
    multiply that must actually be executed on every activation. This is a genuine
    advantage of doing it in the spiking pathway, and unlike the usual "spikes are
    cheaper than MACs" accounting it does not depend on any contested energy model.

    The second reason is information-theoretic. A spiking encoder has a finite spike
    budget. A language-blind encoder must represent everything about the scene, because
    it does not know what will be asked. A conditioned one can spend its spikes on what
    the caption is actually about.

    How the gain implements a threshold
    -----------------------------------
    A LIF integrates `v[t] = decay*v[t-1] + x[t]` and fires when `v >= theta`. The
    dynamics are linear in `x`, so scaling the input by `g > 0` gives `g*v_orig >= theta`,
    i.e. exactly the original neuron at threshold `theta/g`. Multiplying the input by a
    gain IS threshold modulation -- no approximation. `g > 1` lowers the effective
    threshold and the channel fires more readily.

    Gains are bounded to `exp(±max_log_gain)` so a query cannot silence a channel
    outright (gain -> 0 kills the gradient through it) or saturate it.
    """

    def __init__(self, d_query: int, stage_channels: dict[str, int],
                 hidden: int = 128, max_log_gain: float = 0.5):
        super().__init__()
        self.max_log_gain = max_log_gain
        self.heads = nn.ModuleDict()
        for stage, ch in stage_channels.items():
            head = nn.Sequential(nn.Linear(d_query, hidden), nn.GELU(),
                                 nn.Linear(hidden, ch))
            # zero-init the last layer: training starts at gain exactly 1.0, i.e. the
            # unmodulated encoder, so conditioning can only help from a known baseline.
            # Zero here is safe (unlike a zero-init output head) because it is followed
            # by tanh and exp, not by a matmul whose gradient it would kill.
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.heads[stage] = head

    def forward(self, queries: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, n_query, d) -> {stage: (B, C)} multiplicative gains, centred on 1.0."""
        pooled = queries.mean(dim=1)                       # (B, d)
        return {stage: torch.exp(self.max_log_gain * torch.tanh(head(pooled)))
                for stage, head in self.heads.items()}


class EventEncoder(nn.Module):
    """Meta-SpikeFormer over event cubes, multi-scale taps, optionally conditioned.

    Args:
        ckpt_path: Meta-SpikeFormer ImageNet checkpoint. Its 3-channel RGB stem is
            averaged and repeated onto the 2 event-polarity channels.
        taps: which strides to return.
        ilif: replace the backbone's binary LIF with SpikeYOLO's integer I-LIF.
        freeze: run the backbone under no_grad.
        modulated_stages: which stages accept language gains.

    Memory (measured, 480x640, 32 GiB card, T=5): trainable backbone needs 16.0 GiB at
    B=1 and OOMs at B=2, because activations for every timestep are kept for backward.
    **T=9 raises this roughly 1.8x over T=5**, so a trainable backbone at T=9 will not
    fit at B=1 on 32 GiB -- expect to freeze the backbone, tune only the later stages,
    or accumulate gradients. Frozen it runs at inference cost (3.6 GiB at B=1, T=5).

    Gradient checkpointing is deliberately NOT offered: recomputation re-runs stateful
    LIF neurons whose membranes have already advanced, silently producing different
    activations from the forward pass.
    """

    def __init__(
        self,
        ckpt_path: str | None = None,
        taps: tuple[str, ...] = DEFAULT_TAPS,
        in_channels: int = 2,
        ilif: bool = True,
        freeze: bool = False,
        modulated_stages: tuple[str, ...] = MODULATED_STAGES,
    ):
        super().__init__()
        bad = set(taps) - set(TAPS)
        if bad:
            raise ValueError(f"unknown taps {sorted(bad)}; choose from {sorted(TAPS)}")
        self.taps = tuple(taps)
        self.in_channels = in_channels
        self.ilif = ilif

        sdt2 = forks.load_metaspikformer()
        with forks.allow_cupy_construction():
            # The metaspikformer_8_512 factory hard-codes in_channels=3, so passing ours
            # through **kwargs raises "multiple values for keyword argument".
            # Instantiate the class directly with the same 8_512 configuration.
            self.backbone = sdt2.Spiking_vit_MetaFormer(
                img_size_h=224, img_size_w=224,      # unused: fully convolutional
                patch_size=16, embed_dim=[128, 256, 512, 640], num_heads=8,
                mlp_ratios=4, in_channels=in_channels, num_classes=0, qkv_bias=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
            )
        for m in self.backbone.modules():
            if isinstance(m, MultiStepLIFNode):
                m.backend = "torch"

        self.ckpt_report: dict[str, int] = {}
        if ckpt_path is not None:
            self.ckpt_report = self._load_imagenet(ckpt_path)

        if ilif:
            self._swap_to_ilif()

        # feature taps
        self._features: dict[str, torch.Tensor] = {}
        for name in self.taps:
            attr, _, _ = TAPS[name]
            module = getattr(self.backbone, attr)
            target = module[-1] if isinstance(module, nn.ModuleList) else module
            target.register_forward_hook(self._make_tap_hook(name))

        # language conditioning: one pre-hook per modulated stage, reading `self._gains`.
        #
        # Stages BELOW the deepest tap are dropped. MEASURED: with taps ("s8","s16") the
        # deepest tap is `block3`, so `block4` still runs but nothing downstream reads it
        # -- modulating it changes no output, and its 4 modulator tensors came back with
        # `grad is None` while still being counted as trainable. Filtering here keeps the
        # parameter count honest instead of advertising conditioning that cannot act.
        deepest = max(_STAGE_ORDER[TAPS[t][0]] for t in self.taps)
        self.modulated_stages = tuple(
            s for s in modulated_stages
            if hasattr(self.backbone, s) and _STAGE_ORDER.get(s, 99) <= deepest)
        dropped = [s for s in modulated_stages if s not in self.modulated_stages]
        if dropped:
            print(f"[event_encoder] not modulating {dropped}: below the deepest tap "
                  f"({max(self.taps, key=lambda t: _STAGE_ORDER[TAPS[t][0]])})")
        self._gains: dict[str, torch.Tensor] | None = None
        for stage in self.modulated_stages:
            module = getattr(self.backbone, stage)
            target = module[-1] if isinstance(module, nn.ModuleList) else module
            target.register_forward_pre_hook(self._make_gain_hook(stage))

        self.frozen = freeze
        if freeze:
            self.backbone.requires_grad_(False)

    # -- construction helpers --------------------------------------------------
    def _swap_to_ilif(self) -> None:
        """Replace every binary LIF with SpikeYOLO's integer I-LIF.

        `mem_update` is a real LIF -- it integrates across the T axis with a soft reset
        -- but quantises to {0,1,2,3,4} instead of {0,1}. It is parameter-free and starts
        each call from a zero membrane, so it needs no explicit reset.

        CAVEAT worth knowing before reading any result: the ImageNet checkpoint was
        trained with BINARY activations. Integer activations are up to 4x larger, so
        every downstream BatchNorm and threshold in the pretrained weights is
        mis-calibrated at init and has to be re-learned. Expect a worse starting point
        than `--no-ilif` and judge the two at the end of training, not at epoch 0.
        """
        mem_update, _ = forks.load_ilif()
        n = 0
        for parent in self.backbone.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, MultiStepLIFNode):
                    setattr(parent, name, mem_update())
                    n += 1
        self.n_ilif = n

    def _load_imagenet(self, path: str) -> dict[str, int]:
        """Load the RGB ImageNet checkpoint, adapting the 3-channel stem.

        The stem is Conv2d(3, 64, k=7); its pretrained filters are averaged over RGB and
        repeated onto the event-polarity channels. Mean rather than sum keeps activation
        scale intact.
        """
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sd = blob.get("model", blob)
        stem_key = "downsample1_1.encode_conv.weight"
        if stem_key in sd and sd[stem_key].shape[1] != self.in_channels:
            w = sd[stem_key].mean(dim=1, keepdim=True)
            sd[stem_key] = w.repeat(1, self.in_channels, 1, 1)
        msg = self.backbone.load_state_dict(sd, strict=False)
        report = {"loaded": len(sd) - len(msg.unexpected_keys),
                  "missing": len(msg.missing_keys),
                  "unexpected": len(msg.unexpected_keys)}
        print(f"[event_encoder] Meta-SpikeFormer from {path}: "
              f"{report['loaded']}/{len(sd)} tensors (missing {report['missing']}, "
              f"unexpected {report['unexpected']})")
        return report

    def _make_tap_hook(self, name: str):
        def hook(_m, _i, out):
            self._features[name] = out
        return hook

    def _make_gain_hook(self, stage: str):
        """Scale this stage's input by the language gain == modulating its threshold."""
        def pre_hook(_m, args):
            if self._gains is None or stage not in self._gains:
                return None                      # unconditioned: leave the input alone
            x = args[0]
            g = self._gains[stage]               # (B, C)
            if x.dim() != 5 or g.shape[-1] != x.shape[2]:
                return None                      # shape mismatch: skip rather than crash
            return (x * g[None, :, :, None, None].to(x.dtype),) + args[1:]
        return pre_hook

    # -- properties ------------------------------------------------------------
    @property
    def out_channels(self) -> dict[str, int]:
        return {n: TAPS[n][2] for n in self.taps}

    @property
    def strides(self) -> dict[str, int]:
        return {n: TAPS[n][1] for n in self.taps}

    @property
    def stage_channels(self) -> dict[str, int]:
        """Input channels of each modulated stage, for sizing a ThresholdModulator."""
        ch = {"ConvBlock2_1": 256, "ConvBlock2_2": 256, "block3": 512, "block4": 640}
        return {s: ch[s] for s in self.modulated_stages}

    def reset(self) -> None:
        """Zero every spikingjelly membrane. I-LIF holds no cross-call state, so this is
        a no-op once `ilif=True` -- harmless, and correct if only some were swapped."""
        for m in self.backbone.modules():
            if isinstance(m, sj_neuron.BaseNode):
                m.reset()

    # -- forward ---------------------------------------------------------------
    def forward(self, cube: torch.Tensor,
                gains: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """(T,B,2,H,W) -> {tap: (T,B,C,H/s,W/s)} membrane potentials.

        `gains` from a `ThresholdModulator`; None runs the encoder language-blind, which
        is the ablation the conditioned version is measured against.
        """
        if cube.dim() != 5:
            raise ValueError(f"expected (T,B,C,H,W), got {tuple(cube.shape)}")
        T, B, C, H, W = cube.shape
        if C != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {C}")
        if H % 16 or W % 16:
            raise ValueError(f"H and W must be divisible by 16, got {H}x{W}")

        self.reset()
        self._features.clear()
        self._gains = gains
        try:
            # no_grad ONLY when the backbone is frozen AND unconditioned.
            #
            # MEASURED BUG, not a precaution: the gains enter *inside* this block via the
            # pre-hooks, so wrapping a conditioned forward in no_grad severs the graph
            # between them and the loss. `ThresholdModulator`'s 0.35M parameters then get
            # `grad is None` on every step -- the entire language-conditioning pathway is
            # silently untrainable while still appearing in the trainable-parameter
            # count. Verified: 0/16 modulator tensors received gradient before this fix.
            #
            # The cost is real: a conditioned frozen backbone must still build the
            # activation graph for all T steps, so it is priced like a trainable one
            # (~16 GiB at B=1, T=5; more at T=9) rather than like inference. Only the
            # PARAMETERS stay fixed. Pass `condition_encoder=False` to get the cheap path.
            no_grad = self.frozen and gains is None
            ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
            with ctx:
                self.backbone.forward_features(cube)
        finally:
            self._gains = None

        missing = [n for n in self.taps if n not in self._features]
        if missing:
            raise RuntimeError(f"tap hooks did not fire for {missing}")
        return {n: self._features[n] for n in self.taps}

    @staticmethod
    def accumulate(feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Sum over the T axis: {tap: (T,B,C,h,w)} -> {tap: (B,C,h,w)}.

        The binary/integer world ends here. A neuron that fired 7 of 9 steps contributes
        a 7, and everything downstream is real-valued. See the module docstring.
        """
        return {k: v.sum(dim=0) for k, v in feats.items()}
