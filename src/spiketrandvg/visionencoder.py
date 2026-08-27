"""Vision encoders: a spiking event encoder and a spiking RGB encoder.

  `EventEncoder`   Meta-SpikeFormer over event voxel cubes. The main pathway for
                   Talk2Event -- T=9 real timesteps, I-LIF integer activations, and
                   firing thresholds modulated by language (`ThresholdModulator`).

  `VisionEncoder`  SpiLiFormer over an RGB frame. The main pathway for RefCOCO, and the
                   branch a Talk2Event fusion row would use -- though for that row it
                   should be an ANN rather than a spiking network, since the spiking
                   claim concerns the event pathway where the sensor is.

Both return positional, projected tokens at a common width, so the fusion stack in
`model.py` does not care which one produced them.
"""

from __future__ import annotations

from functools import partial
from spikingjelly.clock_driven import neuron as sj_neuron
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
import contextlib
import torch
import torch.nn as nn

from spiketrandvg import utils as forks


# ====================================================================================
# from event_encoder.py
# ====================================================================================

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

# SpiLiFormer-DVS is a two-stage network, so its taps are different: patch_embed1 lands
# at stride 8 (128 ch) and stage2 at stride 16 (256 ch). Verified at 480x640 -> 60x80
# and 30x40. Only ONE genuine downsample separates them, so it is a shallower pyramid
# than Meta-SpikeFormer's.
DVS_TAPS: dict[str, tuple[str, int, int]] = {
    "s8":  ("patch_embed1", 8, 128),
    "s16": ("stage2", 16, 256),
}
_DVS_STAGE_ORDER: dict[str, int] = {"patch_embed1": 0, "stage1": 1,
                                    "patch_embed2": 2, "stage2": 3}
# stage2 is the only block worth conditioning: stage1 sits before the last downsample and
# patch_embed* are plain convolution stacks.
DVS_MODULATED_STAGES: tuple[str, ...] = ("stage2",)

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
        modulated_stages: tuple[str, ...] | None = None,
        backbone: str = "metaspikformer",
        img_size: tuple[int, int] = (480, 640),
    ):
        super().__init__()
        if backbone not in ("metaspikformer", "spiliformer_dvs"):
            raise ValueError(f"unknown backbone {backbone!r}")
        self.backbone_name = backbone
        self.tap_table = TAPS if backbone == "metaspikformer" else DVS_TAPS
        self.stage_order = (_STAGE_ORDER if backbone == "metaspikformer"
                            else _DVS_STAGE_ORDER)
        if modulated_stages is None:
            modulated_stages = (MODULATED_STAGES if backbone == "metaspikformer"
                                else DVS_MODULATED_STAGES)
        bad = set(taps) - set(self.tap_table)
        if bad:
            raise ValueError(f"unknown taps {sorted(bad)} for {backbone}; "
                             f"choose from {sorted(self.tap_table)}")
        self.taps = tuple(taps)
        self.in_channels = in_channels
        self.ilif = ilif

        if backbone == "spiliformer_dvs":
            self._build_spiliformer_dvs(img_size, in_channels)
        else:
            self._build_metaspikformer(in_channels)
        self._finish_init(ckpt_path, ilif, modulated_stages, freeze)

    def _build_metaspikformer(self, in_channels: int) -> None:
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

    def _build_spiliformer_dvs(self, img_size, in_channels: int) -> None:
        """SpiLiFormer's event-native CIFAR10-DVS variant.

        `img_size_h/w` MUST be passed: they default to 128 and the repo's `SpiLiFormer()`
        factory does not forward them, so the model would build a 128x128 assumption and
        raise a broadcast error at any other resolution. Verified: with the real size it
        runs at 480x640 and lands stage2 on a 30x40 grid.
        """
        sl = forks.load_spiliformer_dvs()
        with forks.allow_cupy_construction():
            self.backbone = sl.Spike_Lateral_Transformer(
                img_size_h=img_size[0], img_size_w=img_size[1], patch_size=16,
                embed_dims=256, num_heads=16, mlp_ratios=1, in_channels=in_channels,
                num_classes=0, qkv_bias=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
            )
        forks.use_torch_backend(self.backbone)

    def _finish_init(self, ckpt_path, ilif, modulated_stages, freeze) -> None:
        self.ckpt_report: dict[str, int] = {}
        if ckpt_path is not None:
            self.ckpt_report = self._load_imagenet(ckpt_path)

        if ilif:
            self._swap_to_ilif()

        # feature taps
        self._features: dict[str, torch.Tensor] = {}
        for name in self.taps:
            attr, _, _ = self.tap_table[name]
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
        deepest = max(self.stage_order[self.tap_table[t][0]] for t in self.taps)
        self.modulated_stages = tuple(
            s for s in modulated_stages
            if hasattr(self.backbone, s) and self.stage_order.get(s, 99) <= deepest)
        dropped = [s for s in modulated_stages if s not in self.modulated_stages]
        if dropped:
            print(f"[event_encoder] not modulating {dropped}: below the deepest tap "
                  f"({max(self.taps, key=lambda t: self.stage_order[self.tap_table[t][0]])})")
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
        return {n: self.tap_table[n][2] for n in self.taps}

    @property
    def strides(self) -> dict[str, int]:
        return {n: self.tap_table[n][1] for n in self.taps}

    @property
    def stage_channels(self) -> dict[str, int]:
        """Input channels of each modulated stage, for sizing a ThresholdModulator."""
        ch = ({"ConvBlock2_1": 256, "ConvBlock2_2": 256, "block3": 512, "block4": 640}
              if self.backbone_name == "metaspikformer" else {"stage2": 256})
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
                # both backbones expose forward_features((T,B,C,H,W)); the DVS variant
                # additionally returns an intermediate, which the tap hooks already
                # captured, so the return value is discarded either way
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


# ====================================================================================
# from vision_encoder.py
# ====================================================================================

__all__ = ["VisionEncoder", "sj_reset"]


def sj_reset(module: nn.Module) -> None:
    """Zero every spikingjelly membrane inside `module` (never the caller itself).

    Per-neuron rather than `spikingjelly.functional.reset_net`: that helper calls
    `.reset()` on every submodule that has one, which includes any caller that defines
    its own `reset`, and recurses until the stack blows.
    """
    for m in module.modules():
        if isinstance(m, sj_neuron.BaseNode):
            m.reset()


class VisionEncoder(nn.Module):
    """SpiLiFormer backbone + 1x1 lateral projection + learnable 2D positional embedding.

    Args:
        ckpt: SpiLiFormer ImageNet checkpoint. None starts from scratch, which wastes the
            one pretrained visual prior available.
        d_model: fusion width to project 768 backbone channels down to.
        T_model: SpiLiFormer's own internal timesteps. The shipped checkpoint is T=4;
            1 runs a single feature-extraction pass, which is cheaper but puts the
            backbone's LIF layers outside the regime their weights were calibrated in.
        max_hw: (H/16, W/16) for the input this model will see, so the positional table
            is used directly instead of being interpolated on every forward call.
        freeze: hold the backbone fixed.
        pos_std: init std of the positional embedding -- see the module docstring.
    """

    def __init__(
        self,
        ckpt: str | None = None,
        d_model: int = 256,
        T_model: int = 1,
        max_hw: tuple[int, int] = (24, 24),
        freeze: bool = False,
        pos_std: float = 0.02,
    ):
        super().__init__()
        sl = forks.load_spiliformer()
        with forks.allow_cupy_construction():
            self.backbone = sl.SpiLiFormer_10_768(T=T_model)
        forks.use_torch_backend(self.backbone)
        self.T_model = T_model
        self.report = {}
        if ckpt:
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = blob.get("model", blob)
            msg = self.backbone.load_state_dict(sd, strict=False)
            self.report = {"loaded": len(sd) - len(msg.unexpected_keys),
                           "missing": len(msg.missing_keys),
                           "unexpected": len(msg.unexpected_keys)}
            print(f"[vision_encoder] SpiLiFormer from {ckpt}: "
                  f"{self.report['loaded']}/{len(sd)} tensors "
                  f"(missing {self.report['missing']}, unexpected {self.report['unexpected']})")
        self.frozen = freeze
        self.backbone.requires_grad_(not freeze)

        self.lateral = nn.Conv2d(768, d_model, kernel_size=1)
        self.pos = nn.Parameter(torch.zeros(1, d_model, *max_hw))
        nn.init.trunc_normal_(self.pos, std=pos_std)
        self.d_model = d_model
        # RMS of the pre-position lateral output, refreshed every forward call; compared
        # against RMS(pos) by tools/diagnose.py. See the module docstring.
        self.last_lateral_rms: float | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, rgb: torch.Tensor, T_out: int) -> torch.Tensor:
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W) rgb, got {tuple(rgb.shape)}")
        # membranes persist across calls AND are shaped by the last input's resolution,
        # so a reset is mandatory, not hygiene
        sj_reset(self.backbone)
        ctx = torch.no_grad() if self.frozen else contextlib.nullcontext()
        with ctx:
            x = rgb.unsqueeze(0).repeat(self.T_model, 1, 1, 1, 1)
            feats, _tmp = self.backbone.forward_features(x)      # (Tm,B,768,H/16,W/16)
        feats = feats.mean(0)                                    # collapse SpiLiFormer's T

        B, C, H, W = feats.shape
        y = self.lateral(feats)
        self.last_lateral_rms = y.detach().float().pow(2).mean().sqrt().item()
        pos = self.pos
        if pos.shape[-2:] != (H, W):
            pos = nn.functional.interpolate(pos, size=(H, W), mode="bilinear",
                                            align_corners=False)
        y = y + pos
        # row-major: token index n = row * W + col. `model.py`'s soft-argmax head depends
        # on this ordering to pair marginals with the right coordinate axis.
        tok = y.reshape(B, self.d_model, H * W).permute(0, 2, 1)  # (B, N, d)
        return tok.unsqueeze(0).expand(T_out, -1, -1, -1)         # (T_out, B, N, d)
