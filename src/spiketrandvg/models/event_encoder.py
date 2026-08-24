"""Event vision encoder: a spiking multi-scale backbone over event voxel cubes.

Wraps Meta-SpikeFormer (Spike-Driven Transformer V2) from the frozen fork, adapted
for 2-channel event input and tapped at several strides. Nothing in the fork is
edited: the model is built through its own constructor and features are collected
with forward hooks.

Input / output
-------------
    forward(cube)  cube: (T, B, 2, H, W) event counts, T = T_STEPS
    -> dict of str -> (T, B, C_s, H/s, W/s), one entry per requested stride

**The taps are MEMBRANE potentials, not spikes.** Each MS_Block returns its residual
stream, which is analog (measured range roughly [-53, +26], 100% non-zero) -- exactly
the convention Meta-SpikeFormer uses internally, where every block begins by passing
its input through a LIF. Consumers must therefore apply their own neuron before any
Linear/Conv if they want that matmul to be a spike-driven accumulation. Returning the
membrane rather than pre-spiking it keeps that choice with the consumer, and avoids
double-spiking when the next block would do it anyway.

Why multi-scale, and why not downsample the input
-------------------------------------------------
Talk2Event boxes are small: the median target is ~62x66 px, about 1.3% of the frame,
and accuracy at IoU >= 0.95 tolerates only ~1.6 px of centre error. Feeding the
encoder a downsampled frame throws that precision away before the first layer, and
a single stride-16 tap gives cells (40x60 px at 128x256 input) larger than the
objects themselves. At native 480x640 the taps land on

    stride  4 -> 120x160  (128 ch)
    stride  8 ->  60x80   (256 ch)
    stride 16 ->  30x40   (512 ch after block3, 640 ch after block4)

which is the same feature pyramid the EventRefer baseline uses.

Temporal semantics
------------------
The fork's own `forward()` would replicate a single frame across T steps, so it is
never called. `forward_features` is invoked directly with a real (T, B, C, H, W)
stack, and the LIF membranes evolve across those T steps inside spikingjelly's
multi-step kernels. Membrane state PERSISTS across calls, so `reset()` must run
before every new sequence -- `forward()` does it automatically.
"""

from __future__ import annotations

import contextlib
import types
from functools import partial

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional as sj_functional
from spikingjelly.clock_driven import neuron as sj_neuron
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from spiketrandvg.utils import forks

# tap name -> (attribute path on the backbone, stride, channels) for the 8_512 variant
TAPS: dict[str, tuple[str, int, int]] = {
    "s4": ("ConvBlock1_2", 4, 128),
    "s8": ("ConvBlock2_2", 8, 256),
    "s16": ("block3", 16, 512),
    "s16b": ("block4", 16, 640),
}
DEFAULT_TAPS = ("s8", "s16")


@contextlib.contextmanager
def _allow_cupy_construction():
    """spikingjelly asserts cupy is importable when a neuron is CONSTRUCTED with
    backend='cupy', which the fork hard-codes at every call site. cupy is only
    touched at forward time, so a sentinel lets construction through; every neuron
    is switched to the 'torch' backend immediately afterwards."""
    if getattr(sj_neuron, "cupy", None) is not None:
        yield
        return
    sj_neuron.cupy = types.SimpleNamespace()
    try:
        yield
    finally:
        sj_neuron.cupy = None


class EventVisionEncoder(nn.Module):
    """Meta-SpikeFormer over event cubes, with multi-scale feature taps.

    Memory note (measured, 480x640, T=5, 32 GiB card): trainable backbone needs
    16.0 GiB at B=1 and OOMs at B=2, because activations for all T steps must be
    kept for backprop. With freeze=True the backbone runs under no_grad and the
    cost collapses to inference levels (3.6 GiB at B=1, 7.1 at B=2).

    Gradient checkpointing is deliberately NOT offered: recomputation during
    backward would re-run stateful LIF neurons whose membranes have already been
    advanced, silently producing activations that differ from the forward pass.
    """

    def __init__(
        self,
        in_channels: int = 2,
        taps: tuple[str, ...] = DEFAULT_TAPS,
        ckpt_path: str | None = None,
        freeze: bool = False,
        trainable_from: str | None = None,
    ):
        """
        Args:
            in_channels: event cube channels (2 = one per polarity).
            taps: which strides to return; keys of TAPS.
            ckpt_path: Meta-SpikeFormer ImageNet checkpoint (weights under key 'model').
            freeze: freeze the whole backbone.
            trainable_from: if set (e.g. "downsample3"), freeze everything before that
                submodule and train from it onward. Ignored when freeze=True.
        """
        super().__init__()
        bad = set(taps) - set(TAPS)
        if bad:
            raise ValueError(f"unknown taps {sorted(bad)}; choose from {sorted(TAPS)}")
        self.taps = tuple(taps)
        self.in_channels = in_channels

        sdt2 = forks.load_metaspikformer()
        with _allow_cupy_construction():
            # The metaspikformer_8_512 factory hard-codes in_channels=3 and
            # num_classes=1000 as explicit keywords, so passing ours through **kwargs
            # raises "multiple values for keyword argument". Instantiate the fork's
            # class directly with the same 8_512 configuration instead.
            # num_classes=0 makes the classifier an Identity; we never use it anyway.
            self.backbone = sdt2.Spiking_vit_MetaFormer(
                img_size_h=224,           # unused: the network is fully convolutional
                img_size_w=224,
                patch_size=16,
                embed_dim=[128, 256, 512, 640],
                num_heads=8,
                mlp_ratios=4,             # MS_ConvBlock hard-codes 4x in its reshape
                in_channels=in_channels,
                num_classes=0,
                qkv_bias=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                depths=8,
                sr_ratios=1,
            )
        for m in self.backbone.modules():
            if isinstance(m, MultiStepLIFNode):
                m.backend = "torch"  # read at forward time, so safe to set here

        self.ckpt_report: dict[str, int] = {}
        if ckpt_path is not None:
            self.ckpt_report = self._load_imagenet(ckpt_path)

        self._features: dict[str, torch.Tensor] = {}
        self._handles = []
        for name in self.taps:
            attr, _, _ = TAPS[name]
            module = getattr(self.backbone, attr)
            target = module[-1] if isinstance(module, nn.ModuleList) else module
            self._handles.append(
                target.register_forward_hook(self._make_hook(name))
            )

        self.frozen = freeze
        if freeze:
            self.backbone.requires_grad_(False)
        elif trainable_from is not None:
            self._freeze_before(trainable_from)

    # -- checkpoint ------------------------------------------------------------
    def _load_imagenet(self, path: str) -> dict[str, int]:
        """Load the RGB ImageNet checkpoint, adapting the 3-channel stem to our
        event channels. The stem is Conv2d(3, 64, k=7); its pretrained filters are
        averaged over RGB and repeated, preserving the learned edge responses at
        the right magnitude (mean, not sum, keeps activation scale intact)."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sd = blob.get("model", blob)
        stem_key = "downsample1_1.encode_conv.weight"

        adapted = False
        if stem_key in sd:
            w = sd[stem_key]  # (64, 3, 7, 7)
            if w.shape[1] != self.in_channels:
                sd = dict(sd)
                sd[stem_key] = w.mean(dim=1, keepdim=True).repeat(
                    1, self.in_channels, 1, 1
                )
                adapted = True

        msg = self.backbone.load_state_dict(sd, strict=False)
        report = {
            "in_ckpt": len(sd),
            "missing": len(msg.missing_keys),
            "unexpected": len(msg.unexpected_keys),
            "loaded": len(sd) - len(msg.unexpected_keys),
            "stem_adapted": int(adapted),
        }
        print(
            f"[event_encoder] {path}: loaded {report['loaded']}/{report['in_ckpt']} "
            f"tensors (missing {report['missing']}, unexpected {report['unexpected']})"
            + (f", stem 3ch -> {self.in_channels}ch" if adapted else "")
        )
        return report

    def _freeze_before(self, boundary: str) -> None:
        """Freeze stages preceding `boundary` in forward_features order."""
        order = [
            "downsample1_1", "ConvBlock1_1", "downsample1_2", "ConvBlock1_2",
            "downsample2", "ConvBlock2_1", "ConvBlock2_2",
            "downsample3", "block3", "downsample4", "block4",
        ]
        if boundary not in order:
            raise ValueError(f"trainable_from must be one of {order}")
        for name in order[: order.index(boundary)]:
            getattr(self.backbone, name).requires_grad_(False)

    # -- hooks / state ---------------------------------------------------------
    def _make_hook(self, name: str):
        def hook(_module, _args, output):
            self._features[name] = output
        return hook

    def reset(self) -> None:
        """Zero every LIF membrane. Required before each new sequence."""
        sj_functional.reset_net(self.backbone)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            # BatchNorm sees T folded into the batch; keep pretrained running stats
            self.backbone.eval()
        return self

    @property
    def out_channels(self) -> dict[str, int]:
        return {n: TAPS[n][2] for n in self.taps}

    @property
    def strides(self) -> dict[str, int]:
        return {n: TAPS[n][1] for n in self.taps}

    # -- forward ---------------------------------------------------------------
    def forward(self, cube: torch.Tensor) -> dict[str, torch.Tensor]:
        """(T, B, in_channels, H, W) -> {tap: (T, B, C, H/s, W/s)}."""
        if cube.dim() != 5:
            raise ValueError(f"expected (T, B, C, H, W), got {tuple(cube.shape)}")
        T, B, C, H, W = cube.shape
        if C != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {C}")
        if H % 16 or W % 16:
            raise ValueError(f"H and W must be divisible by 16, got {H}x{W}")

        self.reset()          # no membrane state leaks in from the previous sequence
        self._features.clear()
        if self.frozen:
            # nothing upstream of the backbone needs gradients, so skip storing
            # activations entirely -- this is the difference between 16 GiB and
            # 3.6 GiB at B=1, 480x640, T=5
            with torch.no_grad():
                self.backbone.forward_features(cube)
        else:
            self.backbone.forward_features(cube)

        missing = [n for n in self.taps if n not in self._features]
        if missing:
            raise RuntimeError(f"hooks did not fire for {missing}")
        return {n: self._features[n] for n in self.taps}


def build_event_encoder(
    ckpt_path: str | None = None,
    taps: tuple[str, ...] = DEFAULT_TAPS,
    **kwargs,
) -> EventVisionEncoder:
    return EventVisionEncoder(ckpt_path=ckpt_path, taps=taps, **kwargs)
