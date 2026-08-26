"""SpiLiFormer over an RGB frame -> positional, projected vision tokens.

    forward(rgb, T_out) : (B, 3, H, W) -> (T_out, B, N, d_model)

SpiLiFormer (ICCV 2025) is a spiking transformer with lateral inhibition, ImageNet 85.82%
at T=4. Its stem is convolutional and its stages are hierarchical, so `forward_features`
yields a (T, B, 768, H/16, W/16) SPATIAL map -- at 384x384 a 24x24 grid, i.e. 576 tokens.

The T axis on the output is broadcast: a still image is constant over the fusion's
timesteps, so the same token tensor is expanded to T_out rather than recomputed. What T
buys downstream is membrane state in the fusion stack, not new visual content.

Only the feedforward pass is used. SpiLiFormer's lateral-inhibition feedback is a two-pass
design whose second pass exists to sharpen a classification decision; a feature extractor
wants the features. `RefCOCOGrounding._disable_unused_backbone_grads` switches off the
parameters that pass never reaches -- measured at 4.91M of them returning `grad is None`.

Why `pos_std` is a knob and not a constant
------------------------------------------
`forward` computes `lateral(feats) + pos`, and that sum then goes through a LayerNorm and
a BINARISING LIF in the fusion. At the default `pos_std=0.02` against features of order 1,
the positional embedding is ~2% of the pre-norm magnitude and the threshold can discard it
outright. Measured on two trained checkpoints via `tools/diagnose.py`:

    RMS(pos) / RMS(lateral output)  =  0.0052 - 0.0183

and on the same checkpoints the fusion's attention perplexity sat at 575.8-576.0 out of a
576-key maximum -- statistically uniform, i.e. the attention map carried no location at
all. Those two facts are consistent with position being lost here, before the map is ever
formed. `last_lateral_rms` is stashed every forward call so `diagnose.py` can measure the
ratio without duplicating this module's arithmetic.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron as sj_neuron

from spiketrandvg import forks

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
