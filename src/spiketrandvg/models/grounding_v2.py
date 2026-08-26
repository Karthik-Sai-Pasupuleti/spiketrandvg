"""SpikeGroundingV2: frozen pretrained encoders, text-queried cross-attention, one box.

    events (T,B,2,480,640)   rgb (B,3,480,640)   text tokens (B,L,d_model)
            |                        |                        |
    DetectionBackbone          SpiLiFormer            <supplied by the caller>
    SpikeYOLO, Talk2Event      ICCV 2025, ImageNet    roberta by default, FROZEN
    detection mAP@0.5 0.5307   85.82% @ T=4
            | s8 / s16               | /16                    |
      lateral + learnable 2D pos on each                      |
            +--- concatenated vision tokens ---+              |
      lateral 1x1 + learnable 2D pos                |
            |  (T,B,N_vis,256) vision tokens        |
            +-----------> TextQueryFusion <---------+
                  CMSF spiking cross-attention
                  Q = TEXT, K = V = VISION
                          |  (T,B,L,256) language tokens carrying visual context
                     BoxHead: masked mean over real tokens -> spiking MLP
                          |
                   ONE box (B,4) normalised cxcywh

Trainable: lateral projections, positional embedding, cross-attention, box head.
Frozen: the vision backbone (12.7M).

The text encoder is NOT part of this model
------------------------------------------
`forward` takes `text_tokens` of shape (B, L, d_model) plus an `attention_mask`; producing
them is the caller's job. The SpikeLM encoder that used to live here has been removed --
124.3M parameters of roberta-base transplanted into a spiking BERT, never spike-pretrained,
and with it frozen the model was measured caption-blind over 85 epochs (mean caption delta
+0.0009, against +0.051 when the encoders were allowed to move; see
`docs/research-log.md`). Whatever replaces it only has to satisfy that one contract.

Why the attention runs text->vision
-----------------------------------
The previous design queried with vision and attended over text, which produced 19200 query
tokens and a head that had to pool them back down. Querying with text instead makes the
output sequence the caption itself -- 80 tokens, not 19200 -- so the box is read from a
language-shaped representation that has absorbed visual context. It is also 240x fewer
query tokens.

The catch, and why positional encoding is not optional here
-----------------------------------------------------------
CMSF's `SpikingCrossAttention` is LINEAR attention: it forms `attn = k^T v`, a (D/h, D/h)
matrix, and then `q @ attn`. That product **sums over every key position**. With vision as
the keys and values, the spatial index is marginalised away entirely -- each text token
would receive the same position-independent summary of the whole frame, and no amount of
training could recover where the referred object is.

Adding a positional embedding to the vision tokens is what keeps localisation
*expressible*: `sum_p k_p^T v_p` then contains content-position correlations that a text
query can read out. This is a genuine bottleneck rather than a free fix -- the whole
spatial layout has to pass through h x (D/h)^2 = 8192 numbers -- so it is the first thing
to suspect if the model cannot ground. There is a same-frame overfit test for exactly that
question; run it before committing to a long training run.

Prior context: two earlier grounding attempts failed, and the finished one was measured
caption-blind (feeding deliberately wrong captions changed mIoU by 0.001). See
`docs/research-log.md`, findings 3 and 4.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional as _sjf
from spikingjelly.clock_driven import neuron as sj_neuron


def sj_reset(module: nn.Module) -> None:
    """Zero every spikingjelly membrane inside `module` (never the caller itself)."""
    for m in module.modules():
        if isinstance(m, sj_neuron.BaseNode):
            m.reset()

from spiketrandvg.datasets.events_voxel_cube import T_STEPS
from spiketrandvg.models.spikeyolo_detector import DetectionBackbone
from spiketrandvg.utils import forks

__all__ = ["SpikeGroundingV2", "SpatialCrossAttention", "RgbTokens", "VisionTokens",
           "build_grounding_v2"]

DEFAULT_TAPS = ("s8", "s16")


class SpatialCrossAttention(nn.Module):
    """Softmax cross-attention over the spatial axis. Q = text, K = V = vision.

    The alternative to CMSF's linear attention, and the reason it is worth having: linear
    attention forms `k^T v` and thereby SUMS OVER EVERY KEY POSITION, so with vision as the
    keys the spatial index is marginalised away and no text query can ask *where*
    something is. Softmax attention keeps one weight per vision position, which is exactly
    the "which location does this phrase refer to" operation grounding needs.

    Normally that costs O(L_q * N_kv), which is why CMSF avoids it. Here the queries are
    the CAPTION -- 80 tokens against 6000 vision tokens is 480k pairs per head, nothing.
    Reversing the attention direction is what makes softmax affordable.

    Spike discipline: q/k/v projections are BN + integer-LIF exactly as in CMSF, so those
    matmuls are spike-driven. The attention product itself is analog -- that is the honest
    cost of being able to localise, and it is one matmul, not the bulk of the compute.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        cmsf = forks.load_cmsf()
        self.h = num_heads
        self.dh = dim // num_heads
        self.scale = self.dh ** -0.5

        self.q_linear, self.k_linear, self.v_linear = (nn.Linear(dim, dim) for _ in range(3))
        self.q_bn, self.k_bn, self.v_bn = (nn.BatchNorm1d(dim) for _ in range(3))
        with forks.allow_cupy_construction():
            self.q_lif, self.k_lif, self.v_lif = (
                cmsf.Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend="cupy")
                for _ in range(3))
            self.proj_lif = cmsf.Dynamic_Threshold_LIFNode(
                tau=2.0, detach_reset=True, backend="cupy")
        forks.use_torch_backend(self)
        self.proj = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

    def _spike_proj(self, x, lin, bn, lif):
        """(T,B,L,D) -> (T,B,L,D) through Linear + BN + LIF, CMSF's ordering."""
        T, B, L, D = x.shape
        y = lin(x.flatten(0, 1))                                  # (T*B, L, D)
        y = bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D)
        return lif(y)

    def forward(self, query, key, value, key_mask=None, return_attn: bool = False):
        """query (T,B,Lq,D) text; key/value (T,B,N,D) vision; key_mask (B,N) or None.

        `return_attn=True` additionally returns the post-softmax weights (T,B,h,Lq,N) --
        the one tensor in this block that still carries the spatial index before the
        output is binarised by `proj_lif`. See `RefCOCOGrounding`'s `attn_softargmax`
        head, which reads a box centre off exactly this tensor rather than off the
        binarised residual stream.
        """
        T, B, Lq, D = query.shape
        N = key.shape[2]
        q = self._spike_proj(query, self.q_linear, self.q_bn, self.q_lif)
        k = self._spike_proj(key, self.k_linear, self.k_bn, self.k_lif)
        v = self._spike_proj(value, self.v_linear, self.v_bn, self.v_lif)

        q = q.reshape(T, B, Lq, self.h, self.dh).permute(0, 1, 3, 2, 4)   # (T,B,h,Lq,dh)
        k = k.reshape(T, B, N, self.h, self.dh).permute(0, 1, 3, 2, 4)
        v = v.reshape(T, B, N, self.h, self.dh).permute(0, 1, 3, 2, 4)

        logits = (q @ k.transpose(-2, -1)) * self.scale                   # (T,B,h,Lq,N)
        if key_mask is not None:
            logits = logits.masked_fill(~key_mask[None, :, None, None, :].bool(),
                                        torch.finfo(logits.dtype).min)
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = (attn @ v).permute(0, 1, 3, 2, 4).reshape(T, B, Lq, D)

        out = self.proj(out.flatten(0, 1))
        out = self.proj_bn(out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, Lq, D)
        out = self.proj_lif(out)
        return (out, attn) if return_attn else out


class SpatialBlock(nn.Module):
    """SpatialCrossAttention + CMSF's spiking gated MLP, pre-norm residual."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        cmsf = forks.load_cmsf()
        self.attn = SpatialCrossAttention(dim, num_heads)
        with forks.allow_cupy_construction():
            self.mlp = cmsf.Spiking_GFNN(dim=dim, hidden_dim=int(dim * mlp_ratio))
        forks.use_torch_backend(self)

    def forward(self, x, y, key_mask=None, return_attn: bool = False):
        if return_attn:
            a, attn = self.attn(x, y, y, key_mask, return_attn=True)
            x = x + a
            return x + self.mlp(x), attn
        x = x + self.attn(x, y, y, key_mask)
        return x + self.mlp(x)


class VisionTokens(nn.Module):
    """Frozen SpikeYOLO backbone -> positional, projected vision tokens.

    forward(cube) -> (T, B, N, d_model) with N = sum of H_s*W_s over the taps.

    The backbone is held in eval mode permanently: its BatchNorm running statistics come
    from detection pretraining at batch 4, and letting them drift under grounding's batch
    size would discard the calibration the pretraining produced.
    """

    def __init__(
        self,
        ckpt: str | None,
        taps: tuple[str, ...] = DEFAULT_TAPS,
        d_model: int = 256,
        max_hw: tuple[int, int] = (60, 80),
        freeze: bool = True,
    ):
        super().__init__()
        blob = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt else None
        width = blob["width"] if blob else 0.5
        self.backbone = DetectionBackbone(in_channels=2, width=width)
        self.report = {}
        if blob is not None:
            msg = self.backbone.load_state_dict(blob["backbone"], strict=True)
            self.report = {"loaded": len(blob["backbone"]),
                           "det_mAP@0.5": blob.get("mAP@0.5"), "epoch": blob.get("epoch")}
            print(f"[grounding_v2] backbone from {ckpt}: {self.report['loaded']} tensors, "
                  f"detection mAP@0.5 {self.report['det_mAP@0.5']:.4f} "
                  f"(epoch {self.report['epoch']})")
        self.frozen = freeze
        self.backbone.requires_grad_(not freeze)

        self.taps = tuple(taps)
        ch = self.backbone.channels
        self.d_model = d_model
        # trainable: the backbone speaks 128/256 channels, fusion needs d_model, and a
        # random frozen projection in between would be permanent noise
        self.lateral = nn.ModuleDict(
            {t: nn.Conv2d(ch[t], d_model, kernel_size=1) for t in self.taps}
        )
        # learnable 2D position, one table per tap. Without this the linear attention sums
        # position away -- see the module docstring.
        self.pos = nn.ParameterDict(
            {t: nn.Parameter(torch.zeros(1, d_model, *max_hw)) for t in self.taps}
        )
        for p in self.pos.values():
            nn.init.trunc_normal_(p, std=0.02)
        self.level = nn.Parameter(torch.zeros(len(self.taps), d_model))
        nn.init.trunc_normal_(self.level, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            # keep the detection-pretrained BatchNorm statistics; they were calibrated at
            # batch 4 during detection and drift badly under a different batch size
            self.backbone.eval()
        return self

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        # no_grad ONLY when frozen. Guarding this unconditionally would silently starve the
        # backbone of gradient while `requires_grad=True` still counted its 12.6M
        # parameters as trainable -- a fine-tuning run that quietly trains nothing.
        ctx = torch.no_grad() if self.frozen else contextlib.nullcontext()
        with ctx:
            feats = self.backbone(cube)

        out = []
        for i, t in enumerate(self.taps):
            f = feats[t]                                   # (T,B,C,H,W)
            T, B, C, H, W = f.shape
            x = self.lateral[t](f.flatten(0, 1))           # (T*B, d, H, W)
            pos = self.pos[t]
            if pos.shape[-2:] != (H, W):
                pos = nn.functional.interpolate(pos, size=(H, W), mode="bilinear",
                                                align_corners=False)
            x = x + pos + self.level[i].view(1, -1, 1, 1)
            out.append(x.reshape(T, B, self.d_model, H * W).permute(0, 1, 3, 2))
        return torch.cat(out, dim=2)                       # (T, B, N, d_model)


class RgbTokens(nn.Module):
    """SpiLiFormer over the RGB frame -> positional, projected tokens.

    forward(rgb) : (B, 3, H, W) -> (T_out, B, N, d_model), the T axis broadcast because a
    still image is constant over the event window.

    SpiLiFormer (ICCV 2025) is a spiking transformer with lateral inhibition, ImageNet
    85.82% at T=4. Its stem is convolutional and its stages are hierarchical, so
    `forward_features` yields a (T, B, 768, H/16, W/16) SPATIAL map -- at 480x640 a 30x40
    grid, the same resolution as the event backbone's s16 tap, which is why the two streams
    concatenate cleanly.

    What the RGB stream adds that events cannot carry
    -------------------------------------------------
    Talk2Event captions were written from RGB. Colour and appearance ("a white SUV", "a
    bright orange compact car") have no counterpart in a brightness-change stream. Measured
    over 4,929 same-frame caption pairs, only 0.4% are separated by colour ALONE -- so this
    is not the difference between possible and impossible -- but colour appears in a large
    share of captions as corroborating evidence, and the event branch simply cannot see it.

    Only the feedforward pass is used. SpiLiFormer's lateral-inhibition feedback is a
    two-pass design whose second pass exists to sharpen a classification decision; a
    feature extractor wants the features.
    """

    def __init__(
        self,
        ckpt: str | None = None,
        d_model: int = 256,
        T_model: int = 1,
        max_hw: tuple[int, int] = (30, 40),
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
            print(f"[grounding_v2] SpiLiFormer from {ckpt}: "
                  f"{self.report['loaded']}/{len(sd)} tensors "
                  f"(missing {self.report['missing']}, unexpected {self.report['unexpected']})")
        self.frozen = freeze
        self.backbone.requires_grad_(not freeze)

        self.lateral = nn.Conv2d(768, d_model, kernel_size=1)
        self.pos = nn.Parameter(torch.zeros(1, d_model, *max_hw))
        nn.init.trunc_normal_(self.pos, std=pos_std)
        self.d_model = d_model
        # RMS of the pre-position lateral output, refreshed every forward call. Compared
        # against `self.pos`'s own RMS this is the direct test of whether `pos` survives
        # the LIF threshold it is added before: at pos_std=0.02 against features of order
        # 1, pos starts at ~2% of the pre-norm magnitude and a binarising LIF can discard
        # that outright. See `tools/diagnose.py` and item 4 of the coordinate-precision
        # study in docs/research-log.md.
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
        tok = y.reshape(B, self.d_model, H * W).permute(0, 2, 1)  # (B, N, d)
        return tok.unsqueeze(0).expand(T_out, -1, -1, -1)         # (T_out, B, N, d)


class SpikeGroundingV2(nn.Module):
    """forward(cube, input_ids, attention_mask) -> (B, 4) normalised cxcywh.

    One box, in train and eval alike. No anchors, no scores, no argmax.
    """

    def __init__(
        self,
        d_model: int = 256,
        taps: tuple[str, ...] = DEFAULT_TAPS,
        T: int = T_STEPS,
        backbone_ckpt: str | None = None,
        rgb_ckpt: str | None = None,
        use_rgb: bool = False,
        rgb_T: int = 1,
        freeze_vision: bool = True,
        freeze_rgb: bool = False,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        attn_bn_gain: float = 3.0,
        head_hidden: int = 512,
        attn_type: str = "spatial_softmax",
    ):
        super().__init__()
        cmsf = forks.load_cmsf()
        self.T = T
        self.d_model = d_model

        self.vision = VisionTokens(backbone_ckpt, taps, d_model, freeze=freeze_vision)
        # second vision stream: SpiLiFormer over the RGB frame. Optional, because the
        # event-only configuration remains the comparison baseline.
        self.rgb = (RgbTokens(rgb_ckpt, d_model=d_model, T_model=rgb_T, freeze=freeze_rgb)
                    if use_rgb else None)

        with forks.allow_cupy_construction():
            # spike coders: both sides arrive analog, CMSF's blocks expect spike trains
            self.txt_coder = cmsf.RepeatTextEncoder(T, d_model)
            self.vis_norm = nn.LayerNorm(d_model)
            self.vis_lif = cmsf.Dynamic_Threshold_LIFNode(
                tau=2.0, detach_reset=True, backend="cupy")
            if attn_type == "cmsf_linear":
                self.blocks = nn.ModuleList([
                    cmsf.SCA_Block(dim=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                   norm_layer=nn.LayerNorm)
                    for _ in range(depth)])
            elif attn_type == "spatial_softmax":
                self.blocks = nn.ModuleList([
                    SpatialBlock(d_model, num_heads, mlp_ratio) for _ in range(depth)])
            else:
                raise ValueError(f"unknown attn_type {attn_type!r}")
        self.attn_type = attn_type
        forks.use_torch_backend(self)
        self._scale_attention_bn(attn_bn_gain)

        sy = forks.load_spikeyolo()
        self.head_lif1 = sy.mem_update()
        self.fc1 = nn.Linear(d_model, head_hidden)
        self.head_norm = nn.LayerNorm(head_hidden)
        self.head_lif2 = sy.mem_update()
        self.fc2 = nn.Linear(head_hidden, 4)
        # small, never zero: a zero output weight sends exactly zero gradient back through
        # the whole fusion stack (measured previously, cost a full optimiser step)
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def _scale_attention_bn(self, gain: float) -> None:
        """tdBN-style init. At PyTorch's default gain of 1.0 CMSF's attention emits
        exactly zero and the model is silently caption-blind -- measured, see
        `docs/research-log.md` finding 1."""
        if self.attn_type != "cmsf_linear":
            return          # softmax attention normalises its own logits; no dead branch
        for blk in self.blocks:
            for bn in (blk.attn.q_bn, blk.attn.k_bn, blk.attn.v_bn, blk.attn.proj_bn):
                nn.init.constant_(bn.weight, gain)

    def reset(self) -> None:
        """Zero every spikingjelly membrane. Done per-neuron: `reset_net(self)` would call
        this very method on the way down and recurse."""
        for m in self.modules():
            if isinstance(m, sj_neuron.MultiStepLIFNode):
                m.reset()

    def trainable_parameters(self) -> dict[str, int]:
        groups = {
            "vision.lateral": self.vision.lateral,
            "fusion": self.blocks,
            "head": nn.ModuleList([self.fc1, self.head_norm, self.fc2]),
        }
        out = {k: sum(p.numel() for p in m.parameters() if p.requires_grad)
               for k, m in groups.items()}
        out["vision.pos"] = sum(p.numel() for p in self.vision.pos.values()) + self.level_numel
        if self.rgb is not None:
            out["rgb.backbone"] = sum(p.numel() for p in self.rgb.backbone.parameters()
                                      if p.requires_grad)
            out["rgb.lateral+pos"] = (sum(p.numel() for p in self.rgb.lateral.parameters())
                                      + self.rgb.pos.numel())
        return out

    @property
    def level_numel(self) -> int:
        return self.vision.level.numel()

    def forward(self, cube, text_tokens, attention_mask, rgb=None) -> torch.Tensor:
        """cube (T,B,2,H,W); text_tokens (B,L,d_model); attention_mask (B,L);
        rgb (B,3,H,W) when the model was built with `use_rgb=True`."""
        if cube.dim() != 5 or cube.shape[0] != self.T:
            raise ValueError(f"expected (T={self.T}, B, 2, H, W), got {tuple(cube.shape)}")
        if text_tokens.dim() != 3 or text_tokens.shape[-1] != self.d_model:
            raise ValueError(f"expected (B, L, {self.d_model}) text tokens, "
                             f"got {tuple(text_tokens.shape)}")
        if cube.shape[1] != text_tokens.shape[0]:
            raise ValueError(f"batch mismatch {cube.shape[1]} vs {text_tokens.shape[0]}")
        if text_tokens.shape[1] != attention_mask.shape[1]:
            raise ValueError(f"token/mask length mismatch {text_tokens.shape[1]} vs "
                             f"{attention_mask.shape[1]}")

        if (self.rgb is not None) != (rgb is not None):
            raise ValueError("rgb stream and rgb input must both be present or both absent")

        self.reset()
        vis = self.vision(cube)                                   # (T,B,N_ev,d) analog
        if self.rgb is not None:
            # one token sequence over both modalities: the caption attends across events
            # and RGB together, so a phrase can draw on whichever carries its evidence
            vis = torch.cat([vis, self.rgb(rgb, vis.shape[0])], dim=2)

        # spike-code both sides
        v = self.vis_lif(self.vis_norm(vis))                      # (T,B,N,d)
        q = self.txt_coder(text_tokens)                           # (T,B,L,d)
        mask = attention_mask[None, :, :, None].to(q.dtype)
        q = q * mask                                              # padded queries carry nothing

        for blk in self.blocks:
            q = blk(q, v)                                         # Q=text, K=V=vision

        # masked mean over real caption tokens, then over T
        q = q * mask
        pooled = q.sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)    # (T,B,d)
        x = self.head_lif1(pooled)
        x = self.head_norm(self.fc1(x.flatten(0, 1))).reshape(*pooled.shape[:2], -1)
        x = self.head_lif2(x)
        x = self.fc2(x.flatten(0, 1)).reshape(*pooled.shape[:2], 4)
        return x.mean(0).sigmoid()                                # (B,4)

    @torch.no_grad()
    def predict(self, cube, text_tokens, attention_mask, rgb=None,
                amp: bool = True) -> torch.Tensor:
        """(B,4) xyxy in pixels.

        Runs under bf16 autocast by default because that is the dtype the model is
        trained in, and these spiking layers are NOT dtype-agnostic: precision changes
        which membranes cross threshold, so fp32 evaluation of a bf16-trained model is a
        different function. Measured on a same-frame overfit set at step 150 --
        bf16 0.656 IoU vs fp32 0.060 in the same mode.
        """
        was = self.training
        self.eval()
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if amp and cube.is_cuda else contextlib.nullcontext())
        try:
            with ctx:
                b = self(cube, text_tokens, attention_mask, rgb)
            b = b.float()
        finally:
            self.train(was)
        H, W = cube.shape[-2:]
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * \
            b.new_tensor([W, H, W, H])


def build_grounding_v2(**kwargs) -> SpikeGroundingV2:
    return SpikeGroundingV2(**kwargs)
