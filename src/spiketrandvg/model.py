"""RefCOCOGrounding: SpiLiFormer + roberta + text-queried cross-attention -> one box.

    rgb (B,3,H,W)                          input_ids / attention_mask (B,L)
          |                                              |
    VisionEncoder (SNN)                        TextEncoder (ANN, frozen)
    SpiLiFormer, ImageNet 85.82% @ T=4         roberta-base + 0.2M projection
          | /16 -> 24x24 = 576 tokens                    |
    lateral 1x1 + learnable 2D pos                       |
          |  (T,B,576,256)                               |  (B,L,256)
          |                                     RepeatTextEncoder -> (T,B,L,256)
          +------------> SpatialBlock x depth <----------+
                 softmax cross-attention, Q = TEXT, K = V = VISION
                              |  (T,B,L,256)
                    masked mean over real tokens -> spiking MLP
                              |
                       ONE box (B,4) normalised cxcywh

One box, train and eval alike -- no anchors, no scores, no argmax, so nothing to threshold
or NMS at inference.

Why the attention runs text->vision
-----------------------------------
Querying with vision and attending over text would produce 576 query tokens and a head
that has to pool them back down. Querying with TEXT makes the output sequence the caption
itself -- typically 5-10 real tokens -- so the box is read from a language-shaped
representation that has absorbed visual context, and softmax attention becomes affordable
in the process.

Why softmax and not CMSF's linear attention
-------------------------------------------
`cmsf_linear` forms `k^T v` and thereby SUMS OVER EVERY KEY POSITION: the spatial index is
marginalised away and no text query can ask *where* something is. `spatial_softmax`
(default) keeps one weight per vision position, which is exactly the "which location does
this phrase refer to" operation grounding needs. Do not switch to `cmsf_linear` expecting
localisation.

Measured state of this architecture
-----------------------------------
On RefCOCO, trained on the full 120,624-sample train split (`runs/refcoco_b1`), evaluated
once on the untouched test splits:

    testA  mIoU 0.4075   Acc@0.25 71.2%   Acc@0.5 38.9%   Acc@0.75 8.5%
    testB  mIoU 0.3889   Acc@0.25 70.2%   Acc@0.5 33.5%   Acc@0.75 6.3%
    caption_delta +0.26  (against same-image different-object negatives)

It generalises -- val and test agree closely -- and it genuinely reads the caption. What
it does NOT do is localise precisely, and `tools/diagnose.py` traced that to the fusion's
attention map being statistically uniform (perplexity 575.8-576.0 of a 576-key maximum)
even after 199 epochs of dedicated overfitting. See `vision_encoder.py` on the positional
signal, which is the leading candidate mechanism. Acc@0.75 is the metric to watch for any
fix aimed at this.

Two traps, both measured
------------------------
**These layers are not dtype-agnostic.** Precision changes which membranes cross
threshold, so evaluating a bf16-trained model in fp32 is a different function: measured
0.656 IoU vs 0.060 on identical weights. `predict` autocasts to bf16 for that reason;
train under the same dtype.

**`head_norm0` is load-bearing.** `head_lif1` used to be fed `pooled` raw while
`head_lif2` sat after a LayerNorm, so the first neuron's threshold floated with whatever
scale the fusion happened to produce. Head-normaliser choice was separately measured to
move eval IoU between 0.000 and 0.71-0.84 on an identical 8-sample fit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron as sj_neuron
from torchvision.ops import box_iou, complete_box_iou_loss

from spiketrandvg import forks
from spiketrandvg.text_encoder import TextEncoder
from spiketrandvg.vision_encoder import VisionEncoder

__all__ = ["RefCOCOGrounding", "SpatialCrossAttention", "SpatialBlock", "SingleBoxLoss",
           "cxcywh_to_xyxy_norm", "build_model"]


# ---------------------------------------------------------------------------- loss
def cxcywh_to_xyxy_norm(box: torch.Tensor) -> torch.Tensor:
    """(N, 4) normalised cxcywh -> (N, 4) normalised xyxy."""
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class SingleBoxLoss(nn.Module):
    """L = l1_weight * L1 + ciou_weight * (1 - CIoU), the DETR / TransVG single-box objective.

    Both terms are needed and they fail in opposite directions. L1 on normalised cxcywh
    has a stable gradient everywhere, including when the boxes do not overlap at all --
    the entire early-training regime, where IoU is 0 and its gradient carries no
    direction. CIoU matches the metric and, unlike plain IoU, keeps pulling on centre
    distance and aspect ratio once the boxes do overlap. DETR's 5:2 weighting is the
    default for the same reason it is there: L1 on numbers in (0, 1) is small next to an
    IoU term that starts near 1.

    Args:
        center_weight: multiplies the L1 term on cx, cy only (w, h stay at 1.0). Default
            1.0 is the original, unweighted loss. Hypothesis, not an established fix: an
            oracle swap on a separate run attributed nearly all error to centre placement
            (true centre + predicted size -> mIoU 0.4814; predicted centre + true size ->
            only 0.2407). CIoU's own centre-distance term is left unweighted, since it is
            not separable by coordinate the way L1 is.
    """

    def __init__(self, l1_weight: float = 5.0, ciou_weight: float = 2.0,
                 center_weight: float = 1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ciou_weight = ciou_weight
        self.center_weight = center_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.shape != target.shape or pred.shape[-1] != 4:
            raise ValueError(f"expected matching (B, 4) boxes, got {tuple(pred.shape)} "
                             f"and {tuple(target.shape)}")
        diff = (pred - target).abs()
        l1 = diff.mean()                  # reported unweighted, for cross-run comparability
        if self.center_weight != 1.0:
            w = diff.new_tensor([self.center_weight, self.center_weight, 1.0, 1.0])
            l1_train = (diff * w).mean()
        else:
            l1_train = l1

        p_xyxy = cxcywh_to_xyxy_norm(pred)
        t_xyxy = cxcywh_to_xyxy_norm(target)
        ciou = complete_box_iou_loss(p_xyxy, t_xyxy, reduction="mean")
        with torch.no_grad():
            iou = box_iou(p_xyxy, t_xyxy).diagonal().mean()

        total = self.l1_weight * l1_train + self.ciou_weight * ciou
        return total, {"l1": l1.detach(), "ciou": ciou.detach(), "iou": iou}


# ------------------------------------------------------------------------- attention
class SpatialCrossAttention(nn.Module):
    """Softmax cross-attention over the spatial axis. Q = text, K = V = vision.

    Cost is O(L_q * N_kv), which is why CMSF avoids it in general -- but here the queries
    are the CAPTION, so a handful of tokens against 576 vision positions is nothing.
    Reversing the attention direction is what makes softmax affordable.

    Spike discipline: q/k/v projections are BN + LIF exactly as in CMSF, so those matmuls
    are spike-driven. The attention product itself is analog -- the honest cost of being
    able to localise, and it is one matmul, not the bulk of the compute.
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
        y = lin(x.flatten(0, 1))
        y = bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D)
        return lif(y)

    def forward(self, query, key, value, key_mask=None, return_attn: bool = False):
        """query (T,B,Lq,D) text; key/value (T,B,N,D) vision; key_mask (B,N) or None.

        `return_attn=True` additionally returns the post-softmax weights (T,B,h,Lq,N) --
        the one tensor here that still carries the spatial index before `proj_lif`
        binarises the output. Consumed by the `attn_softargmax` head and by
        `tools/diagnose.py`'s perplexity measurement.
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
    """SpatialCrossAttention + CMSF's spiking gated MLP, residual."""

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


# ----------------------------------------------------------------------------- model
class RefCOCOGrounding(nn.Module):
    """forward(rgb, input_ids, attention_mask) -> (B, 4) normalised cxcywh.

    Args:
        rgb_ckpt: SpiLiFormer ImageNet checkpoint for the vision encoder.
        text_model: any HF encoder; must match the tokenizer the dataloader used.
        img_size: (H, W) the dataloader emits. Sizes the positional table at the /16 grid
            so it is exact rather than interpolated. Must be divisible by 16.
        T: spiking timesteps for the fusion stack. A still image has no temporal axis, so
            T here is purely simulation length -- but it still matters, because LIF
            membranes integrate across timesteps and T=1 degenerates every neuron into a
            plain threshold. Default 4, matching the SpiLiFormer checkpoint. Cost is
            linear in T.
        rgb_T: SpiLiFormer's own internal timesteps (see `VisionEncoder`).
        freeze_rgb / freeze_text: default trains vision, freezes text. On Talk2Event,
            freezing BOTH produced a caption-blind model over 85 epochs (delta +0.0009)
            while unfreezing the vision side gave +0.051 within two.
        depth: SpatialBlocks. Each is cross-attention + a spiking gated MLP.
        attn_type: "spatial_softmax" (default) or "cmsf_linear" -- see module docstring.
        head_type: "pooled_mlp" (default) or "attn_softargmax". MEASURED: on a 100-sample
            200-epoch A/B, `attn_softargmax` collapsed to a near-constant box (pred std
            0.0002 on cx/cy against gt std 0.14-0.23) and never beat its own epoch-0 val
            mIoU, because the attention map it reads is statistically uniform. Do not
            treat it as the better option; it is retained because the underlying
            quantisation argument is untested rather than refuted, and it becomes
            testable the moment the map carries location.
        attn_map: which block's map `attn_softargmax` reads -- "last" or "mean".
        pos_std: init std of the vision positional embedding (see `VisionEncoder`).
        text_unfreeze_last: unfreeze only the last N roberta layers (see `TextEncoder`).
    """

    def __init__(
        self,
        rgb_ckpt: str | None = None,
        text_model: str = "roberta-base",
        d_model: int = 256,
        img_size: tuple[int, int] = (384, 384),
        T: int = 4,
        rgb_T: int = 1,
        freeze_rgb: bool = False,
        freeze_text: bool = True,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        head_hidden: int = 512,
        attn_type: str = "spatial_softmax",
        attn_bn_gain: float = 3.0,
        head_type: str = "pooled_mlp",
        attn_map: str = "last",
        pos_std: float = 0.02,
        text_unfreeze_last: int = 0,
    ):
        super().__init__()
        if img_size[0] % 16 or img_size[1] % 16:
            raise ValueError(f"img_size {img_size} must be divisible by 16")
        if d_model % num_heads:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        if head_type not in ("pooled_mlp", "attn_softargmax"):
            raise ValueError(f"unknown head_type {head_type!r}")
        if attn_map not in ("last", "mean"):
            raise ValueError(f"unknown attn_map {attn_map!r}")
        cmsf = forks.load_cmsf()
        self.T = T
        self.d_model = d_model
        self.img_size = tuple(img_size)
        self.head_type = head_type
        self.attn_map = attn_map

        self.vision = VisionEncoder(
            rgb_ckpt, d_model=d_model, T_model=rgb_T,
            max_hw=(img_size[0] // 16, img_size[1] // 16), freeze=freeze_rgb,
            pos_std=pos_std,
        )
        self._disable_unused_backbone_grads()
        self.text = TextEncoder(text_model, d_model=d_model, freeze=freeze_text,
                                unfreeze_last=text_unfreeze_last)

        # spike coders: both sides arrive analog, CMSF's blocks want spike trains
        with forks.allow_cupy_construction():
            self.txt_coder = cmsf.RepeatTextEncoder(T, d_model)
            self.vis_norm = nn.LayerNorm(d_model)
            self.vis_lif = cmsf.Dynamic_Threshold_LIFNode(
                tau=2.0, detach_reset=True, backend="cupy")
            if attn_type == "spatial_softmax":
                self.blocks = nn.ModuleList(
                    [SpatialBlock(d_model, num_heads, mlp_ratio) for _ in range(depth)])
            elif attn_type == "cmsf_linear":
                self.blocks = nn.ModuleList([
                    cmsf.SCA_Block(dim=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                   norm_layer=nn.LayerNorm)
                    for _ in range(depth)])
            else:
                raise ValueError(f"unknown attn_type {attn_type!r}")
        self.attn_type = attn_type
        forks.use_torch_backend(self)
        self._scale_attention_bn(attn_bn_gain)

        # box head. head_norm0 is applied unconditionally -- see the module docstring.
        sy = forks.load_spikeyolo()
        self.head_norm0 = nn.LayerNorm(d_model)
        self.head_lif1 = sy.mem_update()
        self.fc1 = nn.Linear(d_model, head_hidden)
        self.head_norm = nn.LayerNorm(head_hidden)
        self.head_lif2 = sy.mem_update()
        self.fc2 = nn.Linear(head_hidden, 4)
        # small, never zero: a zero output weight sends exactly zero gradient back through
        # the entire fusion stack, which cost a full optimiser run to diagnose once
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

        if head_type == "attn_softargmax":
            # log space so exp() is positive and 0 is a neutral init (exp(0) = 1)
            self.attn_logit_scale = nn.Parameter(torch.zeros(1))

    def _disable_unused_backbone_grads(self) -> None:
        """Turn off grad for the SpiLiFormer parts `forward_features` never reaches.

        MEASURED: with the backbone unfrozen, 25 of its 256 tensors -- 4.91M parameters --
        came back with `grad is None` after a full backward. They are the ImageNet
        classifier `head` (0.77M) and the lateral-inhibition `decoder` feedback path with
        its two `prompt` vectors (4.13M). Left on, they would be reported as trainable,
        carry AdamW's two moment buffers each (~39MB of optimiser state), and never move.
        """
        bb = self.vision.backbone
        for name in ("head", "decoder", "prompt", "prompt_2"):
            mod = getattr(bb, name, None)
            if isinstance(mod, (nn.Module, nn.Parameter)):
                mod.requires_grad_(False)

    def _scale_attention_bn(self, gain: float) -> None:
        """tdBN-style init for CMSF's linear attention.

        At PyTorch's default BatchNorm gain of 1.0 that branch emits exactly zero -- the
        residual collapses to the identity and the model ignores the caption entirely,
        silently. Measured: gain 1.0 -> 0.0% attention firing and caption sensitivity
        0.0000; gain 3.0 -> 26.3% and 1.0816. Softmax attention normalises its own logits
        and cannot die this way, so it is skipped.
        """
        if self.attn_type != "cmsf_linear":
            return
        for blk in self.blocks:
            for bn in (blk.attn.q_bn, blk.attn.k_bn, blk.attn.v_bn, blk.attn.proj_bn):
                nn.init.constant_(bn.weight, gain)

    def reset(self) -> None:
        """Zero every spikingjelly membrane. Per-neuron: `reset_net(self)` would call this
        very method on the way down and recurse."""
        for m in self.modules():
            if isinstance(m, sj_neuron.BaseNode):
                m.reset()

    def trainable_parameters(self) -> dict[str, int]:
        """Trainable parameter count per group, for the training banner."""
        out = {
            "vision.backbone": sum(p.numel() for p in self.vision.backbone.parameters()
                                   if p.requires_grad),
            "vision.lateral+pos": (sum(p.numel() for p in self.vision.lateral.parameters()
                                       if p.requires_grad)
                                   + (self.vision.pos.numel()
                                      if self.vision.pos.requires_grad else 0)),
            "text.encoder": sum(p.numel() for p in self.text.encoder.parameters()
                                if p.requires_grad),
            "text.proj": sum(p.numel() for p in self.text.proj.parameters()
                             if p.requires_grad),
            "fusion": sum(p.numel() for p in self.blocks.parameters() if p.requires_grad),
            "head": sum(p.numel()
                        for m in (self.head_norm0, self.fc1, self.head_norm, self.fc2)
                        for p in m.parameters() if p.requires_grad)
                    + (self.attn_logit_scale.numel()
                       if self.head_type == "attn_softargmax" else 0),
        }
        return {k: v for k, v in out.items() if v}

    def _softargmax_box(self, raw, attn_maps, attention_mask) -> torch.Tensor:
        """Read the box centre off the attention map, size off the MLP.

        `raw` is (T,B,4) from the same `fc2` as `pooled_mlp`; only its first two columns
        are used differently -- as a bounded offset onto a softargmax centre rather than
        the centre itself. Columns 2:4 are read identically in both heads.

        MUST run in fp32. Everything upstream stays in whatever dtype it was trained in --
        these spiking layers are not dtype-agnostic -- but this block crosses no threshold,
        and bf16's 8-bit mantissa would quantise the very centre estimate it exists to
        recover.
        """
        chosen = (attn_maps[-1] if self.attn_map == "last"
                  else torch.stack(attn_maps, 0).mean(0))
        a = chosen.float().mean(dim=(0, 2))                    # (B,Lq,N) over T and heads
        w = attention_mask.to(a.dtype).unsqueeze(-1)
        heat = (a * w).sum(1) / w.sum(1).clamp(min=1.0)        # (B,N) real tokens only

        Hg, Wg = self.img_size[0] // 16, self.img_size[1] // 16
        if heat.shape[-1] != Hg * Wg:
            raise RuntimeError(f"{heat.shape[-1]} keys but grid is {Hg}x{Wg} "
                               f"({Hg * Wg}) -- a second vision stream would break the "
                               f"row-major index this assumes")
        p = (heat * self.attn_logit_scale.exp()).softmax(-1).view(-1, Hg, Wg)
        # token index n = row * Wg + col (VisionEncoder flattens (H,W) row-major), so
        # p.sum(1) marginalises OUT rows -> the column distribution, paired with `xs`;
        # p.sum(2) -> the row distribution, paired with `ys`. Getting this backwards
        # still converges to a plausible loss, with the grid silently transposed.
        xs = torch.linspace(0.5 / Wg, 1 - 0.5 / Wg, Wg, device=p.device, dtype=p.dtype)
        ys = torch.linspace(0.5 / Hg, 1 - 0.5 / Hg, Hg, device=p.device, dtype=p.dtype)
        cx = (p.sum(1) * xs).sum(-1)
        cy = (p.sum(2) * ys).sum(-1)

        r = raw.mean(0).float()                                # (B,4), collapse T
        cx = (cx + torch.tanh(r[:, 0]) / Wg).clamp(0, 1)
        cy = (cy + torch.tanh(r[:, 1]) / Hg).clamp(0, 1)
        return torch.stack([cx, cy, r[:, 2].sigmoid(), r[:, 3].sigmoid()], dim=-1)

    def forward(self, rgb, input_ids, attention_mask, return_diagnostics: bool = False):
        """rgb (B,3,H,W); input_ids (B,L); attention_mask (B,L) -> (B,4) cxcywh in [0,1].

        `return_diagnostics=True` also returns the attention maps and pooled vector for
        `tools/diagnose.py`. It costs nothing when False -- maps are only requested from
        the blocks when the flag or the head actually needs them.
        """
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(f"expected (B,3,H,W) rgb, got {tuple(rgb.shape)}")
        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"ids {tuple(input_ids.shape)} != mask "
                             f"{tuple(attention_mask.shape)}")
        if rgb.shape[0] != input_ids.shape[0]:
            raise ValueError(f"batch mismatch {rgb.shape[0]} vs {input_ids.shape[0]}")

        self.reset()

        vis = self.vision(rgb, self.T)                       # (T,B,N,d) analog
        v = self.vis_lif(self.vis_norm(vis))                 # (T,B,N,d) spikes

        txt = self.text(input_ids, attention_mask)           # (B,L,d) analog
        q = self.txt_coder(txt)                              # (T,B,L,d) spikes
        mask = attention_mask[None, :, :, None].to(q.dtype)
        q = q * mask                                         # padded queries carry nothing

        need_attn = self.head_type == "attn_softargmax" or return_diagnostics
        attn_maps = []
        for blk in self.blocks:
            if need_attn:
                q, a = blk(q, v, return_attn=True)
                attn_maps.append(a)
            else:
                q = blk(q, v)                                # Q=text, K=V=vision

        # masked mean over the real caption tokens, then over T
        q = q * mask
        pooled = q.sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)   # (T,B,d)
        x = self.head_lif1(self.head_norm0(pooled))
        x = self.head_norm(self.fc1(x.flatten(0, 1))).reshape(*pooled.shape[:2], -1)
        x = self.head_lif2(x)
        raw = self.fc2(x.flatten(0, 1)).reshape(*pooled.shape[:2], 4)   # (T,B,4)

        if self.head_type == "pooled_mlp":
            box = raw.mean(0).sigmoid()
        else:
            box = self._softargmax_box(raw, attn_maps, attention_mask)

        if return_diagnostics:
            return box, {"attn_maps": attn_maps, "pooled": pooled, "vis_tokens": v}
        return box

    @torch.no_grad()
    def predict(self, rgb, input_ids, attention_mask, amp: bool = True) -> torch.Tensor:
        """Eval-mode boxes, (B,4) normalised cxcywh.

        Autocasts to bf16 because that is the dtype these layers are trained in and they
        are NOT dtype-agnostic. Measured on a same-frame overfit set: bf16 0.656 IoU vs
        fp32 0.060, identical weights and mode. Pass `amp=False` only to reproduce that.
        """
        was = self.training
        self.eval()
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if amp and rgb.is_cuda
               else torch.autocast("cpu", enabled=False))
        with ctx:
            out = self(rgb, input_ids, attention_mask).float()
        self.train(was)
        return out


def build_model(**kwargs) -> RefCOCOGrounding:
    return RefCOCOGrounding(**kwargs)
