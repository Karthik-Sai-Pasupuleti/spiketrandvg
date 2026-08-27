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

from spiketrandvg import utils as forks
from spiketrandvg.textencoder import TextEncoder
from spiketrandvg.visionencoder import VisionEncoder

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

    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0,
                 scale: float | None = None, qk_lif: str = "binary"):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        cmsf = forks.load_cmsf()
        self.h = num_heads
        self.dh = dim // num_heads
        # Softmax temperature. `dh ** -0.5` is the transformer default and it is derived
        # for ANALOG q/k: unit-variance Gaussian entries make q.k have variance dh, so
        # dividing by sqrt(dh) restores an O(1) logit spread. Here q and k are BINARY,
        # and the derivation gives the wrong answer by more than an order of magnitude.
        #
        # MEASURED on this model at init (480x640, taps s8+s16, N=6000 keys): q fires at
        # 12.3%, k at 14.5%. Per head, q.k is a sum of dh=32 Bernoulli(0.123*0.145)
        # terms, so it has mean 0.57 and standard deviation 0.75 counts. At the default
        # scale that is a logit spread of 0.13 -- and a softmax over 6000 keys whose
        # logits differ by 0.13 IS uniform. Predicted perplexity N*exp(-sigma^2) = 5896;
        # measured 5980 of 6000. The near-uniform attention map is not a training
        # failure or an initialisation accident, it is arithmetic.
        #
        # Sharpening needs a logit spread of order log(N) ~ 8.7, i.e. a scale tens of
        # times larger. `scale` overrides the default for exactly that experiment.
        self.scale = self.dh ** -0.5 if scale is None else float(scale)

        self.q_linear, self.k_linear, self.v_linear = (nn.Linear(dim, dim) for _ in range(3))
        self.q_bn, self.k_bn, self.v_bn = (nn.BatchNorm1d(dim) for _ in range(3))
        with forks.allow_cupy_construction():
            self.q_lif, self.k_lif, self.v_lif = (
                cmsf.Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend="cupy")
                for _ in range(3))
            self.proj_lif = cmsf.Dynamic_Threshold_LIFNode(
                tau=2.0, detach_reset=True, backend="cupy")
        if qk_lif not in ("binary", "ilif"):
            raise ValueError(f"unknown qk_lif {qk_lif!r}")
        self.qk_lif = qk_lif
        if qk_lif == "ilif":
            # Integer I-LIF on the two projections that FORM the logits. This is not a
            # relaxation of the spiking constraint: `mem_update` is a real LIF with a
            # soft reset that quantises to {0,1,2,3,4} instead of {0,1}, and the event
            # encoder already runs on it (`--no-ilif` is the ablation against it). The
            # attention projections were the one place still binary.
            #
            # Why it is the right knob. A softmax over N keys can only be non-uniform if
            # the logits SPREAD over order log(N). With binary q,k the logit is
            # (q.k)/sqrt(dh), an integer overlap count out of dh=32 -- and MEASURED on
            # probe_00/best.pth, raising the temperature stops helping at perplexity 548
            # of 6000 no matter how far it is raised, because the softmax has become a
            # hard max over the ~548 keys TIED at the maximum overlap. That floor
            # belongs to the binary code, not to the temperature.
            #
            # Five levels per unit instead of two make q.k range over 0..512 and make
            # exact ties rare. MEASURED on the same checkpoint, same 127 val samples:
            #
            #   q/k       scale    logit sd   perplexity
            #   binary   0.1768       0.529       4984.6
            #   binary   4.0000      12.011        679.2   (floor at 548)
            #   I-LIF    0.1768       7.108        679.5
            #   I-LIF    1.0000      39.583        123.3
            #
            # At the SAME scale the map is 7x sharper, and unlike the binary code it
            # keeps sharpening past the floor.
            sy_mem_update, _ = forks.load_ilif()
            self.q_lif, self.k_lif = sy_mem_update(), sy_mem_update()
        forks.use_torch_backend(self)
        self.proj = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        # Diagnostics, off by default so the training path pays nothing. When on,
        # `last_rates` holds the fraction of q and k units that fired -- a dead
        # projection (<1%) makes the attention map meaningless before any other
        # explanation is worth considering. Kept as GPU tensors: reading a Python float
        # here would force a device sync on every forward.
        self.collect_stats = False
        self.last_rates: dict[str, torch.Tensor] = {}

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

        if self.collect_stats:
            self.last_rates = {"q_rate": (q.detach() > 0).float().mean(),
                               "k_rate": (k.detach() > 0).float().mean()}

        q = q.reshape(T, B, Lq, self.h, self.dh).permute(0, 1, 3, 2, 4)   # (T,B,h,Lq,dh)
        k = k.reshape(T, B, N, self.h, self.dh).permute(0, 1, 3, 2, 4)
        v = v.reshape(T, B, N, self.h, self.dh).permute(0, 1, 3, 2, 4)

        logits = (q @ k.transpose(-2, -1)) * self.scale                   # (T,B,h,Lq,N)
        if self.collect_stats:
            # spread of the logits ACROSS KEYS, per query -- the quantity that decides
            # whether a softmax over N keys can be anything but uniform. Needs to reach
            # order log(N) before the map carries location.
            self.last_rates["logit_std"] = logits.detach().float().std(dim=-1).mean()
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

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.0,
                 attn_scale: float | None = None, qk_lif: str = "binary"):
        super().__init__()
        cmsf = forks.load_cmsf()
        self.attn = SpatialCrossAttention(dim, num_heads, scale=attn_scale,
                                          qk_lif=qk_lif)
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
        event_backbone: "metaspikformer" (default, ~54.7M, ImageNet-pretrained with its
            RGB stem averaged onto the 2 polarity channels) or "spiliformer_dvs"
            (SpiLiFormer's CIFAR10-DVS variant, event-native `in_channels=2` by design).
            MEASURED trade-off: the DVS variant is 1.70M parameters -- 32x smaller -- and
            the authors publish NO CIFAR10-DVS checkpoint, so it starts from random init.
            It is the honest event-native architecture; it is not the stronger starting
            point. Both expose the same tap geometry at 480x640 (s8 -> 60x80, s16 ->
            30x40), so they are a clean single-flag A/B.
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


# ======================================================================================
# Talk2Event: event-only, language-conditioned, slot-based box selection
# ======================================================================================

class SlotBoxHead(nn.Module):
    """Four coordinate tokens, each a V-way choice plus a within-slot residual.

        forward(tokens) -> (box (B,4) normalised cxcywh, logits (B,4,V))

    Why choose instead of regress
    -----------------------------
    This is the Gate-1 finding turned into an architecture. A binary/integer spiking
    encoder cannot hand a regression head a smooth field to interpolate a sub-pixel
    coordinate out of -- the RefCOCO study measured the consequence directly: Acc@0.25
    71% but Acc@0.75 only 8%, i.e. roughly the right region, rarely the right box. A
    spike count picking a winner from a list is a far better-posed problem for this kind
    of encoder than asking it to emit a precise real number.

    Predicting in the ANNOTATION frame, not the feature frame
    ---------------------------------------------------------
    The V slots discretise the normalised [0,1] coordinate range, which maps to the
    480x640 annotation frame -- NOT to the stride-16 feature grid. At V=1000 a slot is
    0.64 px wide in x and 0.48 px in y, so the feature map's 30x40 resolution never caps
    achievable precision. That is the whole point of decoupling them; a head that argmaxes
    over feature cells would be stuck at 16 px granularity.

    The residual
    ------------
    `expectation` (default) reads the coordinate as the softmax-weighted mean of slot
    centres, which is differentiable everywhere and already sub-slot. The residual head
    then adds a tanh-bounded correction of at most one slot width. Hard argmax is
    available for inference-time reporting but is not differentiable and is not the
    training path.
    """

    def __init__(self, d_model: int, n_slots: int = 1000, hidden: int = 512):
        super().__init__()
        self.n_slots = n_slots
        # one head per coordinate: cx, cy, w, h
        self.slot_head = nn.Linear(d_model, 4 * n_slots)
        self.res_head = nn.Linear(d_model, 4)
        # slot centres in [0,1]; a buffer so it moves with .to(device) and is saved
        self.register_buffer(
            "centres", (torch.arange(n_slots, dtype=torch.float32) + 0.5) / n_slots)
        nn.init.normal_(self.slot_head.weight, std=1e-3)
        nn.init.zeros_(self.slot_head.bias)
        nn.init.normal_(self.res_head.weight, std=1e-3)
        nn.init.zeros_(self.res_head.bias)

    def forward(self, x: torch.Tensor, hard: bool = False,
                prior: torch.Tensor | None = None):
        """x (B, d) pooled real-valued features -> box (B,4), logits (B,4,V).

        `prior` (B,2,V) is an additive log-prior on the cx and cy slot logits -- the
        attention map's own opinion about where the referent is, resampled onto the slot
        grid. It is added in LOG space because that is what a softmax consumes: adding
        log p multiplies the head's posterior by p, which is Bayes rather than a
        heuristic blend. w and h get no prior; the map says where, not how big.
        """
        B = x.shape[0]
        logits = self.slot_head(x).view(B, 4, self.n_slots)
        if prior is not None:
            logits = torch.cat([logits[:, :2] + prior, logits[:, 2:]], dim=1)
        p = logits.float().softmax(-1)
        if hard:
            coord = self.centres[p.argmax(-1)]                     # (B,4)
        else:
            coord = (p * self.centres).sum(-1)                     # (B,4) expectation
        res = torch.tanh(self.res_head(x).float()) / self.n_slots  # <= one slot
        box = (coord + res).clamp(0, 1)
        # w and h must be positive and are meaningless at 0; clamp to one slot minimum
        wh = box[:, 2:].clamp(min=1.0 / self.n_slots)
        return torch.cat([box[:, :2], wh], dim=-1), logits


class Talk2EventGrounding(nn.Module):
    """Event-only spike-driven referring grounding.

        forward(cube, input_ids, attention_mask)
            -> {"box": (B,4), "slot_logits": (B,4,V), "tag_logits": (B,L,5)}

        events (T=5,B,2,480,640)                caption ids (B,L)
                 |                                     |
                 |                          TextEncoder (roberta, FROZEN)
                 |                                     |
                 |                          AttributeQueryTagger -> 4 sub-queries
                 |                                     |
                 |  <--- ThresholdModulator sets firing thresholds, stages 2-4
                 v
          EventEncoder (SpiLiFormer-DVS, T=5 real timesteps, I-LIF)
                 |
            ACCUMULATE over T   <-- binary/integer ends here, real numbers resume
                 |
          lateral 1x1 + learnable 2D position -> vision tokens
                 |
          SpatialCrossAttention (Q = the 4 sub-queries)  x depth
                 |
          SlotBoxHead: 4 coordinate tokens over V=1000 slots + residual
                 |
             ONE box (B,4) normalised cxcywh

    Why there is no RGB branch here
    -------------------------------
    Deliberate, and the single most important design decision in this class. The novelty
    claim is spike-driven referring grounding *on event streams*. Talk2Event's own
    baselines are frame-only 55.47 vs event-only 31.96 mAcc, so a fused model's headline
    number is mostly the frame encoder, and the contribution disappears underneath it.
    The per-attribute analysis this architecture exists to support also only means
    anything if event-only and frame-only are separate rows -- fusion is a third row, not
    the headline.

    When fusion IS added for completeness against the benchmark's third column, the RGB
    branch should be an **ANN, not spiking**. The spiking claim concerns the event
    pathway, where the sensor is. Spiking an RGB branch buys nothing scientifically,
    doubles training memory, and adds a confound to every ablation.
    `vision_encoder.VisionEncoder` remains in the codebase for exactly that row.
    """

    def __init__(
        self,
        event_ckpt: str | None = None,
        text_model: str = "roberta-base",
        d_model: int = 256,
        img_size: tuple[int, int] = (480, 640),
        T: int = 5,
        taps: tuple[str, ...] = ("s8", "s16"),
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        n_slots: int = 1000,
        ilif: bool = True,
        condition_encoder: bool = True,
        freeze_event: bool = False,
        freeze_text: bool = True,
        text_unfreeze_last: int = 0,
        max_log_gain: float = 0.5,
        attn_bn_gain: float = 3.0,
        event_backbone: str = "spiliformer_dvs",
        pos_std: float = 0.02,
        attn_scale: float | None = None,
        qk_lif: str = "ilif",
        attn_prior: bool = False,
        attn_prior_gain: float = 0.0,
        attn_prior_eps: float = 0.01,
        query_weights: bool = False,
        pos_ratio: float | None = None,
        return_map: bool = False,
    ):
        super().__init__()
        from spiketrandvg.visionencoder import EventEncoder, ThresholdModulator
        from spiketrandvg.textencoder import ATTRIBUTES, AttributeQueryTagger

        if img_size[0] % 16 or img_size[1] % 16:
            raise ValueError(f"img_size {img_size} must be divisible by 16")
        self.T = T
        self.d_model = d_model
        self.img_size = tuple(img_size)
        self.taps = tuple(taps)
        self.n_attr = len(ATTRIBUTES)

        self.events = EventEncoder(ckpt_path=event_ckpt, taps=self.taps, in_channels=2,
                                   ilif=ilif, freeze=freeze_event,
                                   backbone=event_backbone, img_size=self.img_size)
        self.text = TextEncoder(text_model, d_model=d_model, freeze=freeze_text,
                                unfreeze_last=text_unfreeze_last)
        self.tagger = AttributeQueryTagger(d_model=d_model, n_attr=self.n_attr)

        self.condition_encoder = condition_encoder
        self.modulator = (ThresholdModulator(d_model, self.events.stage_channels,
                                             max_log_gain=max_log_gain)
                          if condition_encoder else None)

        # post-accumulator projections: the tensors here are REAL-VALUED spike counts
        ch = self.events.out_channels
        self.lateral = nn.ModuleDict(
            {t: nn.Conv2d(ch[t], d_model, kernel_size=1) for t in self.taps})
        self.tap_norm = nn.ModuleDict({t: nn.BatchNorm2d(ch[t]) for t in self.taps})
        self.pos = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(
                1, d_model, img_size[0] // self.events.strides[t],
                img_size[1] // self.events.strides[t]))
            for t in self.taps})
        for p in self.pos.values():
            nn.init.trunc_normal_(p, std=pos_std)
        self.level = nn.Parameter(torch.zeros(len(self.taps), d_model))
        nn.init.trunc_normal_(self.level, std=0.02)

        # the head still cross-attends -- conditioning the encoder ADDS a pathway, it
        # does not replace this one
        self.blocks = nn.ModuleList(
            [SpatialBlock(d_model, num_heads, mlp_ratio, attn_scale=attn_scale,
                          qk_lif=qk_lif) for _ in range(depth)])
        self._revive_attention_bn(attn_bn_gain)
        self.q_norm = nn.LayerNorm(d_model)
        self.v_norm = nn.LayerNorm(d_model)
        self.head_norm = nn.LayerNorm(d_model)
        self.box_head = SlotBoxHead(d_model, n_slots=n_slots)

        # Two leading indicators, measured in eval only (see `set_collect_stats`). They
        # move before accuracy does, which is the whole reason they are here:
        #   attn_perplexity  effective number of vision positions a query attends.
        #                    Equal to the key count = uniform = the map carries no
        #                    location at all, whatever the loss says.
        #   pos_rms_ratio    RMS(positional table) / RMS(lateral output it is added to).
        #                    Below ~0.05 position is a sub-1% perturbation on features
        #                    that then cross a firing threshold, so it never survives
        #                    into the attention logits.
        self.collect_stats = False
        self.stats: dict[str, torch.Tensor] = {}

        # Zero-init, so training starts at EXACTLY the unmodified model and the prior can
        # only be adopted if it earns gradient. Safe to zero here, unlike a head's output
        # weight: it multiplies a log-density that is an input, not a matmul whose
        # backward it would kill.
        self.attn_prior = attn_prior
        self.attn_prior_eps = attn_prior_eps
        # zero-init -> softmax -> uniform -> exactly the plain mean at step 0
        self.query_logits = (nn.Parameter(torch.zeros(self.n_attr))
                             if query_weights else None)
        if attn_prior:
            self.attn_prior_gain = nn.Parameter(torch.full((1,), attn_prior_gain))

        # Pin RMS(pos) to a fixed fraction of RMS(lateral) on every forward, instead of
        # letting it be whatever a fixed init std happens to leave behind.
        #
        # MEASURED on probe_00, the ratio is not a constant and drifts on its own: 0.039
        # at init, 0.018 after one epoch as the lateral output grows while the table sits
        # still, 0.034 by epoch 5. `--pos-std` sets it only at step 0 and then loses
        # control of it. Rescaling per forward makes it an actual hyperparameter.
        #
        # The scale factor is DETACHED, so gradient shapes the positional PATTERN while
        # its amplitude stays pinned. Without that the optimiser can shrink the table and
        # the rescale silently undoes the shrink, which is a fight, not a control.
        self.pos_ratio = pos_ratio
        # hand the last block's attention map back to the trainer, so it can be
        # supervised directly (see `attn_box_mass`)
        self.return_map = return_map

    def _revive_attention_bn(self, gain: float) -> None:
        """tdBN-style init on the cross-attention BatchNorms. NOT optional here.

        MEASURED on this model: at PyTorch's default BN gain of 1.0, `q_lif` and
        `proj_lif` fire at exactly 0.00% and stay there. With `proj_lif` dead the block's
        residual `x + attn(x, y, y)` collapses to `x`, so the vision branch contributes
        nothing and `lateral`/`ThresholdModulator` receive exactly zero gradient. A
        41-step fit on 2 samples reached IoU 0.9964 in that state -- entirely through the
        language path, memorising the caption. Both the conditioning pathway and the
        event encoder were inert while the loss looked excellent.

        This is research-log finding 1 in a place its original guard did not reach:
        `RefCOCOGrounding._scale_attention_bn` skips `spatial_softmax` on the reasoning
        that softmax cannot die. The softmax indeed cannot, but the LIFs around it can.
        RefCOCO survived it because 7539 steps/epoch let BatchNorm running statistics
        drift the pre-activations over threshold eventually (measured 7-40% firing by
        epoch 18); a small event-side run has no such runway and settles into the
        language-only solution first.

        RefCOCOGrounding's behaviour is deliberately NOT changed -- `runs/refcoco_b1` and
        the in-flight full run were trained without this, and altering it would break
        comparability.
        """
        for blk in self.blocks:
            for bn in (blk.attn.q_bn, blk.attn.k_bn, blk.attn.v_bn, blk.attn.proj_bn):
                nn.init.constant_(bn.weight, gain)

    def trainable_parameters(self) -> dict[str, int]:
        g = {
            "events.backbone": self.events.backbone,
            "text.encoder": self.text.encoder,
            "text.proj": self.text.proj,
            "tagger": self.tagger,
            "lateral+norm": nn.ModuleList([self.lateral, self.tap_norm]),
            "fusion": self.blocks,
            "box_head": self.box_head,
        }
        out = {k: sum(p.numel() for p in m.parameters() if p.requires_grad)
               for k, m in g.items()}
        if self.modulator is not None:
            out["modulator"] = sum(p.numel() for p in self.modulator.parameters()
                                   if p.requires_grad)
        out["pos+level"] = (sum(p.numel() for p in self.pos.values())
                            + self.level.numel())
        return {k: v for k, v in out.items() if v}

    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, sj_neuron.BaseNode):
                m.reset()

    def _pooled_map(self, attn: torch.Tensor) -> torch.Tensor:
        """(T,B,h,Lq,N) -> (B,N), pooling T, heads and the four attribute queries.

        The queries are not interchangeable. Section 16 of the research log measures the
        caption reduced to one attribute group at a time: `relation_viewer` alone reaches
        mIoU 0.1507 while `appearance` (0.0501) and `status` (0.0532) sit at the trivial
        floor. Averaging the four maps equally therefore dilutes the one that knows where
        the referent is with three that do not.

        `query_logits` is zero-initialised, so the pooling starts as the plain mean and
        any departure from it has to be earned. Both the supervision and the prior read
        the map through here, so they always agree about what "the map" is.
        """
        a = attn.float().mean(dim=(0, 2))                        # (B,Lq,N) over T, heads
        if self.query_logits is None:
            return a.mean(1)
        w = self.query_logits.softmax(0)[None, :a.shape[1], None]
        return (a * w).sum(1) / w.sum(1).clamp_min(1e-8)

    def attn_box_mass(self, attn: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Fraction of the attention map's mass that lands inside the true box. (B,)

        The map is the only tensor in this model indexed by spatial position, and
        NOTHING currently supervises where it points. The box loss reaches it only
        through `attn @ v -> proj_lif`, which is binary, so the gradient that survives
        says almost nothing about location. `-log(mass in box)` is the direct statement
        of the label we already have: the referent is HERE.

        A cell counts as inside if its centre is inside the box, plus the cell
        containing the box centre unconditionally -- a box narrower than one cell
        (the 10th percentile object is 38 px against a 16 px s16 cell) can otherwise
        contain no cell centre at all and give -log(0).
        """
        a = self._pooled_map(attn)                              # (B,N)
        cx, cy, w_, h_ = box.unbind(-1)
        x0, x1 = cx - w_ / 2, cx + w_ / 2
        y0, y1 = cy - h_ / 2, cy + h_ / 2
        mass, off = a.new_zeros(a.shape[0]), 0
        for t in self.taps:
            gh = self.img_size[0] // self.events.strides[t]
            gw = self.img_size[1] // self.events.strides[t]
            m = a[:, off:off + gh * gw].view(-1, gh, gw)
            off += gh * gw
            xs = (torch.arange(gw, device=a.device, dtype=a.dtype) + 0.5) / gw
            ys = (torch.arange(gh, device=a.device, dtype=a.dtype) + 0.5) / gh
            inx = (xs[None] >= x0[:, None]) & (xs[None] <= x1[:, None])      # (B,gw)
            iny = (ys[None] >= y0[:, None]) & (ys[None] <= y1[:, None])      # (B,gh)
            # the cell holding the centre, so the mask is never empty
            ci = (cx * gw).long().clamp(0, gw - 1)
            ri = (cy * gh).long().clamp(0, gh - 1)
            inx = inx.clone(); inx[torch.arange(len(ci), device=a.device), ci] = True
            iny = iny.clone(); iny[torch.arange(len(ri), device=a.device), ri] = True
            mask = (iny[:, :, None] & inx[:, None, :]).to(a.dtype)           # (B,gh,gw)
            mass = mass + (m * mask).sum(dim=(1, 2))
        return mass

    def _attn_position_prior(self, attn: torch.Tensor) -> torch.Tensor:
        """(T,B,h,Lq,N) attention map -> (B,2,V) log-density over the cx/cy slot grids.

        Why this exists
        ---------------
        Everything the fusion learns about WHERE the referent is has to reach the box
        head through `proj_lif`, which is binary: the analog attention map -- the one
        tensor in the model that is indexed by spatial position -- is quantised to 256
        bits before the head ever sees it. This path reads the map's own spatial
        marginals instead, and hands them to the slot head as a prior.

        It is not the RefCOCO `attn_softargmax` head, which REPLACED the regressed
        centre with the map's expectation and collapsed. A prior cannot collapse the
        same way: at `attn_prior_gain` 0 the model is exactly the unmodified one, the
        gain is learnable, and the slot cross-entropy trains it directly.

        Both taps contribute. Each tap's marginal is converted to a DENSITY on [0,1]
        (multiply the pmf by its own bin count) before resampling, so a 80-column s8
        marginal and a 40-column s16 marginal are weighted by their attention mass
        rather than by how many bins they happen to have.
        """
        V = self.box_head.n_slots
        a = self._pooled_map(attn)                              # (B,N)
        px = a.new_zeros(a.shape[0], V)
        py = a.new_zeros(a.shape[0], V)
        off = 0
        for t in self.taps:
            h = self.img_size[0] // self.events.strides[t]
            w = self.img_size[1] // self.events.strides[t]
            m = a[:, off:off + h * w].view(-1, h, w)
            off += h * w
            mass = m.sum(dim=(1, 2)).clamp_min(1e-12)           # (B,) this tap's share
            mx = m.sum(1) / mass[:, None] * w                   # (B,w) density on [0,1]
            my = m.sum(2) / mass[:, None] * h                   # (B,h)
            up = lambda d, n: F.interpolate(d[:, None], size=V, mode="linear",
                                            align_corners=False)[:, 0]
            px = px + mass[:, None] * up(mx, w)
            py = py + mass[:, None] * up(my, h)
        if off != a.shape[1]:
            raise RuntimeError(f"taps cover {off} keys but the map has {a.shape[1]}")
        # normalise away any drift from the interpolation, so a uniform map maps to
        # exactly 1.0 everywhere and contributes exactly zero once logged
        px = px / px.mean(-1, keepdim=True).clamp_min(1e-12)
        py = py / py.mean(-1, keepdim=True).clamp_min(1e-12)
        # Mix with the uniform density before the log. Not cosmetic: with --map-weight
        # the map collapses to ~41 effective positions of 6000, so most slots carry
        # essentially zero density, and d log(p)/dp = 1/p would hand the softmax
        # gradients of order 1e8. The mixture bounds the prior to
        # [log(eps), log(1/eps)] and the gradient to 1/eps, and leaves the mean at 1.
        e = self.attn_prior_eps
        p = (1 - e) * torch.stack([px, py], dim=1) + e
        return self.attn_prior_gain * p.log()

    def set_collect_stats(self, on: bool) -> None:
        """Turn the two leading indicators on. Eval only -- it materialises the last
        block's attention map, which the training path is careful not to keep."""
        self.collect_stats = on
        for blk in self.blocks:
            blk.attn.collect_stats = on

    def forward(self, cube, input_ids, attention_mask):
        if cube.dim() != 5 or cube.shape[0] != self.T:
            raise ValueError(f"expected (T={self.T},B,2,H,W) cube, got {tuple(cube.shape)}")
        if cube.shape[1] != input_ids.shape[0]:
            raise ValueError(f"batch mismatch {cube.shape[1]} vs {input_ids.shape[0]}")
        self.reset()

        # --- language first: the encoder must know what is being asked ---------
        txt = self.text(input_ids, attention_mask)              # (B,L,d)
        queries, tag_logits = self.tagger(txt, attention_mask)  # (B,4,d), (B,L,5)
        gains = self.modulator(queries) if self.modulator is not None else None

        # --- conditioned spiking encoder, then the accumulator -----------------
        feats = self.events(cube, gains=gains)                  # {tap:(T,B,C,h,w)}
        pooled_feats = self.events.accumulate(feats)            # {tap:(B,C,h,w)} REAL

        toks = []
        lat_ms, pos_ms = [], []
        for i, t in enumerate(self.taps):
            f = self.tap_norm[t](pooled_feats[t])
            x = self.lateral[t](f)                              # (B,d,h,w)
            pos = self.pos[t]
            if pos.shape[-2:] != x.shape[-2:]:
                pos = nn.functional.interpolate(pos, size=x.shape[-2:],
                                                mode="bilinear", align_corners=False)
            if self.pos_ratio is not None:
                lat_rms = x.detach().float().pow(2).mean().sqrt()
                pos_rms = pos.detach().float().pow(2).mean().sqrt().clamp_min(1e-12)
                pos = pos * (self.pos_ratio * lat_rms / pos_rms).to(pos.dtype)
            if self.collect_stats:
                # measured BEFORE the addition -- the ratio of the two is the question
                lat_ms.append(x.detach().float().pow(2).mean())
                pos_ms.append(pos.detach().float().pow(2).mean())
            x = x + pos + self.level[i].view(1, -1, 1, 1)
            B, d, h, w = x.shape
            toks.append(x.reshape(B, d, h * w).permute(0, 2, 1))
        vis = torch.cat(toks, dim=1)                            # (B,N,d)

        # SpatialBlock expects a leading T axis; downstream of the accumulator there is
        # no time left, so run it as T=1 -- one real-valued step, not a spiking one.
        q = self.q_norm(queries).unsqueeze(0)                   # (1,B,4,d)
        v = self.v_norm(vis).unsqueeze(0)                       # (1,B,N,d)
        attn = None
        need_attn = self.collect_stats or self.attn_prior or self.return_map
        for i, blk in enumerate(self.blocks):
            if need_attn and i == len(self.blocks) - 1:
                q, attn = blk(q, v, return_attn=True)
            else:
                q = blk(q, v)
        q = q.squeeze(0)                                        # (B,4,d)

        if self.collect_stats:
            # perplexity = exp(entropy) of the post-softmax weights, per (T,B,head,
            # query), then averaged. fp32: under bf16 autocast the log of a ~1/6000
            # weight has no mantissa left to be right with.
            a = attn.detach().float().clamp_min(1e-12)          # (T,B,h,Lq,N)
            ppl = torch.exp(-(a * a.log()).sum(-1)).mean()
            self.stats = {"attn_perplexity": ppl,
                          "n_keys": torch.as_tensor(float(vis.shape[1]), device=ppl.device),
                          "pos_rms_ratio": (torch.stack(pos_ms).mean().sqrt()
                                            / torch.stack(lat_ms).mean().sqrt().clamp_min(1e-12))}
            for k_, v_ in self.blocks[-1].attn.last_rates.items():
                self.stats[k_] = v_
            if self.attn_prior:
                # a gain still at its init means the head never found the map useful
                self.stats["prior_gain"] = self.attn_prior_gain.detach().squeeze()
            if self.query_logits is not None:
                w = self.query_logits.detach().softmax(0)
                for i, nm in enumerate(("appear", "status", "viewer", "others")):
                    self.stats[f"qw_{nm}"] = w[i]

        pooled = self.head_norm(q.mean(dim=1))                  # (B,d)
        prior = self._attn_position_prior(attn) if self.attn_prior else None
        box, slot_logits = self.box_head(pooled, prior=prior)
        out = {"box": box, "slot_logits": slot_logits, "tag_logits": tag_logits}
        if self.return_map or self.collect_stats:
            out["attn"] = attn
        return out

    @torch.no_grad()
    def predict(self, cube, input_ids, attention_mask, amp: bool = True):
        was = self.training
        self.eval()
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if amp and cube.is_cuda
               else torch.autocast("cpu", enabled=False))
        with ctx:
            out = self(cube, input_ids, attention_mask)
        self.train(was)
        return out["box"].float()
