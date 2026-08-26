"""RefCOCOGrounding: SpiLiFormer + roberta + text-queried cross-attention -> one box.

    rgb (B,3,H,W)                          input_ids / attention_mask (B,L)
          |                                              |
    SpiLiFormer (SNN)                          roberta-base (ANN, frozen)
    ICCV 2025, ImageNet 85.82% @ T=4           124.6M + a trainable 0.2M projection
          | /16                                          |
    lateral 1x1 + learnable 2D pos                       |
          |  (T,B,N_vis,256)                             |  (B,L,256)
          |                                     RepeatTextEncoder -> (T,B,L,256)
          +------------> SpatialBlock x depth <----------+
                 softmax cross-attention, Q = TEXT, K = V = VISION
                              |  (T,B,L,256)
                    masked mean over real tokens -> spiking MLP
                              |
                       ONE box (B,4) normalised cxcywh

This is `SpikeGroundingV2` with the event branch removed and the text encoder brought
inside. Everything else is the same object: `RgbTokens`, `SpatialBlock` and the box head
are imported from `models/grounding_v2.py` rather than reimplemented, so a finding about
the fusion or the head applies to both models and there is one place to fix it.

Why a separate class instead of a flag on SpikeGroundingV2
----------------------------------------------------------
`SpikeGroundingV2.forward` requires an event cube and validates its shape against T_STEPS.
Making that optional would put a second, RGB-only code path through a class whose whole
purpose is the two-stream comparison, and every event-stream invariant would have to grow
an "unless RGB-only" clause. RefCOCO has no events, and hybrid experiments are still the
Talk2Event baseline, so the two stay separate and share components.

The text encoder IS part of this model
--------------------------------------
Unlike `SpikeGroundingV2`, which takes `(B, L, d_model)` tokens and leaves their
production to the caller, this model owns its `TextEmbedder`. On Talk2Event the encoder
was a moving research variable -- SpikeLM, then roberta, frozen or not -- and keeping it
outside made swapping it a one-line change. Here it is fixed: RefCOCO is a standard
benchmark and the point is to measure the fusion, not to re-litigate the encoder. Owning
it means `forward(rgb, input_ids, attention_mask)` is the whole contract.

What T means with no event stream
---------------------------------
On Talk2Event, T came from the event window: 5 voxel bins, a real temporal axis. A still
image has none, so T here is purely the spiking simulation length. It still matters --
LIF membranes integrate across timesteps, so T=1 degenerates every neuron into a plain
threshold and discards the dynamics that make this an SNN. Default 4, matching the T the
SpiLiFormer checkpoint was pretrained at. Cost is linear in T.

Two traps carried over from the Talk2Event work, both measured
--------------------------------------------------------------
**Positional embedding on the vision tokens is load-bearing** under `cmsf_linear`
attention, which forms `k^T v` and sums over every key position -- the spatial index is
marginalised away and no text query can ask *where*. `spatial_softmax` (the default) keeps
one weight per position and does not have this problem, which is why it is the default.

**These layers are not dtype-agnostic.** Precision changes which membranes cross
threshold, so evaluating a bf16-trained model in fp32 is a different function: measured
0.656 IoU vs 0.060 on identical weights. `predict` autocasts to bf16 for that reason;
train under the same dtype.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron as sj_neuron

from spiketrandvg.models.grounding_v2 import RgbTokens, SpatialBlock
from spiketrandvg.models.text_embedder import TextEmbedder
from spiketrandvg.utils import forks

__all__ = ["RefCOCOGrounding", "build_model"]


class RefCOCOGrounding(nn.Module):
    """forward(rgb, input_ids, attention_mask) -> (B, 4) normalised cxcywh.

    One box, train and eval alike -- no anchors, no scores, no argmax, so nothing to
    threshold or NMS at inference.

    Args:
        rgb_ckpt: SpiLiFormer ImageNet checkpoint. None starts the backbone from scratch,
            which on 120k RefCOCO expressions is possible but wastes the one pretrained
            visual prior available.
        text_model: any HF encoder; must be the tokenizer the dataloader used.
        img_size: (H, W) the dataloader emits. Used to size the positional table at the
            backbone's /16 grid so it is exact rather than interpolated at every step.
            Must be divisible by 16.
        T: spiking timesteps for the fusion stack. See the module docstring.
        rgb_T: SpiLiFormer's own internal timesteps. 1 is a single feature-extraction
            pass; the checkpoint was trained at 4.
        freeze_rgb / freeze_text: on Talk2Event, freezing BOTH encoders produced a
            caption-blind model over 85 epochs (delta +0.0009) while unfreezing the vision
            side gave +0.051 within two. The default here follows that result: vision
            trains, text does not.
        depth: SpatialBlocks. Each is cross-attention + a spiking gated MLP.
        attn_type: "spatial_softmax" (default) or "cmsf_linear".
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
    ):
        super().__init__()
        if img_size[0] % 16 or img_size[1] % 16:
            raise ValueError(f"img_size {img_size} must be divisible by 16")
        if d_model % num_heads:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        cmsf = forks.load_cmsf()
        self.T = T
        self.d_model = d_model
        self.img_size = tuple(img_size)

        # --- vision: SpiLiFormer over the RGB frame -------------------------------
        # max_hw sized to this input so the positional table is used directly; RgbTokens
        # interpolates it when the grid differs, which is correct but lossy every step.
        self.vision = RgbTokens(
            rgb_ckpt, d_model=d_model, T_model=rgb_T,
            max_hw=(img_size[0] // 16, img_size[1] // 16), freeze=freeze_rgb,
        )
        self._disable_unused_backbone_grads()

        # --- text: ANN roberta, frozen, + a trainable projection to d_model -------
        self.text = TextEmbedder(text_model, d_model=d_model, freeze=freeze_text)

        # --- spike coders: both sides arrive analog, CMSF's blocks want spikes -----
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

        # --- box head: identical to SpikeGroundingV2's -----------------------------
        sy = forks.load_spikeyolo()
        self.head_lif1 = sy.mem_update()
        self.fc1 = nn.Linear(d_model, head_hidden)
        self.head_norm = nn.LayerNorm(head_hidden)
        self.head_lif2 = sy.mem_update()
        self.fc2 = nn.Linear(head_hidden, 4)
        # small, never zero: a zero output weight sends exactly zero gradient back through
        # the entire fusion stack, which cost a full optimiser run to diagnose once
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def _disable_unused_backbone_grads(self) -> None:
        """Turn off grad for the SpiLiFormer parts `forward_features` never reaches.

        MEASURED, not defensive: with the backbone unfrozen, 25 of its 256 tensors --
        4.91M parameters -- came back with `grad is None` after a full backward. They are
        the ImageNet classifier `head` (0.77M) and the lateral-inhibition `decoder`
        feedback path with its two `prompt` vectors (4.13M), and `RgbTokens` uses only the
        feedforward pass. Left on, they would be reported as trainable, carry AdamW's two
        moment buffers each (~39MB of optimiser state), and never move -- so any
        "trainable parameters" figure including them overstates the model by 7%.
        """
        bb = self.vision.backbone
        for name in ("head", "decoder", "prompt", "prompt_2"):
            mod = getattr(bb, name, None)
            if isinstance(mod, nn.Module):
                mod.requires_grad_(False)
            elif isinstance(mod, nn.Parameter):
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
        """Zero every spikingjelly membrane.

        Per-neuron rather than `functional.reset_net(self)`: that helper calls `.reset()`
        on every submodule that has one, which includes THIS module, and recurses until
        the stack blows.
        """
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
            "head": sum(p.numel() for m in (self.fc1, self.head_norm, self.fc2)
                        for p in m.parameters() if p.requires_grad),
        }
        return {k: v for k, v in out.items() if v}

    def forward(self, rgb: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """rgb (B,3,H,W); input_ids (B,L); attention_mask (B,L) -> (B,4) cxcywh in [0,1]."""
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

        for blk in self.blocks:
            q = blk(q, v)                                    # Q=text, K=V=vision

        # masked mean over the real caption tokens, then over T
        q = q * mask
        pooled = q.sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)   # (T,B,d)
        x = self.head_lif1(pooled)
        x = self.head_norm(self.fc1(x.flatten(0, 1))).reshape(*pooled.shape[:2], -1)
        x = self.head_lif2(x)
        x = self.fc2(x.flatten(0, 1)).reshape(*pooled.shape[:2], 4)
        return x.mean(0).sigmoid()                           # (B,4) normalised cxcywh

    @torch.no_grad()
    def predict(self, rgb, input_ids, attention_mask, amp: bool = True) -> torch.Tensor:
        """Eval-mode boxes, (B,4) normalised cxcywh.

        Autocasts to bf16 because that is the dtype these layers are trained in and they
        are NOT dtype-agnostic -- precision decides which membranes cross threshold.
        Measured on a same-frame overfit set: bf16 0.656 IoU vs fp32 0.060, identical
        weights and mode. Pass `amp=False` only to reproduce that comparison.
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
