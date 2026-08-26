"""Diagnostics for a RefCOCOGrounding checkpoint: firing rates, positional signal,
attention precision, and the train/eval mode gap.

    uv run python tools/diagnose.py --run runs/of100_pooled

Four numbers this exists to surface, each tied to a specific failure mode already found
in this codebase (see docs/research-log.md and the coordinate-precision study it links):

1. **Firing rate per spiking layer.** A binarising LIF stuck at ~0% makes its residual
   branch the identity, silently -- research-log finding 1 measured CMSF's cross-attention
   at EXACTLY 0.0% firing with a healthy-looking loss and a caption-blind model. Anything
   under ~1% or over ~90% is worth a second look; healthy is roughly 5-35%.
2. **Positional RMS ratio** -- RMS(`vision.pos`) / RMS(pre-position lateral output). Under
   ~0.05, the positional embedding is a small perturbation next to the features it is
   added to, and the LIF it feeds may threshold it away entirely before the map can carry
   any location information.
3. **Attention perplexity** -- exp(entropy) of each real caption token's attention
   distribution, i.e. the effective number of vision positions attended. This is the
   direct test of the quantisation argument behind the `attn_softargmax` head: with q, k
   both spike-valued and dh=32, `q @ k^T` is an integer in [0, 32], so the pre-softmax
   logits take at most 33 distinct values over ~576 keys. Perplexity near the true key
   count (~576) means the map is close to uniform and carries no location at all;
   perplexity near 1 means it has collapsed onto a single position.
4. **Train mIoU, train mode vs eval mode**, on the SAME samples. A large gap here means
   the eval path itself is suspect and every other number in this script -- and in
   `log.tsv` -- should be distrusted until it is fixed.

Nothing here is a pass/fail gate on its own; it is context for reading the A/B numbers.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spiketrandvg.dataset import RefCOCO, make_collate
from spiketrandvg.model import RefCOCOGrounding, cxcywh_to_xyxy_norm
from spiketrandvg.text_encoder import build_tokenizer
from torchvision.ops import box_iou


def build_from_args(run_args: dict, device: str) -> RefCOCOGrounding:
    """Reconstruct the exact architecture a run was trained with.

    `rgb_ckpt=None` deliberately: the checkpoint's own state_dict overwrites every
    backbone weight immediately after construction, so loading SpiLiFormer's ImageNet
    checkpoint first would be wasted work.
    """
    size = tuple(run_args["size"])
    return RefCOCOGrounding(
        rgb_ckpt=None, text_model=run_args.get("text_model", "roberta-base"),
        img_size=size, T=run_args.get("T", 4), rgb_T=run_args.get("rgb_T", 1),
        depth=run_args.get("depth", 2), attn_type=run_args.get("attn_type", "spatial_softmax"),
        freeze_rgb=False, freeze_text=not run_args.get("train_text", False),
        head_type=run_args.get("head_type", "pooled_mlp"),
        attn_map=run_args.get("attn_map", "last"),
        pos_std=run_args.get("pos_std", 0.02),
        text_unfreeze_last=(0 if run_args.get("train_text") else
                            run_args.get("text_unfreeze_last", 0)),
    ).to(device)


@torch.no_grad()
def firing_rates(model: RefCOCOGrounding, rgb, ids, mask) -> dict[str, float]:
    """Mean fraction of nonzero activations per spiking layer, one forward pass.

    Hooked by class name (`"LIF" in type(m).__name__` or `"mem_update"`) rather than by
    isinstance against imported classes, so this covers CMSF's `Dynamic_Threshold_LIFNode`,
    spikingjelly's `MultiStepLIFNode`, and SpikeYOLO's `mem_update` without needing to know
    their exact module paths.
    """
    rates: dict[str, float] = {}
    handles = []

    def make_hook(name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            rates[name] = (t != 0).float().mean().item()
        return hook

    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if "LIF" in cls or cls == "mem_update":
            handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        model(rgb, ids, mask)
    finally:
        for h in handles:
            h.remove()
    return rates


@torch.no_grad()
def positional_ratio(model: RefCOCOGrounding) -> float | None:
    """RMS(vision.pos) / RMS(pre-position lateral output from the last forward call)."""
    lat = model.vision.last_lateral_rms
    if lat is None or lat == 0:
        return None
    pos_rms = model.vision.pos.detach().float().pow(2).mean().sqrt().item()
    return pos_rms / lat


@torch.no_grad()
def attention_perplexity(model: RefCOCOGrounding, rgb, ids, mask) -> dict[str, float]:
    """exp(entropy) of each real token's attention distribution, per block.

    Uses `return_diagnostics=True` rather than re-deriving attention outside the model,
    so this always reflects exactly what the model itself computed -- no duplicated
    forward logic to drift out of sync.
    """
    _, diag = model(rgb, ids, mask, return_diagnostics=True)
    out = {}
    m = mask.to(torch.float32).unsqueeze(-1)                    # (B,L,1)
    for i, attn in enumerate(diag["attn_maps"]):
        a = attn.float().mean(dim=(0, 2))                       # (B,L,N)
        a = a.clamp_min(1e-12)
        entropy = -(a * a.log()).sum(-1)                        # (B,L)
        perplexity = entropy.exp()
        real = m.squeeze(-1).bool()
        out[f"block{i}"] = perplexity[real].mean().item()
    return out


@torch.no_grad()
def mode_gap(model: RefCOCOGrounding, loader, device, amp_ctx) -> dict[str, float]:
    """mIoU on the same samples, train() vs eval(). See the module docstring."""
    def run(training: bool) -> float:
        model.train(training)
        ious = []
        for rgb, ids, mask, gt, _c, _m in loader:
            rgb, ids, mask, gt = (t.to(device) for t in (rgb, ids, mask, gt))
            with amp_ctx():
                pred = model(rgb, ids, mask)
            ious.append(box_iou(cxcywh_to_xyxy_norm(pred.float()),
                                cxcywh_to_xyxy_norm(gt)).diagonal().cpu())
        return torch.cat(ious).mean().item()

    train_mode = run(True)
    model.eval()
    eval_mode = run(False)
    return {"train_mode_mIoU": train_mode, "eval_mode_mIoU": eval_mode,
            "gap": eval_mode - train_mode}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory, e.g. runs/of100_pooled")
    ap.add_argument("--ckpt", default="best.pth", help="filename inside --run")
    ap.add_argument("--n-batches", type=int, default=4,
                    help="batches to average firing rates / perplexity over")
    ap.add_argument("--mode-gap-samples", type=int, default=100,
                    help="samples for the train/eval mode-gap check")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run)
    run_args = json.loads((run_dir / "args.json").read_text())
    blob = torch.load(run_dir / args.ckpt, map_location=device, weights_only=False)

    model = build_from_args(run_args, device)
    # strict=False: a checkpoint from before head_norm0 (item 2) existed is missing that
    # key. Report it rather than silently leaving head_norm0 at its random init, which
    # would make this run's mode-gap and firing-rate numbers for head_lif1 incomparable
    # to a fresh run without saying so.
    missing = model.load_state_dict(blob["model"], strict=False)
    print(f"loaded {run_dir / args.ckpt} (epoch {blob.get('epoch')}, "
          f"head_type={run_args.get('head_type', 'pooled_mlp')})")
    if missing.missing_keys or missing.unexpected_keys:
        print(f"  NOTE: missing {missing.missing_keys}, unexpected {missing.unexpected_keys}"
              f" -- checkpoint predates a code change; those params are at random init")
    print()

    tokenizer = build_tokenizer()
    size = tuple(run_args["size"])
    ds = RefCOCO(run_args["dataset"], "train", size=size, augment=None,
                limit=run_args.get("limit_train"))
    dl = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True,
                    collate_fn=make_collate(tokenizer))
    amp_ctx = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
               if device == "cuda" else __import__("contextlib").nullcontext)

    # --- 1 & 3: firing rates + attention perplexity, averaged over n_batches ----------
    model.eval()
    rate_sums: dict[str, float] = defaultdict(float)
    perp_sums: dict[str, float] = defaultdict(float)
    n = 0
    it = iter(dl)
    for _ in range(args.n_batches):
        try:
            rgb, ids, mask, gt, _c, _m = next(it)
        except StopIteration:
            it = iter(dl)
            rgb, ids, mask, gt, _c, _m = next(it)
        rgb, ids, mask = (t.to(device) for t in (rgb, ids, mask))
        with amp_ctx():
            rates = firing_rates(model, rgb, ids, mask)
            perps = attention_perplexity(model, rgb, ids, mask)
        for k, v in rates.items():
            rate_sums[k] += v
        for k, v in perps.items():
            perp_sums[k] += v
        n += 1

    print(f"=== firing rate per spiking layer (n={n} batches) ===")
    print("    (< 1% = likely dead, silently identity-passing its residual branch)")
    print("    (> 90% = likely saturated)")
    for k in sorted(rate_sums):
        v = rate_sums[k] / n
        flag = "  <-- DEAD" if v < 0.01 else "  <-- SATURATED" if v > 0.90 else ""
        print(f"  {k:45s} {v*100:6.2f}%{flag}")

    print(f"\n=== positional RMS ratio (RMS(pos) / RMS(pre-pos lateral output)) ===")
    ratio = positional_ratio(model)
    if ratio is None:
        print("  unavailable (no forward call captured a lateral RMS)")
    else:
        flag = "  <-- position may be lost before the LIF threshold" if ratio < 0.05 else ""
        print(f"  {ratio:.4f}{flag}")

    print(f"\n=== attention perplexity per block (n={n} batches, ~{model.vision.pos.shape[-2]}"
          f"x{model.vision.pos.shape[-1]} = "
          f"{model.vision.pos.shape[-2]*model.vision.pos.shape[-1]} keys) ===")
    print("    (near the key count = map is ~uniform, carries no location)")
    print("    (near 1 = map has collapsed onto a single position)")
    for k in sorted(perp_sums):
        print(f"  {k:15s} {perp_sums[k]/n:8.2f}")

    # --- 4: train mode vs eval mode, on the exact training samples --------------------
    print(f"\n=== train/eval mode gap (mIoU on {min(args.mode_gap_samples, len(ds))} "
          f"training samples) ===")
    from torch.utils.data import Subset
    n_gap = min(args.mode_gap_samples, len(ds))
    gap_idxs = list(range(0, len(ds), max(1, len(ds) // n_gap)))[:n_gap]
    gap_dl = DataLoader(Subset(ds, gap_idxs), batch_size=8, shuffle=False,
                        collate_fn=make_collate(tokenizer))
    g = mode_gap(model, gap_dl, device, amp_ctx)
    flag = "  <-- SUSPECT: fix the eval path before trusting other numbers" \
        if abs(g["gap"]) > 0.10 else ""
    print(f"  train() mIoU {g['train_mode_mIoU']:.4f}  eval() mIoU {g['eval_mode_mIoU']:.4f}"
          f"  gap {g['gap']:+.4f}{flag}")

    if run_args.get("head_type") == "attn_softargmax":
        scale = model.attn_logit_scale.exp().item()
        print(f"\nattn_logit_scale.exp() = {scale:.4f}  "
              f"(collapsing toward 0 flattens the map toward uniform, which pins "
              f"every centre to (0.5, 0.5) -- the same output a fresh sigmoid head "
              f"gives at init, so this failure looks like 'no progress', not a bug)")


if __name__ == "__main__":
    main()
