"""How sharp does the attention map get as a function of the softmax temperature?

    uv run python tools/temperature_response.py --run runs/probe_00 --limit 400

`SpatialCrossAttention` divides its logits by sqrt(dh), the standard transformer value.
That constant is derived for ANALOG q and k -- unit-variance Gaussian entries make q.k
have variance dh. Here q and k are binary, so the derivation does not apply and the
resulting logit spread is tens of times too small for a softmax over thousands of keys.

This sweeps the scale on an ALREADY TRAINED checkpoint and reports what the map does.
It cannot tell you what training at a given scale would produce -- the weights were
fitted under the default -- but it brackets the range where the map stops being uniform,
which is what a probe budget should be spent inside rather than on finding.

Runs on `val`, the split checkpoints are selected on. It never opens `test`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou

from spiketrandvg.dataloader import Talk2Event, make_t2e_collate
from spiketrandvg.model import Talk2EventGrounding, cxcywh_to_xyxy_norm
from spiketrandvg.textencoder import build_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pth")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--scales", type=float, nargs="*", default=None)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("this is a diagnostic; it does not open the test split")

    device = "cuda"
    run = Path(args.run)
    ra = json.loads((run / "args.json").read_text())
    model = Talk2EventGrounding(
        event_ckpt=None, event_backbone=ra.get("event_backbone", "spiliformer_dvs"),
        text_model=ra.get("text_model", "roberta-base"), img_size=tuple(ra["size"]),
        T=ra.get("T", 5), depth=ra.get("depth", 2), n_slots=ra.get("n_slots", 1000),
        ilif=not ra.get("no_ilif", False),
        condition_encoder=not ra.get("no_condition", False),
        freeze_event=ra.get("freeze_event", False),
        freeze_text=not ra.get("train_text", False),
        text_unfreeze_last=ra.get("text_unfreeze_last", 0),
        pos_std=ra.get("pos_std", 0.02), attn_scale=ra.get("attn_scale"),
    ).to(device)
    blob = torch.load(run / args.ckpt, map_location=device, weights_only=False)
    rep = model.load_state_dict(blob["model"], strict=False)
    if rep.missing_keys or rep.unexpected_keys:
        print(f"WARNING missing {rep.missing_keys} unexpected {rep.unexpected_keys}")
    model.eval()

    ds = Talk2Event(args.split, t_steps=ra.get("T", 5))
    idxs = list(range(0, len(ds), max(1, len(ds) // args.limit))) if args.limit else None
    src = Subset(ds, idxs) if idxs else ds
    dl = DataLoader(src, batch_size=args.batch_size, shuffle=False, num_workers=6,
                    collate_fn=make_t2e_collate(build_tokenizer()), pin_memory=True)

    default = model.blocks[0].attn.dh ** -0.5
    scales = args.scales or [default, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    print(f"{run}/{args.ckpt} (epoch {blob.get('epoch')}) on {args.split}: "
          f"{len(src)} of {len(ds)} samples\ndefault scale = dh**-0.5 = {default:.4f}\n")
    print(f"{'scale':>8}  {'xdefault':>8}  {'perplex':>9}  {'mIoU':>7}  {'Acc@0.5':>7}  "
          f"{'Acc@0.75':>8}")

    for sc in scales:
        for blk in model.blocks:
            blk.attn.scale = float(sc)
        model.set_collect_stats(True)
        ious, ppl, n = [], 0.0, 0
        with torch.no_grad():
            for cube, ids, mask, gt, _l, _c, _r in dl:
                cube, ids, mask, gt = (t.to(device, non_blocking=True)
                                       for t in (cube, ids, mask, gt))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(cube, ids, mask)
                ious.append(box_iou(cxcywh_to_xyxy_norm(out["box"].float()),
                                    cxcywh_to_xyxy_norm(gt)).diagonal().cpu())
                ppl += model.stats["attn_perplexity"].item() * gt.shape[0]
                n += gt.shape[0]
        model.set_collect_stats(False)
        t = torch.cat(ious)
        print(f"{sc:8.4f}  {sc/default:8.1f}  {ppl/n:9.1f}  {t.mean():7.4f}  "
              f"{(t >= 0.5).float().mean():7.4f}  {(t >= 0.75).float().mean():8.4f}",
              flush=True)


if __name__ == "__main__":
    main()
