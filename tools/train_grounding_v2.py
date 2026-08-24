"""Stage 2: train SpikeGroundingV2 -- text-queried cross-attention over a frozen backbone.

    uv run python tools/train_grounding_v2.py --run-name ground_v2 --epochs 20

Trainable: lateral projections, positional embedding, cross-attention, box head (4.02M),
plus the text projection. Frozen by default: the SpikeYOLO detection backbone (12.6M,
mAP@0.5 0.5307) and the HuggingFace text encoder chosen with `--text-model`.

The text encoder is a separate module (`models/text_embedder.py`), not part of the
grounding model. SpikeLM was removed: 124.3M parameters of roberta-base transplanted into
a spiking BERT, never spike-pretrained, and frozen it left the model caption-blind over 85
epochs (mean delta +0.0009).

Two things this script does that the earlier grounding trainer did not
----------------------------------------------------------------------
**Everything runs in bf16, evaluation included.** These spiking layers are not
dtype-agnostic: precision decides which membranes cross threshold, so a bf16-trained model
evaluated in fp32 is a different network. Measured on a same-frame overfit set at step 150
-- bf16 0.656 IoU against fp32 0.060, identical weights and mode. The earlier trainer
trained in bf16 and measured in fp32, which makes some of its conclusions unsafe.

**A caption-blindness control is a first-class metric.** The previous grounding run
reached mIoU 0.2225 and looked reasonable, but feeding deliberately WRONG captions changed
it by 0.001 -- the language branch contributed nothing and the model was emitting a
plausible salient box from events alone. `blind_mIoU` here is that same control: the same
events paired with a shuffled caption. What matters is the DELTA. A model whose delta is
near zero has not learned to ground, whatever its mIoU says.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.ops import box_iou

from spiketrandvg.datasets.events_voxel_cube import T_STEPS, talk2event_cube
from spiketrandvg.datasets.talk2event_dataset import Talk2EventDataset
from spiketrandvg.models.grounding_loss import SingleBoxLoss, cxcywh_to_xyxy_norm
from spiketrandvg.models.grounding_v2 import SpikeGroundingV2
from spiketrandvg.models.text_embedder import MAX_TEXT_LEN, TextEmbedder, build_tokenizer

IOU_THRESHOLDS = (0.25, 0.5, 0.75, 0.9)


class GroundingSamples(Dataset):
    """One sample per (object, caption): event cube, caption, box in normalised cxcywh."""

    def __init__(self, split: str):
        self.ds = Talk2EventDataset(SimpleNamespace(attribute="all"), image_set=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        inp, tgt = self.ds[i]
        return talk2event_cube(inp["event"]), tgt["caption"], tgt["boxes"][0]


def make_collate(tokenizer):
    def collate(batch):
        cubes, caps, boxes = zip(*batch)
        tok = tokenizer(list(caps), padding="max_length", truncation=True,
                        max_length=MAX_TEXT_LEN, return_tensors="pt")
        return (torch.stack(cubes, dim=1), tok["input_ids"], tok["attention_mask"],
                torch.stack(boxes), list(caps))
    return collate


@torch.no_grad()
def evaluate(model, text, loader, device, amp_ctx, tokenizer, blind: bool = False):
    """mIoU and Acc@thresholds. With `blind`, captions are shuffled across the batch --
    the model sees the right events with the wrong words."""
    model.eval(); text.eval()
    ious = []
    for cube, ids, mask, gt, caps in loader:
        if blind:
            perm = list(range(len(caps)))
            if len(perm) > 1:                       # derange within the batch
                perm = perm[1:] + perm[:1]
            tok = tokenizer([caps[j] for j in perm], padding="max_length", truncation=True,
                            max_length=MAX_TEXT_LEN, return_tensors="pt")
            ids, mask = tok["input_ids"], tok["attention_mask"]
        cube, ids, mask, gt = (t.to(device, non_blocking=True) for t in (cube, ids, mask, gt))
        with amp_ctx():
            pred = model(cube, text(ids, mask), mask)
        p, g = cxcywh_to_xyxy_norm(pred.float()), cxcywh_to_xyxy_norm(gt)
        ious += box_iou(p, g).diagonal().tolist()
    t = torch.tensor(ious)
    return {"n": len(ious), "mIoU": t.mean().item(),
            **{f"Acc@{th}": (t >= th).float().mean().item() for th in IOU_THRESHOLDS}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="ground_v2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--attn-type", default="spatial_softmax",
                    choices=("spatial_softmax", "cmsf_linear"))
    ap.add_argument("--backbone-ckpt", default="runs/det_aug/backbone.pth")
    ap.add_argument("--freeze-vision", action="store_true",
                    help="hold the detection-pretrained backbone fixed")
    ap.add_argument("--text-model", default="roberta-base",
                    help="HuggingFace encoder that produces the caption tokens")
    ap.add_argument("--freeze-text", action="store_true",
                    help="hold the text encoder fixed")
    ap.add_argument("--encoder-lr", type=float, default=1e-5,
                    help="LR for the PRETRAINED encoders; the from-scratch modules use --lr")
    ap.add_argument("--taps", nargs="+", default=["s8", "s16"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-samples", type=int, default=800, help="0 = full test split")
    ap.add_argument("--blind-every", type=int, default=1,
                    help="run the wrong-caption control every N epochs; 0 = off")
    ap.add_argument("--patience", type=int, default=0, help="0 = off")
    ap.add_argument("--log-every", type=int, default=200, help="optimiser steps")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("runs") / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    amp_ctx = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
               if device == "cuda" else contextlib.nullcontext)

    tokenizer = build_tokenizer()
    collate = make_collate(tokenizer)
    train_ds, test_ds = GroundingSamples("train"), GroundingSamples("test")
    eval_src = test_ds if not args.eval_samples else Subset(
        test_ds, range(0, len(test_ds), max(1, len(test_ds) // args.eval_samples)))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, collate_fn=collate, drop_last=True,
                          pin_memory=True, persistent_workers=args.workers > 0)
    test_dl = DataLoader(eval_src, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, collate_fn=collate, pin_memory=True)

    # The text encoder now lives OUTSIDE the grounding model: it produces (B, L, d_model)
    # tokens and the model consumes them. Swapping encoders is a one-flag change.
    text = TextEmbedder(args.text_model, d_model=256, freeze=args.freeze_text).to(device)
    model = SpikeGroundingV2(
        taps=tuple(args.taps), T=T_STEPS, backbone_ckpt=args.backbone_ckpt,
        depth=args.depth, attn_type=args.attn_type,
        freeze_vision=args.freeze_vision,
    ).to(device)
    crit = SingleBoxLoss().to(device)
    params = [p for p in model.parameters() if p.requires_grad] + \
             [p for p in text.parameters() if p.requires_grad]

    # Pretrained encoders get a much smaller LR than the from-scratch fusion and head.
    # At a shared 5e-4 the 124M-parameter text encoder would overwrite roberta-base's
    # language knowledge long before the 1.3M-parameter fusion learns to use it.
    enc_ids = {id(p) for p in
               list(model.vision.backbone.parameters()) + list(text.encoder.parameters())}
    enc = [p for p in params if id(p) in enc_ids]
    new = [p for p in params if id(p) not in enc_ids]
    groups = [g for g in ({"params": enc, "lr": args.encoder_lr},
                          {"params": new, "lr": args.lr}) if g["params"]]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]

    steps_per_epoch = len(train_dl) // args.accum
    total_steps = steps_per_epoch * args.epochs
    if args.max_iters:
        total_steps = min(total_steps, args.max_iters // args.accum)

    def set_lr(step):
        f = ((step + 1) / args.warmup if step < args.warmup else
             0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup)
                                               / max(1, total_steps - args.warmup)))))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        return f

    tot = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in text.parameters())
    tr = sum(p.numel() for p in params)
    print(f"device {device} | bf16 everywhere (train AND eval) | batch {args.batch_size} "
          f"x accum {args.accum}")
    print(f"attn {args.attn_type} | depth {args.depth} | taps {args.taps}")
    print(f"encoders: vision {'FROZEN' if args.freeze_vision else f'trainable @ lr {args.encoder_lr}'} | "
          f"text {'FROZEN' if args.freeze_text else f'trainable @ lr {args.encoder_lr}'}")
    print(f"trainable {tr/1e6:.2f}M of {tot/1e6:.1f}M  " +
          "  ".join(f"{k} {v/1e6:.2f}M" for k, v in model.trainable_parameters().items()))
    print(f"train {len(train_ds)} | eval {len(eval_src)} of {len(test_ds)}")
    print(f"{steps_per_epoch} steps/epoch, {total_steps} total\n", flush=True)

    start_epoch, best, since_best = 0, -1.0, 0
    if args.resume:
        blob = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"])
        start_epoch, best = blob["epoch"] + 1, blob.get("best", -1.0)
        print(f"resumed {args.resume} at epoch {start_epoch} (best mIoU {best:.4f})")

    log = out / "log.tsv"
    if not log.exists():
        log.write_text("epoch\tstep\tloss\tl1\tciou\ttrain_iou\tlr\tmIoU\tAcc@0.25\t"
                       "Acc@0.5\tAcc@0.75\tAcc@0.9\tblind_mIoU\tdelta\tsec\n")

    step, micro, t0, stop = start_epoch * steps_per_epoch, 0, time.time(), False
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        model.train(); text.train()
        run = {"loss": 0.0, "l1": 0.0, "ciou": 0.0, "iou": 0.0, "n": 0}
        opt.zero_grad(set_to_none=True)
        for cube, ids, mask, gt, _caps in train_dl:
            cube, ids, mask, gt = (t.to(device, non_blocking=True)
                                   for t in (cube, ids, mask, gt))
            with amp_ctx():
                pred = model(cube, text(ids, mask), mask)
            loss, parts = crit(pred.float(), gt)
            (loss / args.accum).backward()
            run["loss"] += loss.item(); run["n"] += 1
            for k in ("l1", "ciou", "iou"):
                run[k] += float(parts[k])
            micro += 1
            if micro % args.accum:
                continue
            f = set_lr(step)
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if args.log_every and step % args.log_every == 0:
                n = max(1, run["n"])
                print(f"  ep {epoch} step {step}/{total_steps}  loss {run['loss']/n:6.3f}  "
                      f"train-IoU {run['iou']/n:.3f}  lr x{f:.3f}  "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
            if step >= total_steps:
                stop = True
                break

        n = max(1, run["n"])
        m = evaluate(model, text, test_dl, device, amp_ctx, tokenizer)
        blind = (evaluate(model, text, test_dl, device, amp_ctx, tokenizer, blind=True)
                 if args.blind_every and epoch % args.blind_every == 0 else None)
        model.train(); text.train()
        delta = (m["mIoU"] - blind["mIoU"]) if blind else float("nan")

        print(f"EPOCH {epoch:3d} | loss {run['loss']/n:6.3f} (l1 {run['l1']/n:.4f} "
              f"ciou {run['ciou']/n:.4f}) train-IoU {run['iou']/n:.3f} | "
              f"mIoU {m['mIoU']:.4f}  " +
              "  ".join(f"Acc@{th} {m[f'Acc@{th}']:.4f}" for th in IOU_THRESHOLDS) +
              (f" | blind {blind['mIoU']:.4f} delta {delta:+.4f}" if blind else "") +
              f" | {(time.time()-t0)/60:.1f} min", flush=True)

        with log.open("a") as fh:
            fh.write(f"{epoch}\t{step}\t{run['loss']/n:.4f}\t{run['l1']/n:.4f}\t"
                     f"{run['ciou']/n:.4f}\t{run['iou']/n:.4f}\t{f:.4f}\t{m['mIoU']:.4f}\t"
                     + "\t".join(f"{m[f'Acc@{th}']:.4f}" for th in IOU_THRESHOLDS)
                     + f"\t{blind['mIoU'] if blind else float('nan'):.4f}\t{delta:.4f}"
                     + f"\t{int(time.time()-t0)}\n")

        blob = {"model": model.state_dict(), "text": text.state_dict(),
                "opt": opt.state_dict(), "epoch": epoch,
                "best": best, "args": vars(args), "metrics": m}
        torch.save(blob, out / "last.pth")
        if m["mIoU"] > best:
            best, since_best = m["mIoU"], 0
            blob["best"] = best
            torch.save(blob, out / "best.pth")
            print(f"         new best mIoU {best:.4f}", flush=True)
        else:
            since_best += 1
            if args.patience and since_best >= args.patience:
                print(f"\nEARLY STOP: {since_best} epochs without improvement.", flush=True)
                stop = True

    print(f"\nfinal full-split evaluation ({len(test_ds)} samples)", flush=True)
    full_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, collate_fn=collate, pin_memory=True)
    m = evaluate(model, text, full_dl, device, amp_ctx, tokenizer)
    b = evaluate(model, text, full_dl, device, amp_ctx, tokenizer, blind=True)
    m["blind_mIoU"], m["caption_delta"] = b["mIoU"], m["mIoU"] - b["mIoU"]
    print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in m.items()), flush=True)
    (out / "final_metrics.json").write_text(json.dumps(m, indent=1))
    print(f"done in {(time.time()-t0)/60:.1f} min | best mIoU {best:.4f}")


if __name__ == "__main__":
    main()
