"""Stage 1: train SpikeYOLO on Talk2Event as plain object detection, T=5.

    uv run python tools/train_spikeyolo_det.py --run-name det_t5 --epochs 20

Why this stage exists
---------------------
Training the grounding model end to end never learned to localise: the finished 4-hour run
reached 0.22 mIoU, and a caption-blind control -- same events, deliberately wrong captions
-- scored within 0.001 of it. The model was predicting a salient box from events alone.
An oracle swap confirmed where the error sat: substituting the true box CENTRE more than
doubled mIoU (0.22 -> 0.48) while substituting the true size barely moved it.

Localisation needs no language, so it is learned here first, as ordinary detection on the
same 4433 event frames. The resulting backbone is saved on its own (`backbone.pth`) for
stage 2, where it is frozen and only the cross-attention has to learn selection.

Why batch 4 matters as much as the pretraining
----------------------------------------------
The grounding model could only fit batch 1 at 480x640, and batch-1 BatchNorm has already
caused one measured catastrophe in this project (a regression head whose eval mIoU
collapsed to 0.000 while its training loss fell). This detector is 23M parameters against
199M, so it trains at batch 4 in ~11 GiB -- every BatchNorm sees T*B = 20 samples instead
of 5, which is a materially different optimisation problem.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.ops import batched_nms, box_iou

from spiketrandvg.datasets.event_augment import AugmentConfig
from spiketrandvg.datasets.events_voxel_cube import T_STEPS
from spiketrandvg.datasets.talk2event_frames import (
    CLASSES,
    Talk2EventFrames,
    detection_collate,
)
from spiketrandvg.models.detection_loss import DetectionLoss
from spiketrandvg.models.spikeyolo_detector import SpikeYOLODetector


def xywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def average_precision(tp: torch.Tensor, conf: torch.Tensor, n_gt: int) -> float:
    """101-point interpolated AP from a sorted-by-confidence TP/FP vector."""
    if n_gt == 0:
        return float("nan")
    if tp.numel() == 0:
        return 0.0
    order = conf.argsort(descending=True)
    tp = tp[order]
    tpc = tp.cumsum(0)
    fpc = (1 - tp).cumsum(0)
    recall = tpc / n_gt
    precision = tpc / (tpc + fpc).clamp_min(1e-9)
    # make precision monotonically decreasing, then sample at 101 recall points
    precision = precision.flip(0).cummax(0).values.flip(0)
    pts = torch.linspace(0, 1, 101, device=tp.device)
    idx = torch.searchsorted(recall, pts, right=False).clamp(max=len(recall) - 1)
    return precision[idx].mean().item()


@torch.no_grad()
def evaluate(model, loader, device, amp_ctx, iou_thr=(0.5, 0.75), conf=0.001, max_det=300):
    """mAP at each IoU threshold, over the classes that actually occur."""
    model.eval()
    stats = {t: defaultdict(lambda: {"tp": [], "conf": []}) for t in iou_thr}
    n_gt = defaultdict(int)

    for cube, gtb, gtl, mask, _ in loader:
        cube = cube.to(device, non_blocking=True)
        with amp_ctx():
            pred, _ = model(cube)                      # (B, 4+nc, A)
        pred = pred.float().cpu()
        boxes_all = xywh_to_xyxy(pred[:, :4].permute(0, 2, 1))       # (B, A, 4)
        scores_all = pred[:, 4:].permute(0, 2, 1)                    # (B, A, nc)

        for i in range(pred.shape[0]):
            keep_gt = mask[i, :, 0] > 0
            g_box, g_lab = gtb[i][keep_gt], gtl[i][keep_gt]
            for c in g_lab.tolist():
                n_gt[c] += 1

            sc, cls = scores_all[i].max(dim=1)
            sel = sc > conf
            if sel.sum() == 0:
                continue
            b, s, c = boxes_all[i][sel], sc[sel], cls[sel]
            keep = batched_nms(b, s, c, 0.65)[:max_det]
            b, s, c = b[keep], s[keep], c[keep]

            for t in iou_thr:
                matched = torch.zeros(len(g_box), dtype=torch.bool)
                ious = box_iou(b, g_box) if len(g_box) else torch.zeros(len(b), 0)
                for k in range(len(b)):
                    hit = 0.0
                    if ious.shape[1]:
                        cand = (ious[k] >= t) & (~matched) & (g_lab == c[k])
                        if cand.any():
                            j = torch.where(cand, ious[k], torch.zeros_like(ious[k])).argmax()
                            matched[j] = True
                            hit = 1.0
                    stats[t][int(c[k])]["tp"].append(hit)
                    stats[t][int(c[k])]["conf"].append(float(s[k]))

    out = {}
    for t in iou_thr:
        aps = []
        for c, n in n_gt.items():
            d = stats[t][c]
            tp = torch.tensor(d["tp"]) if d["tp"] else torch.zeros(0)
            cf = torch.tensor(d["conf"]) if d["conf"] else torch.zeros(0)
            ap = average_precision(tp, cf, n)
            if not math.isnan(ap):
                aps.append(ap)
        out[f"mAP@{t}"] = sum(aps) / len(aps) if aps else 0.0
    out["n_gt"] = sum(n_gt.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="det_t5")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=1, help="epochs")
    ap.add_argument("--eval-frames", type=int, default=300, help="0 = full test split")
    ap.add_argument("--patience", type=int, default=30,
                    help="stop after this many epochs with no mAP@0.5 improvement; 0 = off")
    ap.add_argument("--train-eval-every", type=int, default=5,
                    help="also score a slice of TRAIN every N epochs, to see the "
                         "generalisation gap open; 0 = off")
    ap.add_argument("--train-eval-frames", type=int, default=150)
    ap.add_argument("--augment", action="store_true",
                    help="enable event-cube augmentation on the TRAIN split only")
    ap.add_argument("--aug-hflip", type=float, default=0.5)
    ap.add_argument("--aug-scale", type=float, default=0.5)
    ap.add_argument("--aug-mosaic", type=float, default=0.5)
    ap.add_argument("--aug-dropout", type=float, default=0.3)
    ap.add_argument("--log-every", type=int, default=50, help="optimiser steps")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("runs") / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    amp_ctx = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
               if args.precision == "bf16" and device == "cuda" else contextlib.nullcontext)

    aug = AugmentConfig(hflip=args.aug_hflip, scale_jitter=args.aug_scale,
                        mosaic=args.aug_mosaic, event_dropout=args.aug_dropout) \
        if args.augment else None
    train_ds = Talk2EventFrames("train", augment=aug)
    # A SEPARATE un-augmented view of train for the overfit probe. Subsetting train_ds
    # would inherit its augmentation and turn the train/test gap into a comparison of
    # augmented-train against clean-test -- which measures the augmentation, not the gap.
    train_clean = Talk2EventFrames("train")
    test_ds = Talk2EventFrames("test")
    eval_src = test_ds if not args.eval_frames else Subset(
        test_ds, range(0, len(test_ds), max(1, len(test_ds) // args.eval_frames)))
    train_probe = Subset(train_clean, range(0, len(train_clean),
                                        max(1, len(train_clean) // max(1, args.train_eval_frames))))
    train_probe_dl = DataLoader(train_probe, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, collate_fn=detection_collate,
                                pin_memory=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, collate_fn=detection_collate,
                          drop_last=True, pin_memory=True,
                          persistent_workers=args.workers > 0)
    test_dl = DataLoader(eval_src, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, collate_fn=detection_collate,
                         pin_memory=True)

    model = SpikeYOLODetector(num_classes=len(CLASSES), width=args.width, T=T_STEPS).to(device)
    crit = DetectionLoss(model, topk=args.topk).to(device)

    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    steps_per_epoch = len(train_dl) // args.accum
    total_steps = steps_per_epoch * args.epochs
    if args.max_iters:
        total_steps = min(total_steps, args.max_iters // args.accum)

    def set_lr(step):
        f = ((step + 1) / args.warmup if step < args.warmup else
             0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup)
                                               / max(1, total_steps - args.warmup)))))
        for g in opt.param_groups:
            g["lr"] = args.lr * f
        return f

    print(f"device {device} | precision {args.precision} | batch {args.batch_size} "
          f"x accum {args.accum} = effective {args.batch_size * args.accum}")
    print(f"model {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"(backbone {sum(p.numel() for p in model.backbone.parameters())/1e6:.2f}M)")
    print(f"train {len(train_ds)} frames / {train_ds.num_boxes} boxes | "
          f"eval {len(eval_src)} frames")
    print("augmentation: " + (f"hflip {args.aug_hflip} | scale {args.aug_scale} | "
                              f"mosaic {args.aug_mosaic} | event-dropout {args.aug_dropout}"
                              if aug else "NONE"))
    print(f"{steps_per_epoch} optimiser steps/epoch, {total_steps} total", flush=True)

    start_epoch, best = 0, -1.0
    if args.resume:
        blob = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"])
        start_epoch, best = blob["epoch"] + 1, blob.get("best", -1.0)
        print(f"resumed {args.resume} at epoch {start_epoch} (best mAP@0.5 {best:.4f})")

    log = out / "log.tsv"
    if not log.exists():
        log.write_text("epoch\tstep\tloss\tbox\tcls\tdfl\tlr\tmAP@0.5\tmAP@0.75"
                       "\ttrain_mAP@0.5\tgap\tsec\n")

    step, micro, t0 = start_epoch * steps_per_epoch, 0, time.time()
    stop = False
    since_best = 0          # epochs since mAP@0.5 last improved -- drives early stopping
    train_map = float("nan")
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        model.train()
        run = defaultdict(float); run["n"] = 0
        opt.zero_grad(set_to_none=True)
        for cube, gtb, gtl, mask, _ in train_dl:
            cube, gtb, gtl, mask = (t.to(device, non_blocking=True)
                                    for t in (cube, gtb, gtl, mask))
            with amp_ctx():
                feats = model(cube)
            loss, parts = crit([f.float() for f in feats], gtb, gtl, mask)
            (loss / args.accum).backward()
            run["loss"] += loss.item(); run["n"] += 1
            for k in ("box", "cls", "dfl"):
                run[k] += float(parts[k])
            micro += 1
            if micro % args.accum:
                continue

            f = set_lr(step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1

            if step % args.log_every == 0:
                n = max(1, run["n"])
                print(f"ep {epoch} step {step:6d}/{total_steps}  loss {run['loss']/n:7.3f}  "
                      f"box {run['box']/n:.3f} cls {run['cls']/n:.3f} dfl {run['dfl']/n:.3f}  "
                      f"lr x{f:.3f}  {(time.time()-t0)/60:.1f} min", flush=True)
            if step >= total_steps:
                stop = True
                break

        if (epoch + 1) % args.eval_every == 0 or stop:
            m = evaluate(model, test_dl, device, amp_ctx)
            # Recomputed on EVERY eval, never carried over. `gap` is only meaningful if
            # both halves come from the same weights; keeping a stale train score and
            # differencing a fresh test score against it yields a number that looks like
            # a generalisation gap and is not one. NaN on skipped epochs is the honest
            # value, and both the console line and the TSV suppress it.
            train_map = (
                evaluate(model, train_probe_dl, device, amp_ctx,
                         iou_thr=(0.5,))["mAP@0.5"]
                if args.train_eval_every and epoch % args.train_eval_every == 0
                else float("nan")
            )
            model.train()
            n = max(1, run["n"])
            with log.open("a") as fh:
                gap = train_map - m["mAP@0.5"]
                fh.write(f"{epoch}\t{step}\t{run['loss']/n:.4f}\t{run['box']/n:.4f}\t"
                         f"{run['cls']/n:.4f}\t{run['dfl']/n:.4f}\t{f:.4f}\t"
                         f"{m['mAP@0.5']:.4f}\t{m['mAP@0.75']:.4f}\t{train_map:.4f}\t"
                         f"{gap:.4f}\t{int(time.time()-t0)}\n")
            extra = ("" if train_map != train_map else
                     f"  | train {train_map:.4f} (gap {train_map - m['mAP@0.5']:+.4f})")
            print(f"  [eval ep {epoch}] mAP@0.5 {m['mAP@0.5']:.4f}  "
                  f"mAP@0.75 {m['mAP@0.75']:.4f}{extra}  "
                  f"[{since_best}/{args.patience} since best]", flush=True)
            blob = {"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best": best, "args": vars(args), "metrics": m}
            torch.save(blob, out / "last.pth")
            if m["mAP@0.5"] > best:
                best = m["mAP@0.5"]
                blob["best"] = best
                torch.save(blob, out / "best.pth")
                # the artefact stage 2 actually consumes
                torch.save({"backbone": model.backbone.state_dict(),
                            "width": args.width, "channels": model.backbone.channels,
                            "epoch": epoch, "mAP@0.5": best},
                           out / "backbone.pth")
                since_best = 0
                print(f"  new best mAP@0.5 {best:.4f} -> {out/'best.pth'} "
                      f"(+ backbone.pth for stage 2)", flush=True)
            else:
                since_best += 1
                if args.patience and since_best >= args.patience:
                    print(f"\nEARLY STOP: {since_best} epochs without improving on "
                          f"mAP@0.5 {best:.4f}. Stopping at epoch {epoch} of "
                          f"{args.epochs}.", flush=True)
                    stop = True

    print(f"\nfinal full-split evaluation ({len(test_ds)} frames)")
    full_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, collate_fn=detection_collate,
                         pin_memory=True)
    m = evaluate(model, full_dl, device, amp_ctx)
    print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in m.items()))
    (out / "final_metrics.json").write_text(json.dumps(m, indent=1))
    print(f"done in {(time.time()-t0)/60:.1f} min | best mAP@0.5 {best:.4f}")


if __name__ == "__main__":
    main()
