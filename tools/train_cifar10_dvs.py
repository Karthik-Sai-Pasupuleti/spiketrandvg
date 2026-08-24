"""Sanity-check the event vision encoder by classifying CIFAR10-DVS at T=5.

This is a capability test for the encoder, not a benchmark chase: if it cannot learn
10-way classification from event cubes, it will not support referring-expression
grounding either.

    uv run python tools/train_cifar10_dvs.py --epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spiketrandvg.datasets.cifar10_dvs import CLASSES, CIFAR10DVSVoxel
from spiketrandvg.models.event_classifier import EventClassifier


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, dict[str, float]]:
    model.eval()
    correct = total = 0
    per_cls_hit = [0] * len(CLASSES)
    per_cls_tot = [0] * len(CLASSES)
    for cube, label in loader:
        cube = cube.to(device, non_blocking=True).permute(1, 0, 2, 3, 4)  # (T,B,C,H,W)
        label = label.to(device, non_blocking=True)
        pred = model(cube).argmax(1)
        correct += (pred == label).sum().item()
        total += label.numel()
        for p, l in zip(pred.tolist(), label.tolist()):
            per_cls_tot[l] += 1
            per_cls_hit[l] += int(p == l)
    per_cls = {
        CLASSES[c]: (per_cls_hit[c] / per_cls_tot[c] if per_cls_tot[c] else 0.0)
        for c in range(len(CLASSES))
    }
    return correct / max(total, 1), per_cls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tap", default="s16b")
    ap.add_argument("--ckpt", default="ckpts/spikeformer_v2_weights/55M_kd_T4.pth",
                    help="Meta-SpikeFormer ImageNet init; pass '' for random init")
    ap.add_argument("--trainable-from", default=None,
                    help="freeze stages before this one, e.g. downsample3")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--run-name", default="cifar10dvs")
    ap.add_argument("--max-iters", type=int, default=None, help="cap iters/epoch (debug)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    run = Path("runs") / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=1))

    train_ds = CIFAR10DVSVoxel(split="train", T=args.T, workers=args.workers)
    test_ds = CIFAR10DVSVoxel(split="test", T=args.T, workers=args.workers)
    print(f"train {len(train_ds)} | test {len(test_ds)} | classes {len(CLASSES)}")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    model = EventClassifier(
        num_classes=len(CLASSES), tap=args.tap,
        ckpt_path=args.ckpt or None,
        freeze_backbone=args.freeze_backbone,
        trainable_from=args.trainable_from,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"({trainable/1e6:.1f}M trainable)")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    log = run / "log.tsv"
    if not log.exists():
        log.write_text("epoch\ttrain_loss\ttrain_acc\ttest_acc\tsec\n")

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0, run_loss, hit, seen = time.time(), 0.0, 0, 0
        for i, (cube, label) in enumerate(train_ld):
            if args.max_iters and i >= args.max_iters:
                break
            cube = cube.to(device, non_blocking=True).permute(1, 0, 2, 3, 4)
            label = label.to(device, non_blocking=True)
            logits = model(cube)
            loss = crit(logits, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            run_loss += loss.item() * label.numel()
            hit += (logits.argmax(1) == label).sum().item()
            seen += label.numel()
            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch} iter {i+1}/{len(train_ld)} "
                      f"loss {run_loss/seen:.4f} acc {hit/seen:.4f}", flush=True)
        sched.step()
        test_acc, per_cls = evaluate(model, test_ld, device)
        dt = time.time() - t0
        print(f"== epoch {epoch}: train_loss {run_loss/max(seen,1):.4f} "
              f"train_acc {hit/max(seen,1):.4f} | TEST ACC {test_acc:.4f} | {dt:.0f}s",
              flush=True)
        with log.open("a") as f:
            f.write(f"{epoch}\t{run_loss/max(seen,1):.4f}\t{hit/max(seen,1):.4f}"
                    f"\t{test_acc:.4f}\t{dt:.0f}\n")
        if test_acc > best:
            best = test_acc
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "test_acc": test_acc}, run / "best.pth")
            print(f"   new best {best:.4f}; per-class " +
                  ", ".join(f"{k} {v:.2f}" for k, v in per_cls.items()), flush=True)

    print(f"\nbest test accuracy: {best:.4f}  (chance = {1/len(CLASSES):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
