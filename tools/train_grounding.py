"""Train SpikeTransDVG on Talk2Event: everything trainable except the text encoder.

    uv run python tools/train_grounding.py --run-name grounding_t5 --epochs 2

What "except the text encoder" means here
-----------------------------------------
The head is `SingleBoxHead`: a spiking MLP that regresses four numbers, so the model
emits exactly one box and the objective is a plain L1 + CIoU regression -- no anchors, no
assignment, no classification term.

`freeze_text=True` freezes SpikeLM's 124.1M-parameter BERT and holds it in eval mode.
`SpikeLMTextEncoder.proj` (768 -> 256) stays trainable: it has no donor counterpart by
construction -- it exists only to match the fusion width -- so freezing it would leave a
random projection in the middle of the model forever. Trainable total is 74.9M: the
vision backbone, fusion, neck, head, and that projection.

Two consequences of the freeze worth knowing. SpikeLM's `BertConfig` carries dropout 0.1,
and holding the encoder in eval turns it off, so the text branch becomes deterministic.
And the frozen encoder still runs a full forward every step -- its features could be
precomputed once for all 23025 captions if the text forward ever dominates the step.

Memory, measured on a 32 GiB RTX 5090 at 480x640, T=5, bf16 autocast
--------------------------------------------------------------------
    batch 1, fuse all three scales      17.10 GiB    <- the configuration used here
    batch 2, any fusion setting         OOM

**bf16 autocast is not optional** for this configuration: in fp32 a fully trainable
backbone OOMs at batch 1 (>31.3 GiB) whatever the fusion scales, which is why
`--precision fp32` also forces a partial unfreeze recommendation rather than silently
failing. Batch 1 is a hard ceiling, matching `EventVisionEncoder`'s own documented limit,
so `--accum` provides the effective batch instead. Gradient checkpointing is deliberately
not offered: recomputation would re-run stateful LIF neurons whose membranes have already
advanced, silently producing activations that differ from the forward pass.

BatchNorm at batch 1
--------------------
Every BatchNorm sees T*B = 5 samples, and this model is unusually BatchNorm-sensitive:
an uncalibrated BatchNorm makes the eval path return an all-zero pyramid (see
`CrossModalFusion._scale_attention_bn` and `SpikeTransDVG.calibrate_bn`). `--bn-momentum`
defaults to 0.03 rather than PyTorch's 0.1 so the running statistics average over ~5x
more history and the eval path tracks the train path more closely.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.ops import box_iou

from spiketrandvg.datasets.events_voxel_cube import T_STEPS, talk2event_cube
from spiketrandvg.datasets.talk2event_dataset import Talk2EventDataset
from spiketrandvg.models.grounding_loss import SingleBoxLoss, cxcywh_to_xyxy_norm
from spiketrandvg.models.grounding_model import DEFAULT_TAPS, SpikeTransDVG
from spiketrandvg.models.text_encoder import MAX_TEXT_LEN, build_tokenizer
from spiketrandvg.utils import forks

IOU_THRESHOLDS = (0.25, 0.5, 0.75, 0.9)


class GroundingFrames(Dataset):
    """Talk2Event as (event cube, caption, one box in normalised cxcywh, class id).

    `Talk2EventDataset` yields normalised cxcywh boxes and also decodes the RGB frame,
    which this model never uses; the frame is dropped here rather than upstream so the
    dataset stays the shared one.
    """

    def __init__(self, split: str, attribute: str = "all"):
        self.ds = Talk2EventDataset(SimpleNamespace(attribute=attribute), image_set=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i):
        inp, tgt = self.ds[i]
        cube = talk2event_cube(inp["event"])                # (T, 2, H, W)
        # boxes stay in normalised cxcywh -- the dataset's own units and the head's
        # output units, so no conversion happens anywhere in the loop
        return cube, tgt["caption"], tgt["boxes"][0], tgt["labels"][0]


def make_collate(tokenizer):
    def collate(batch):
        cubes, caps, boxes, labels = zip(*batch)
        tok = tokenizer(
            list(caps), padding="max_length", truncation=True,
            max_length=MAX_TEXT_LEN, return_tensors="pt",
        )
        return (
            torch.stack(cubes, dim=1),          # (T, B, 2, H, W)
            tok["input_ids"],
            tok["attention_mask"],
            torch.stack(boxes),                 # (B, 4) normalised cxcywh
            torch.stack(labels),                # (B,)
        )
    return collate


@torch.no_grad()
def evaluate(model, loader, device, amp_ctx, limit: int | None = None):
    """mean IoU and Acc@thresholds of the model's one predicted box per sample."""
    model.eval()
    ious: list[float] = []
    seen = 0
    for cube, ids, mask, gt, _labels in loader:
        cube, ids, mask, gt = (t.to(device, non_blocking=True) for t in (cube, ids, mask, gt))
        with amp_ctx():
            pred = model(cube, ids, mask)                    # (B, 4) normalised cxcywh
        p, g = cxcywh_to_xyxy_norm(pred.float()), cxcywh_to_xyxy_norm(gt)
        ious += box_iou(p, g).diagonal().tolist()
        seen += len(p)
        if limit is not None and seen >= limit:
            break
    t = torch.tensor(ious)
    return {
        "n": len(ious),
        "mIoU": t.mean().item(),
        **{f"Acc@{th}": (t >= th).float().mean().item() for th in IOU_THRESHOLDS},
    }


def param_groups(model, lr: float, backbone_lr: float, weight_decay: float):
    """Pretrained backbone at a lower rate than the modules trained from scratch.

    Norm layers and biases are excluded from weight decay -- with BatchNorm this
    load-bearing (see the module docstring) decaying its affine parameters toward zero
    would fight the calibration the model depends on.
    """
    def split(mod):
        decay, no_decay = [], []
        for n, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        return decay, no_decay

    bb_d, bb_n = split(model.vision)
    rest = nn.ModuleList([model.fusion, model.neck, model.head, model.text.proj])
    r_d, r_n = split(rest)
    groups = [
        {"params": bb_d, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": bb_n, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": r_d, "lr": lr, "weight_decay": weight_decay},
        {"params": r_n, "lr": lr, "weight_decay": 0.0},
    ]
    return [g for g in groups if g["params"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="grounding_t5")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-iters", type=int, default=None, help="stop early; for smoke tests")
    ap.add_argument("--batch-size", type=int, default=1, help="1 is the memory ceiling")
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-4, help="fusion / neck / head / proj")
    ap.add_argument("--backbone-lr", type=float, default=2e-5, help="pretrained vision")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500, help="optimiser steps")
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--bn-momentum", type=float, default=0.03)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--head-grid", nargs=2, type=int, default=(6, 8),
                    help="spatial grid the head pools to before its MLP")
    ap.add_argument("--head-hidden", type=int, default=512)
    ap.add_argument("--l1-weight", type=float, default=5.0)
    ap.add_argument("--ciou-weight", type=float, default=2.0)
    ap.add_argument("--taps", nargs="+", default=list(DEFAULT_TAPS))
    ap.add_argument("--fuse-scales", nargs="+", default=None)
    ap.add_argument("--attn-bn-gain", type=float, default=3.0)
    ap.add_argument("--ckpt", default="ckpts/spikeformer_v2_weights/55M_kd.pth")
    ap.add_argument("--eval-every", type=int, default=2000, help="optimiser steps")
    ap.add_argument("--eval-samples", type=int, default=600, help="None-like 0 = full split")
    ap.add_argument("--log-every", type=int, default=50, help="optimiser steps")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("runs") / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    if args.precision == "fp32" and args.batch_size >= 1:
        print("[warn] fp32 with a fully trainable backbone OOMs at 480x640 on 32 GiB.\n"
              "       Use --precision bf16, or pass --vision-trainable-from downsample3.")

    amp_ctx = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.precision == "bf16" and device == "cuda"
        else contextlib.nullcontext
    )

    tokenizer = build_tokenizer()
    collate = make_collate(tokenizer)
    train_ds, test_ds = GroundingFrames("train"), GroundingFrames("test")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, collate_fn=collate, drop_last=True,
                          pin_memory=True, persistent_workers=args.workers > 0)
    eval_src = test_ds if not args.eval_samples else Subset(
        test_ds, range(0, len(test_ds), max(1, len(test_ds) // args.eval_samples))
    )
    test_dl = DataLoader(eval_src, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, collate_fn=collate, pin_memory=True)

    ckpt_path = Path(args.ckpt)
    model = SpikeTransDVG(
        taps=tuple(args.taps),
        T=T_STEPS,
        vision_ckpt=str(ckpt_path) if ckpt_path.exists() else None,
        freeze_vision=False,                    # <- the whole point of this run
        text_donor="roberta-base",
        freeze_text=True,                       # <- everything except the text encoder
        fuse_scales=tuple(args.fuse_scales) if args.fuse_scales else None,
        attn_bn_gain=args.attn_bn_gain,
        head_grid=tuple(args.head_grid),
        head_hidden=args.head_hidden,
    ).to(device)

    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = args.bn_momentum

    crit = SingleBoxLoss(args.l1_weight, args.ciou_weight).to(device)
    opt = torch.optim.AdamW(
        param_groups(model, args.lr, args.backbone_lr, args.weight_decay), betas=(0.9, 0.999)
    )
    base_lrs = [g["lr"] for g in opt.param_groups]

    steps_per_epoch = len(train_dl) // args.accum
    total_steps = steps_per_epoch * args.epochs
    if args.max_iters:
        total_steps = min(total_steps, args.max_iters // args.accum)

    def set_lr(step: int) -> float:
        if step < args.warmup:
            f = (step + 1) / args.warmup
        else:
            p = (step - args.warmup) / max(1, total_steps - args.warmup)
            f = 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        return f

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"device {device} | precision {args.precision} | batch {args.batch_size} "
          f"x accum {args.accum} = effective {args.batch_size * args.accum}")
    print(f"trainable {trainable/1e6:.1f}M | frozen {frozen/1e6:.1f}M "
          f"(text encoder {sum(p.numel() for p in model.text.encoder.parameters())/1e6:.1f}M)")
    print(f"train {len(train_ds)} | eval {len(eval_src)} of {len(test_ds)}")
    print(f"{steps_per_epoch} optimiser steps/epoch, {total_steps} total")

    start_step, best = 0, -1.0
    if args.resume:
        blob = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"])
        start_step, best = blob["step"], blob.get("best", -1.0)
        print(f"resumed from {args.resume} at step {start_step} (best Acc@0.5 {best:.4f})")

    log = out / "log.tsv"
    if not log.exists():
        log.write_text("step\tepoch\tloss\tl1\tciou\ttrain_iou\tlr\tmIoU\tAcc@0.5\tAcc@0.75\tsec\n")

    step, micro, t0 = start_step, 0, time.time()
    run = {"loss": 0.0, "l1": 0.0, "ciou": 0.0, "iou": 0.0, "n": 0}
    opt.zero_grad(set_to_none=True)
    stop = False

    for epoch in range(args.epochs):
        if stop:
            break
        model.train()
        for cube, ids, mask, gt, labels in train_dl:
            cube, ids, mask, gt, labels = (
                t.to(device, non_blocking=True) for t in (cube, ids, mask, gt, labels)
            )
            with amp_ctx():
                pred = model(cube, ids, mask)
            loss, parts = crit(pred.float(), gt)
            (loss / args.accum).backward()

            run["loss"] += loss.item(); run["n"] += 1
            for k in ("l1", "ciou", "iou"):
                run[k] += float(parts[k])
            micro += 1

            if micro % args.accum:
                continue

            f = set_lr(step)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                n = max(1, run["n"])
                print(f"step {step:6d}/{total_steps} ep {epoch}  loss {run['loss']/n:7.3f}  "
                      f"l1 {run['l1']/n:.4f} ciou {run['ciou']/n:.4f} "
                      f"train-IoU {run['iou']/n:.3f}  "
                      f"lr x{f:.3f}  {(time.time()-t0)/60:.1f} min", flush=True)

            if step % args.eval_every == 0 or step == total_steps:
                n = max(1, run["n"])
                m = evaluate(model, test_dl, device, amp_ctx)
                model.train()
                with log.open("a") as fh:
                    fh.write(f"{step}\t{epoch}\t{run['loss']/n:.4f}\t{run['l1']/n:.4f}\t"
                             f"{run['ciou']/n:.4f}\t{run['iou']/n:.4f}\t{f:.4f}\t"
                             f"{m['mIoU']:.4f}\t{m['Acc@0.5']:.4f}\t{m['Acc@0.75']:.4f}\t"
                             f"{int(time.time()-t0)}\n")
                print(f"  [eval @ {step}] n={m['n']}  mIoU {m['mIoU']:.4f}  "
                      + "  ".join(f"Acc@{th} {m[f'Acc@{th}']:.4f}" for th in IOU_THRESHOLDS),
                      flush=True)
                run = {"loss": 0.0, "l1": 0.0, "ciou": 0.0, "iou": 0.0, "n": 0}
                blob = {"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best": best, "args": vars(args), "metrics": m}
                torch.save(blob, out / "last.pth")
                if m["Acc@0.5"] > best:
                    best = m["Acc@0.5"]
                    blob["best"] = best
                    torch.save(blob, out / "best.pth")
                    print(f"  new best Acc@0.5 {best:.4f} -> {out/'best.pth'}", flush=True)

            if step >= total_steps:
                stop = True
                break

    # A full-split pass is ~25 min at this model's eval throughput, so it is skipped for
    # the short runs --max-iters exists for; those get the same sampled split as training.
    if args.max_iters:
        print(f"\n--max-iters set: skipping the full-split pass, "
              f"re-running the {len(eval_src)}-sample eval instead")
        full_dl = test_dl
    else:
        print(f"\nfinal full-split evaluation ({len(test_ds)} samples)")
        full_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, collate_fn=collate, pin_memory=True)
    m = evaluate(model, full_dl, device, amp_ctx)
    print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in m.items()))
    (out / "final_metrics.json").write_text(json.dumps(m, indent=1))
    print(f"done in {(time.time()-t0)/60:.1f} min | best Acc@0.5 {best:.4f}")


if __name__ == "__main__":
    main()
