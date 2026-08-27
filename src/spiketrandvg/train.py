"""Train RefCOCOGrounding on RefCOCO / RefCOCO+ / RefCOCOg.

    uv run python -m spiketrandvg.train --run-name refcoco_b1 --epochs 20

Frozen: roberta-base (124.6M). Trainable: everything else -- the SpiLiFormer backbone,
its lateral+positional projection, the text projection, the cross-attention stack and the
box head (66.19M).

Two learning rates
------------------
The SpiLiFormer backbone carries an ImageNet prior worth keeping and gets `--encoder-lr`
(1e-5); everything else is randomly initialised and gets `--lr` (5e-4). At a single shared
5e-4 the 64M-parameter backbone would overwrite its pretraining long before the 1.3M
cross-attention learns to use it.

bf16 in training AND evaluation
-------------------------------
Not an optimisation -- a correctness requirement. Spiking layers are not dtype-agnostic:
precision decides which membranes cross threshold, so a bf16-trained model evaluated in
fp32 is a different function. Measured on a same-frame overfit set, identical weights and
mode: bf16 0.656 IoU vs fp32 0.060. An earlier harness in this project trained bf16 and
measured fp32, and the resulting "this architecture cannot fit 16 samples" conclusion was
an artefact of that mismatch.

The caption-blind control, and why it is not a caption shuffle
---------------------------------------------------------------
`blind_mIoU` re-runs the eval with the right image and the WRONG caption; the delta
against the normal score is how much of the model's localisation is actually attributable
to language. A model with delta ~0 is a detector with a text input it ignores, which is
exactly what this project measured for 85 epochs before catching it.

Getting the negative right is the whole game. The obvious implementation -- rotate the
captions within each batch -- is WRONG on RefCOCO, measured: 56.7% of the pairs it forms
share the same object, because RefCOCO gives each object ~2.8 expressions and stores them
consecutively. A different expression for the same box is a paraphrase, not a negative;
scoring the model as wrong for predicting the same box makes the delta collapse toward
zero and hides real grounding. The identical bug on Talk2Event (50.0% same-object) turned
a true delta of +0.22 into a reported +0.017.

So negatives are precomputed per sample: a caption belonging to a DIFFERENT object in the
SAME image, which is available for 97.8% of RefCOCO val. Same scene, same distractors,
only the referent differs -- the hardest honest negative, and the one that measures
referring-expression grounding rather than scene recognition. Samples with no such sibling
fall back to a different image and are counted separately.

train_mIoU: the metric that catches a broken eval path
--------------------------------------------------------
The `train-IoU` printed during the epoch loop is measured DURING training -- BatchNorm on
batch statistics, dropout active where applicable, LIF thresholds responding to whatever
that specific mini-batch looks like. `evaluate()` runs in eval mode: running BatchNorm
statistics, dropout off. Both respond to the SAME weights differently, so a model that has
genuinely fitted its training set can look identical, in the epoch line, to one whose eval
path is silently broken -- the only way to tell them apart is to score the training set
itself through `evaluate()`. `train_mIoU` in `log.tsv` and the printed `(mode gap ...)`
are exactly that: a `RefCOCO(split="train")` set, unaugmented, scored the same way `val`
is. A large, persistent gap here means look here BEFORE trusting any other number in this
file.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.ops import box_iou

import torch.nn.functional as F

from spiketrandvg.dataloader import (NO_ATTR, RefCOCO, RefCOCOAugment, SPLITS, Talk2Event,
                                     make_collate, make_t2e_collate)
from spiketrandvg.model import (RefCOCOGrounding, SingleBoxLoss, Talk2EventGrounding,
                                cxcywh_to_xyxy_norm)
from spiketrandvg.textencoder import build_tokenizer

IOU_THRESHOLDS = (0.25, 0.5, 0.75, 0.9)


# --------------------------------------------------------------------------- negatives
def hard_negative_captions(ds: RefCOCO, seed: int = 0) -> tuple[list[str], list[bool]]:
    """For each sample, a caption naming a DIFFERENT object in the SAME image.

    Returns `(captions, is_hard)`. `is_hard[i]` is False where the image held no other
    object and the negative had to come from a different image -- those are easy negatives
    and are excluded from the headline delta so it cannot be inflated by them.
    """
    rng = random.Random(seed)
    okey = lambda im, a: (im["file_name"], tuple(a["bbox"]))
    obj = [okey(im, a) for im, a in ds.items]
    caps = [im["caption"] for im, _ in ds.items]
    by_img: dict[str, list[int]] = defaultdict(list)
    for i, (im, _) in enumerate(ds.items):
        by_img[im["file_name"]].append(i)

    neg, hard = [], []
    n = len(ds.items)
    for i, (im, _) in enumerate(ds.items):
        sib = [j for j in by_img[im["file_name"]] if obj[j] != obj[i]]
        if sib:
            neg.append(caps[rng.choice(sib)]); hard.append(True)
        else:
            j = rng.randrange(n)
            while ds.items[j][0]["file_name"] == im["file_name"]:
                j = rng.randrange(n)
            neg.append(caps[j]); hard.append(False)
    return neg, hard


class _CaptionOverride(Dataset):
    """The same images and boxes, each paired with a supplied caption."""

    def __init__(self, ds: RefCOCO, captions: list[str]):
        self.ds, self.captions = ds, captions

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, _cap, box, meta = self.ds[i]
        return img, self.captions[i], box, meta


# ------------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, device, amp_ctx, keep: list[bool] | None = None):
    """mIoU and Acc@thresholds. `keep` masks out samples that should not count."""
    model.eval()
    ious, flags = [], []
    seen = 0
    for rgb, ids, mask, gt, _caps, _metas in loader:
        rgb, ids, mask, gt = (t.to(device, non_blocking=True) for t in (rgb, ids, mask, gt))
        with amp_ctx():
            pred = model(rgb, ids, mask)
        i = box_iou(cxcywh_to_xyxy_norm(pred.float()), cxcywh_to_xyxy_norm(gt)).diagonal()
        ious.append(i.cpu())
        if keep is not None:
            flags.extend(keep[seen:seen + len(i)])
        seen += len(i)
    t = torch.cat(ious)
    if keep is not None:
        t = t[torch.tensor(flags, dtype=torch.bool)]
    return {"n": len(t), "mIoU": t.mean().item(),
            **{f"Acc@{th}": (t >= th).float().mean().item() for th in IOU_THRESHOLDS}}


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="refcoco", choices=["refcoco", "talk2event"],
                    help="refcoco: RGB referring expressions (RefCOCOGrounding). "
                         "talk2event: event streams (Talk2EventGrounding)")
    ap.add_argument("--run-name", default="refcoco")
    ap.add_argument("--dataset", default="refcoco", choices=sorted(SPLITS))
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4, help="from-scratch modules")
    ap.add_argument("--encoder-lr", type=float, default=1e-5, help="SpiLiFormer backbone")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500, help="optimiser steps")
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--size", type=int, nargs=2, default=(384, 384))
    ap.add_argument("--T", type=int, default=4, help="fusion timesteps")
    ap.add_argument("--rgb-T", type=int, default=1, help="SpiLiFormer internal timesteps")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--attn-type", default="spatial_softmax",
                    choices=["spatial_softmax", "cmsf_linear"])
    ap.add_argument("--head-type", default="pooled_mlp",
                    choices=["pooled_mlp", "attn_softargmax"],
                    help="pooled_mlp: centre from a binarised pooled vector (prior "
                         "behaviour, plus head_norm0). attn_softargmax: centre read "
                         "off the attention map itself, before it is binarised -- "
                         "see model.py RefCOCOGrounding._softargmax_box")
    ap.add_argument("--attn-map", default="last", choices=["last", "mean"],
                    help="which block's attention map attn_softargmax reads")
    ap.add_argument("--pos-std", type=float, default=0.02,
                    help="init std of the vision positional embedding")
    ap.add_argument("--center-weight", type=float, default=1.0,
                    help="extra weight on the L1 cx/cy terms (w/h stay at 1.0)")
    ap.add_argument("--rgb-ckpt", default="ckpts/spiliformer/checkpoint_spiliformer_T4_224.pth")
    ap.add_argument("--text-model", default="roberta-base")
    # The text encoder is frozen and the flag exists to record that choice in args.json,
    # not to invite flipping it: roberta is the one component here that is genuinely
    # pretrained for its job, and letting 124M parameters move at grounding's batch size
    # overwrites that before the 1.3M fusion learns to read it.
    ap.add_argument("--train-text", action="store_true",
                    help="unfreeze roberta (default: frozen)")
    ap.add_argument("--text-unfreeze-last", type=int, default=0,
                    help="unfreeze only the last N roberta layers, at --encoder-lr; "
                         "ignored if --train-text is set (already fully unfrozen)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-samples", type=int, default=2000,
                    help="strided subset of the val split; 0 = all")
    ap.add_argument("--train-eval-samples", type=int, default=2000,
                    help="strided subset of TRAIN scored in eval mode every epoch "
                         "(train_mIoU); 0 = all. See the module docstring.")
    ap.add_argument("--blind-every", type=int, default=1, help="0 disables")
    ap.add_argument("--patience", type=int, default=8, help="0 disables early stopping")
    ap.add_argument("--keep-all-best", action="store_true",
                    help="also archive a weights-only best_ep<N>_miou<X>.pth on every new "
                         "best, instead of only overwriting best.pth. ~780MB each -- see "
                         "the disk note before enabling for a long run")
    ap.add_argument("--log-every", type=int, default=200, help="optimiser steps")
    ap.add_argument("--max-iters", type=int, default=None, help="smoke test")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--resume", default=None)

    # --- talk2event only ---------------------------------------------------------
    t2e = ap.add_argument_group("talk2event")
    t2e.add_argument("--event-backbone", default="spiliformer_dvs",
                     choices=["spiliformer_dvs", "metaspikformer"],
                     help="spiliformer_dvs: 1.70M, event-native, no pretrained weights. "
                          "metaspikformer: 54.71M, ImageNet-pretrained, adapted stem")
    t2e.add_argument("--event-ckpt", default="",
                     help="checkpoint for the event backbone; only metaspikformer has one")
    t2e.add_argument("--n-slots", type=int, default=1000,
                     help="SlotBoxHead bins per coordinate, over the annotation frame")
    t2e.add_argument("--slot-weight", type=float, default=1.0,
                     help="cross-entropy on the true slot; 0 leaves only the soft "
                          "expectation gradient the box loss provides")
    t2e.add_argument("--tag-weight", type=float, default=0.5,
                     help="cross-entropy on the attribute span labels; 0 lets the tagger "
                          "learn any partition, which is a valid ablation but is no "
                          "longer 'the four attributes'")
    t2e.add_argument("--no-condition", action="store_true",
                     help="disable ThresholdModulator -- the language-blind-encoder "
                          "ablation the conditioned model is measured against")
    t2e.add_argument("--no-ilif", action="store_true",
                     help="keep binary LIF instead of integer I-LIF")
    t2e.add_argument("--freeze-event", action="store_true")

    args = ap.parse_args()

    if args.task == "talk2event":
        # Talk2Event's own defaults differ from RefCOCO's, and argparse cannot express
        # "default depends on another flag". Only override values the user left at the
        # RefCOCO default, so an explicit flag always wins.
        d = ap.parse_args([])                       # what the defaults would have been
        if args.size == d.size:       args.size = (480, 640)
        if args.T == d.T:             args.T = 5
        if args.val_split == d.val_split:   args.val_split = "test"
        if args.run_name == d.run_name:     args.run_name = "talk2event"
        if args.batch_size == d.batch_size: args.batch_size = 4
        return main_t2e(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("runs") / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    amp_ctx = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
               if device == "cuda" else contextlib.nullcontext)

    tokenizer = build_tokenizer()
    collate = make_collate(tokenizer)
    size = tuple(args.size)
    aug = None if args.no_augment else RefCOCOAugment(hflip=0.5, jitter=args.jitter)

    train_ds = RefCOCO(args.dataset, "train", size=size, augment=aug, limit=args.limit_train)
    val_ds = RefCOCO(args.dataset, args.val_split, size=size, augment=None)
    # Same samples the model trains on (respects --limit-train), unaugmented, scored
    # through the same evaluate() as val -- see the module docstring on train_mIoU.
    train_eval_ds = RefCOCO(args.dataset, "train", size=size, augment=None,
                            limit=args.limit_train)

    # Precomputed hard negatives -- see the module docstring for why a batch shuffle is
    # not an acceptable substitute here.
    neg_caps, is_hard = hard_negative_captions(val_ds)
    blind_ds = _CaptionOverride(val_ds, neg_caps)

    idxs = (list(range(len(val_ds))) if not args.eval_samples else
            list(range(0, len(val_ds), max(1, len(val_ds) // args.eval_samples))))
    keep = [is_hard[i] for i in idxs]
    train_eval_idxs = (list(range(len(train_eval_ds))) if not args.train_eval_samples else
                       list(range(0, len(train_eval_ds),
                                  max(1, len(train_eval_ds) // args.train_eval_samples))))

    def loader(ds, shuffle=False, subset=None):
        src = Subset(ds, subset) if subset is not None else ds
        return DataLoader(src, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers, collate_fn=collate, pin_memory=True,
                          drop_last=shuffle, persistent_workers=args.workers > 0)

    train_dl = loader(train_ds, shuffle=True)
    val_dl = loader(val_ds, subset=idxs)
    blind_dl = loader(blind_ds, subset=idxs)
    train_eval_dl = loader(train_eval_ds, subset=train_eval_idxs)

    model = RefCOCOGrounding(
        rgb_ckpt=args.rgb_ckpt, text_model=args.text_model, img_size=size,
        T=args.T, rgb_T=args.rgb_T, depth=args.depth, attn_type=args.attn_type,
        freeze_rgb=False, freeze_text=not args.train_text,
        head_type=args.head_type, attn_map=args.attn_map, pos_std=args.pos_std,
        text_unfreeze_last=0 if args.train_text else args.text_unfreeze_last,
    ).to(device)
    crit = SingleBoxLoss(center_weight=args.center_weight).to(device)

    # Both the vision backbone AND any unfrozen tail of roberta (item 8) are pretrained
    # weights and share encoder_lr; "new" is everything randomly initialised.
    pretrained_sources = list(model.vision.backbone.parameters()) + \
        list(model.text.encoder.parameters())
    backbone_ids = {id(p) for p in pretrained_sources}
    enc = [p for p in model.parameters() if p.requires_grad and id(p) in backbone_ids]
    new = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    groups = [g for g in ({"params": enc, "lr": args.encoder_lr},
                          {"params": new, "lr": args.lr}) if g["params"]]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]

    steps_per_epoch = max(1, len(train_dl) // args.accum)
    total_steps = steps_per_epoch * args.epochs

    def set_lr(step: int) -> float:
        """Linear warmup then cosine decay, as a multiplier on each group's base lr."""
        if step < args.warmup:
            m = (step + 1) / max(1, args.warmup)
        else:
            p = (step - args.warmup) / max(1, total_steps - args.warmup)
            m = 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * m
        return m

    start_epoch, best, since_best, gstep = 0, -1.0, 0, 0
    if args.resume:
        blob = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"])
        start_epoch, best = blob["epoch"] + 1, blob.get("best", -1.0)
        gstep = blob.get("gstep", start_epoch * steps_per_epoch)
        print(f"resumed {args.resume} at epoch {start_epoch} (best mIoU {best:.4f})")

    tp = model.trainable_parameters()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"device {device} | bf16 train AND eval | batch {args.batch_size} x accum {args.accum}")
    print(f"{args.dataset}: train {len(train_ds)} | val {len(val_ds)} "
          f"(eval on {len(idxs)}, {sum(keep)} with a same-image hard negative)")
    print(f"attn {args.attn_type} | head {args.head_type} (attn_map={args.attn_map}) | "
          f"depth {args.depth} | T {args.T} | size {size[0]}x{size[1]}")
    text_state = ("TRAINABLE" if args.train_text else
                 f"last {args.text_unfreeze_last} layers @ lr {args.encoder_lr}"
                 if args.text_unfreeze_last else "FROZEN")
    print(f"text {args.text_model} {text_state} | "
          f"backbone @ lr {args.encoder_lr} | new modules @ lr {args.lr}")
    print(f"trainable {n_tr/1e6:.2f}M of {n_all/1e6:.1f}M  " +
          "  ".join(f"{k} {v/1e6:.2f}M" for k, v in tp.items()))
    print(f"{steps_per_epoch} steps/epoch, {total_steps} total\n", flush=True)

    if not (out / "log.tsv").exists():
        (out / "log.tsv").write_text(
            "epoch\tstep\tloss\tl1\tciou\ttrain_iou\ttrain_mIoU\tlr\tmIoU\tAcc@0.25\t"
            "Acc@0.5\tAcc@0.75\tAcc@0.9\tblind_mIoU\tdelta\tsec\n")

    t0 = time.time()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run = defaultdict(float)
        seen = 0
        opt.zero_grad(set_to_none=True)
        for it, (rgb, ids, mask, gt, _c, _m) in enumerate(train_dl):
            rgb, ids, mask, gt = (t.to(device, non_blocking=True)
                                  for t in (rgb, ids, mask, gt))
            with amp_ctx():
                pred = model(rgb, ids, mask)
            loss, parts = crit(pred.float(), gt)
            (loss / args.accum).backward()

            if (it + 1) % args.accum == 0:
                mult = set_lr(gstep)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.clip)
                opt.step(); opt.zero_grad(set_to_none=True)
                gstep += 1

            b = gt.shape[0]
            seen += b
            run["loss"] += loss.item() * b
            for k, v in parts.items():
                run[k] += float(v) * b

            if args.log_every and (it + 1) % (args.log_every * args.accum) == 0:
                print(f"  ep {epoch} step {gstep}/{total_steps}  "
                      f"loss {run['loss']/seen:6.3f}  train-IoU {run['iou']/seen:.3f}  "
                      f"lr x{mult:.3f}  {(time.time()-t0)/60:.1f} min", flush=True)
            if args.max_iters and it + 1 >= args.max_iters:
                break

        m = evaluate(model, val_dl, device, amp_ctx)
        train_m = evaluate(model, train_eval_dl, device, amp_ctx)
        mode_gap = train_m["mIoU"] - run["iou"] / seen
        blind = (evaluate(model, blind_dl, device, amp_ctx, keep=keep)
                 if args.blind_every and epoch % args.blind_every == 0 else None)
        # delta is computed on the SAME subset the blind score used -- the samples that
        # have a real same-image negative -- so the two numbers are comparable
        m_hard = (evaluate(model, val_dl, device, amp_ctx, keep=keep)
                  if blind is not None else None)
        delta = (m_hard["mIoU"] - blind["mIoU"]) if blind else float("nan")

        print(f"EPOCH {epoch:3d} | loss {run['loss']/seen:6.3f} "
              f"(l1 {run['l1']/seen:.4f} ciou {run['ciou']/seen:.4f}) "
              f"train-IoU {run['iou']/seen:.3f} (mode gap {mode_gap:+.4f}) | "
              f"mIoU {m['mIoU']:.4f}  Acc@0.25 {m['Acc@0.25']:.4f}  "
              f"Acc@0.5 {m['Acc@0.5']:.4f}  Acc@0.75 {m['Acc@0.75']:.4f}  "
              f"Acc@0.9 {m['Acc@0.9']:.4f}"
              + (f" | blind {blind['mIoU']:.4f} delta {delta:+.4f}" if blind else "")
              + f" | {(time.time()-t0)/60:.1f} min", flush=True)

        with open(out / "log.tsv", "a") as f:
            f.write(f"{epoch}\t{gstep}\t{run['loss']/seen:.4f}\t{run['l1']/seen:.4f}\t"
                    f"{run['ciou']/seen:.4f}\t{run['iou']/seen:.4f}\t{train_m['mIoU']:.4f}\t"
                    f"{mult:.4f}\t"
                    f"{m['mIoU']:.4f}\t{m['Acc@0.25']:.4f}\t{m['Acc@0.5']:.4f}\t"
                    f"{m['Acc@0.75']:.4f}\t{m['Acc@0.9']:.4f}\t"
                    f"{blind['mIoU'] if blind else float('nan'):.4f}\t{delta:.4f}\t"
                    f"{int(time.time()-t0)}\n")

        blob = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                "gstep": gstep, "args": vars(args), "metrics": m}
        torch.save(blob, out / "last.pth")
        if m["mIoU"] > best:
            best, since_best = m["mIoU"], 0
            blob["best"] = best
            torch.save(blob, out / "best.pth")
            print(f"         new best mIoU {best:.4f}", flush=True)
            if args.keep_all_best:
                # Archive copy, model weights ONLY -- no optimiser state, which is ~40% of
                # the file and is useless for a snapshot you will never resume from.
                # `best.pth` above stays the full resumable checkpoint.
                snap = out / f"best_ep{epoch:03d}_miou{m['mIoU']:.4f}.pth"
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "args": vars(args), "metrics": m, "best": best}, snap)
                print(f"         archived {snap.name}", flush=True)
        else:
            since_best += 1
            if args.patience and since_best >= args.patience:
                print(f"\nEARLY STOP: {since_best} epochs without improvement.", flush=True)
                stop = True
        if stop or args.max_iters:
            break

    # Final pass on the FULL val split with the best weights, not the last ones.
    ck = out / "best.pth"
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=device,
                                         weights_only=False)["model"])
    print(f"\nfinal evaluation on all {len(val_ds)} {args.val_split} samples "
          f"(best checkpoint)", flush=True)
    full, full_blind = loader(val_ds), loader(blind_ds)
    m = evaluate(model, full, device, amp_ctx)
    b = evaluate(model, full_blind, device, amp_ctx, keep=is_hard)
    m_hard = evaluate(model, full, device, amp_ctx, keep=is_hard)
    m["blind_mIoU"] = b["mIoU"]
    m["mIoU_on_hard_subset"] = m_hard["mIoU"]
    m["caption_delta"] = m_hard["mIoU"] - b["mIoU"]
    m["n_hard"] = int(sum(is_hard))
    m["_delta_note"] = ("negatives are a caption for a DIFFERENT object in the SAME image; "
                        "delta compares only the n_hard samples that have one")
    print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in m.items() if not k.startswith("_")), flush=True)
    (out / "final_metrics.json").write_text(json.dumps(m, indent=1))
    print(f"done in {(time.time()-t0)/60:.1f} min | best val mIoU {best:.4f}")




# ======================================================================================
# Talk2Event
# ======================================================================================

def t2e_hard_negatives(ds: Talk2Event, seed: int = 0) -> tuple[list[str], list[bool]]:
    """A caption naming a DIFFERENT object in the SAME event frame, per sample.

    Same reasoning as `hard_negative_captions` for RefCOCO, and the same trap: Talk2Event
    gives each object several paraphrases, so any position-based shuffle would score
    paraphrases as negatives and collapse the delta toward zero. The negative is chosen by
    OBJECT IDENTITY -- `(event_path, bbox)` -- never by index.

    `is_hard[i]` is False where the frame held no other object; those fall back to a
    different frame and are excluded from the headline delta, since rejecting a caption
    from another scene needs no grounding at all.
    """
    rng = random.Random(seed)
    recs = ds.items
    okey = lambda r: (r["event_path"], r["bbox"]["x"], r["bbox"]["y"],
                      r["bbox"]["w"], r["bbox"]["h"])
    obj = [okey(r) for r in recs]
    caps = [ds._caption_and_attrs(r, i)[0] for i, r in enumerate(recs)]
    by_frame: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        by_frame[r["event_path"]].append(i)

    neg, hard, n = [], [], len(recs)
    for i, r in enumerate(recs):
        sib = [j for j in by_frame[r["event_path"]] if obj[j] != obj[i]]
        if sib:
            neg.append(caps[rng.choice(sib)]); hard.append(True)
        else:
            j = rng.randrange(n)
            while recs[j]["event_path"] == r["event_path"]:
                j = rng.randrange(n)
            neg.append(caps[j]); hard.append(False)
    return neg, hard


class _T2ECaptionOverride(Dataset):
    """Same cubes and boxes, each paired with a supplied caption.

    Attribute spans are emitted empty: they only feed the auxiliary tagger loss, which is
    a training-time signal, and a span located in the ORIGINAL caption would be wrong for
    the substituted one.
    """

    def __init__(self, ds: Talk2Event, captions: list[str]):
        self.ds, self.captions = ds, captions

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        cube, _cap, box, _spans, rec = self.ds[i]
        return cube, self.captions[i], box, [], rec


@torch.no_grad()
def evaluate_t2e(model, loader, device, amp_ctx, keep: list[bool] | None = None):
    """mIoU and Acc@thresholds for Talk2EventGrounding (which returns a dict)."""
    model.eval()
    ious, flags, seen = [], [], 0
    for cube, ids, mask, gt, _lab, _caps, _recs in loader:
        cube, ids, mask, gt = (t.to(device, non_blocking=True)
                               for t in (cube, ids, mask, gt))
        with amp_ctx():
            out = model(cube, ids, mask)
        i = box_iou(cxcywh_to_xyxy_norm(out["box"].float()),
                    cxcywh_to_xyxy_norm(gt)).diagonal()
        ious.append(i.cpu())
        if keep is not None:
            flags.extend(keep[seen:seen + len(i)])
        seen += len(i)
    t = torch.cat(ious)
    if keep is not None:
        t = t[torch.tensor(flags, dtype=torch.bool)]
    return {"n": len(t), "mIoU": t.mean().item(),
            **{f"Acc@{th}": (t >= th).float().mean().item() for th in IOU_THRESHOLDS}}


def slot_targets(box: torch.Tensor, n_slots: int) -> torch.Tensor:
    """(B,4) normalised cxcywh -> (B,4) int64 slot indices for the cross-entropy.

    The box loss already reaches `slot_logits` through the soft-argmax expectation, but
    only as a diffuse signal: many distributions share a mean. A direct cross-entropy on
    the slot that CONTAINS the true coordinate is the supervision the head's
    "four multiple-choice questions" framing actually implies, and it is what makes the
    argmax (rather than just the expectation) meaningful.
    """
    return (box.clamp(0, 1 - 1e-6) * n_slots).long().clamp(0, n_slots - 1)


def main_t2e(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("runs") / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    amp_ctx = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
               if device == "cuda" else contextlib.nullcontext)

    tokenizer = build_tokenizer()
    collate = make_t2e_collate(tokenizer)
    train_ds = Talk2Event("train", t_steps=args.T, limit=args.limit_train)
    val_ds = Talk2Event(args.val_split, t_steps=args.T)
    train_eval_ds = Talk2Event("train", t_steps=args.T, limit=args.limit_train)

    neg_caps, is_hard = t2e_hard_negatives(val_ds)
    blind_ds = _T2ECaptionOverride(val_ds, neg_caps)

    idxs = (list(range(len(val_ds))) if not args.eval_samples else
            list(range(0, len(val_ds), max(1, len(val_ds) // args.eval_samples))))
    keep = [is_hard[i] for i in idxs]
    tr_idxs = (list(range(len(train_eval_ds))) if not args.train_eval_samples else
               list(range(0, len(train_eval_ds),
                          max(1, len(train_eval_ds) // args.train_eval_samples))))

    def loader(ds, shuffle=False, subset=None):
        src = Subset(ds, subset) if subset is not None else ds
        return DataLoader(src, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers, collate_fn=collate, pin_memory=True,
                          drop_last=shuffle, persistent_workers=args.workers > 0)

    train_dl = loader(train_ds, shuffle=True)
    val_dl, blind_dl = loader(val_ds, subset=idxs), loader(blind_ds, subset=idxs)
    train_eval_dl = loader(train_eval_ds, subset=tr_idxs)

    model = Talk2EventGrounding(
        event_ckpt=args.event_ckpt or None, event_backbone=args.event_backbone,
        text_model=args.text_model, img_size=tuple(args.size), T=args.T,
        depth=args.depth, n_slots=args.n_slots, ilif=not args.no_ilif,
        condition_encoder=not args.no_condition, freeze_event=args.freeze_event,
        freeze_text=not args.train_text, text_unfreeze_last=args.text_unfreeze_last,
    ).to(device)
    crit = SingleBoxLoss(center_weight=args.center_weight).to(device)

    pretrained = list(model.events.backbone.parameters()) + \
        list(model.text.encoder.parameters())
    enc_ids = {id(p) for p in pretrained}
    enc = [p for p in model.parameters() if p.requires_grad and id(p) in enc_ids]
    new = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    groups = [g for g in ({"params": enc, "lr": args.encoder_lr},
                          {"params": new, "lr": args.lr}) if g["params"]]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]

    steps_per_epoch = max(1, len(train_dl) // args.accum)
    total_steps = steps_per_epoch * args.epochs

    def set_lr(step: int) -> float:
        if step < args.warmup:
            m = (step + 1) / max(1, args.warmup)
        else:
            p = (step - args.warmup) / max(1, total_steps - args.warmup)
            m = 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * m
        return m

    start_epoch, best, since_best, gstep = 0, -1.0, 0, 0
    if args.resume:
        blob = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"])
        start_epoch, best = blob["epoch"] + 1, blob.get("best", -1.0)
        gstep = blob.get("gstep", start_epoch * steps_per_epoch)
        print(f"resumed {args.resume} at epoch {start_epoch} (best mIoU {best:.4f})")

    tp = model.trainable_parameters()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"device {device} | bf16 train AND eval | batch {args.batch_size} x accum {args.accum}")
    print(f"talk2event: train {len(train_ds)} | {args.val_split} {len(val_ds)} "
          f"(eval on {len(idxs)}, {sum(keep)} with a same-frame hard negative)")
    print(f"events {args.event_backbone} | T {args.T} x 2ch | {args.size[0]}x{args.size[1]} "
          f"| taps {model.taps} | slots {args.n_slots}")
    print(f"conditioning {'OFF' if args.no_condition else 'ON'} "
          f"({model.events.modulated_stages}) | ilif {not args.no_ilif} | "
          f"event backbone {'FROZEN' if args.freeze_event else 'trainable'}")
    print(f"loss = box + {args.slot_weight}*slotCE + {args.tag_weight}*tagCE")
    print(f"trainable {n_tr/1e6:.2f}M of {n_all/1e6:.1f}M  " +
          "  ".join(f"{k} {v/1e6:.2f}M" for k, v in tp.items()))
    print(f"{steps_per_epoch} steps/epoch, {total_steps} total\n", flush=True)

    if not (out / "log.tsv").exists():
        (out / "log.tsv").write_text(
            "epoch\tstep\tloss\tbox\tslot\ttag\ttrain_iou\ttrain_mIoU\tlr\tmIoU\tAcc@0.25\t"
            "Acc@0.5\tAcc@0.75\tAcc@0.9\tblind_mIoU\tdelta\tsec\n")

    t0, stop = time.time(), False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run, seen = defaultdict(float), 0
        opt.zero_grad(set_to_none=True)
        for it, (cube, ids, mask, gt, lab, _c, _r) in enumerate(train_dl):
            cube, ids, mask, gt, lab = (t.to(device, non_blocking=True)
                                        for t in (cube, ids, mask, gt, lab))
            with amp_ctx():
                o = model(cube, ids, mask)
            box_loss, parts = crit(o["box"].float(), gt)
            slot_ce = F.cross_entropy(
                o["slot_logits"].float().flatten(0, 1),
                slot_targets(gt, args.n_slots).flatten(0, 1))
            # ignore_index on the "no attribute" class would train only the positives;
            # it is kept as a real class so the tagger learns what is NOT an attribute
            tag_ce = F.cross_entropy(o["tag_logits"].float().transpose(1, 2), lab)
            loss = box_loss + args.slot_weight * slot_ce + args.tag_weight * tag_ce
            (loss / args.accum).backward()

            if (it + 1) % args.accum == 0:
                mult = set_lr(gstep)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.clip)
                opt.step(); opt.zero_grad(set_to_none=True)
                gstep += 1

            b = gt.shape[0]; seen += b
            run["loss"] += loss.item() * b; run["box"] += box_loss.item() * b
            run["slot"] += slot_ce.item() * b; run["tag"] += tag_ce.item() * b
            run["iou"] += float(parts["iou"]) * b

            if args.log_every and (it + 1) % (args.log_every * args.accum) == 0:
                print(f"  ep {epoch} step {gstep}/{total_steps}  loss {run['loss']/seen:6.3f} "
                      f"(box {run['box']/seen:.3f} slot {run['slot']/seen:.3f} "
                      f"tag {run['tag']/seen:.3f})  train-IoU {run['iou']/seen:.3f}  "
                      f"lr x{mult:.3f}  {(time.time()-t0)/60:.1f} min", flush=True)
            if args.max_iters and it + 1 >= args.max_iters:
                break

        m = evaluate_t2e(model, val_dl, device, amp_ctx)
        train_m = evaluate_t2e(model, train_eval_dl, device, amp_ctx)
        gap = train_m["mIoU"] - run["iou"] / seen
        blind = (evaluate_t2e(model, blind_dl, device, amp_ctx, keep=keep)
                 if args.blind_every and epoch % args.blind_every == 0 else None)
        m_hard = (evaluate_t2e(model, val_dl, device, amp_ctx, keep=keep)
                  if blind is not None else None)
        delta = (m_hard["mIoU"] - blind["mIoU"]) if blind else float("nan")

        print(f"EPOCH {epoch:3d} | loss {run['loss']/seen:6.3f} (box {run['box']/seen:.3f} "
              f"slot {run['slot']/seen:.3f} tag {run['tag']/seen:.3f}) "
              f"train-IoU {run['iou']/seen:.3f} (mode gap {gap:+.4f}) | "
              f"mIoU {m['mIoU']:.4f}  Acc@0.25 {m['Acc@0.25']:.4f}  Acc@0.5 {m['Acc@0.5']:.4f}  "
              f"Acc@0.75 {m['Acc@0.75']:.4f}  Acc@0.9 {m['Acc@0.9']:.4f}"
              + (f" | blind {blind['mIoU']:.4f} delta {delta:+.4f}" if blind else "")
              + f" | {(time.time()-t0)/60:.1f} min", flush=True)

        with open(out / "log.tsv", "a") as f:
            f.write(f"{epoch}\t{gstep}\t{run['loss']/seen:.4f}\t{run['box']/seen:.4f}\t"
                    f"{run['slot']/seen:.4f}\t{run['tag']/seen:.4f}\t{run['iou']/seen:.4f}\t"
                    f"{train_m['mIoU']:.4f}\t{mult:.4f}\t{m['mIoU']:.4f}\t"
                    f"{m['Acc@0.25']:.4f}\t{m['Acc@0.5']:.4f}\t{m['Acc@0.75']:.4f}\t"
                    f"{m['Acc@0.9']:.4f}\t{blind['mIoU'] if blind else float('nan'):.4f}\t"
                    f"{delta:.4f}\t{int(time.time()-t0)}\n")

        blob = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                "gstep": gstep, "args": vars(args), "metrics": m}
        torch.save(blob, out / "last.pth")
        if m["mIoU"] > best:
            best, since_best = m["mIoU"], 0
            blob["best"] = best
            torch.save(blob, out / "best.pth")
            print(f"         new best mIoU {best:.4f}", flush=True)
            if args.keep_all_best:
                snap = out / f"best_ep{epoch:03d}_miou{m['mIoU']:.4f}.pth"
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "args": vars(args), "metrics": m, "best": best}, snap)
                print(f"         archived {snap.name}", flush=True)
        else:
            since_best += 1
            if args.patience and since_best >= args.patience:
                print(f"\nEARLY STOP: {since_best} epochs without improvement.", flush=True)
                stop = True
        if stop or args.max_iters:
            break

    ck = out / "best.pth"
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=device,
                                         weights_only=False)["model"])
    print(f"\nfinal evaluation on all {len(val_ds)} {args.val_split} samples "
          f"(best checkpoint)", flush=True)
    full, full_blind = loader(val_ds), loader(blind_ds)
    m = evaluate_t2e(model, full, device, amp_ctx)
    b = evaluate_t2e(model, full_blind, device, amp_ctx, keep=is_hard)
    m_hard = evaluate_t2e(model, full, device, amp_ctx, keep=is_hard)
    m["blind_mIoU"] = b["mIoU"]
    m["mIoU_on_hard_subset"] = m_hard["mIoU"]
    m["caption_delta"] = m_hard["mIoU"] - b["mIoU"]
    m["n_hard"] = int(sum(is_hard))
    m["_delta_note"] = ("negatives are a caption for a DIFFERENT object in the SAME event "
                        "frame; delta compares only the n_hard samples that have one")
    print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in m.items() if not k.startswith("_")), flush=True)
    (out / "final_metrics.json").write_text(json.dumps(m, indent=1))
    print(f"done in {(time.time()-t0)/60:.1f} min | best {args.val_split} mIoU {best:.4f}")


if __name__ == "__main__":
    main()
