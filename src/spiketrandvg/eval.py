"""Evaluate a trained RefCOCOGrounding checkpoint on any RefCOCO split.

    uv run python -m spiketrandvg.eval --run runs/refcoco_b1 --split testA

Reports mIoU, Acc@{0.25,0.5,0.75,0.9}, and the caption-blind control. Writes
`eval_<split>.json` into the run directory.

Why this exists separately from `train.py`
------------------------------------------
`train.py` evaluates on `--val-split` and selects `best.pth` by that same score. Reporting
the headline on the split you selected on is model selection on the test set -- the
research log flags this trap for the detector, and it is easy to reproduce by accident.
This script exists so the test splits can be touched exactly once, after selection is
finished, with no path back into training.

The caption-blind control
-------------------------
`blind_mIoU` re-runs the same images with the WRONG caption; `caption_delta` is the
difference. A model with delta ~0 is a detector with a text input it ignores -- which this
project measured for 85 epochs before catching it.

The negative must be a caption for a DIFFERENT OBJECT IN THE SAME IMAGE, and it is
precomputed per sample rather than shuffled within the batch. Rotating captions inside a
batch is WRONG on RefCOCO, measured: 56.7% of the pairs it forms share the same object,
because RefCOCO gives each object ~2.8 expressions and stores them consecutively. A
paraphrase scored as a negative collapses the delta toward zero and hides real grounding
-- the identical bug on Talk2Event turned a true +0.22 into a reported +0.017. The
implementation is shared with `train.py` (`hard_negative_captions`) so the two can never
drift apart.

bf16
----
Evaluation runs under the same autocast the model was trained in. These spiking layers are
not dtype-agnostic: precision decides which membranes cross threshold, and fp32 evaluation
of a bf16-trained model measured 0.060 IoU against 0.656 on identical weights.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from spiketrandvg.dataset import RefCOCO, SPLITS, make_collate
from spiketrandvg.model import RefCOCOGrounding
from spiketrandvg.text_encoder import build_tokenizer
from spiketrandvg.train import (IOU_THRESHOLDS, _CaptionOverride, evaluate,
                                hard_negative_captions)


def load_run(run_dir: Path, ckpt: str, device: str):
    """Rebuild a run's architecture from its args.json and load the checkpoint.

    `rgb_ckpt=None`: the checkpoint's state_dict overwrites every backbone weight
    immediately, so fetching SpiLiFormer's ImageNet file first would be wasted work.

    `strict=False` with an explicit report: a checkpoint predating a code change (e.g.
    `head_norm0`) is missing those keys, and they would otherwise sit at random init and
    quietly corrupt the numbers. Loud is better than subtly wrong.
    """
    run_args = json.loads((run_dir / "args.json").read_text())
    blob = torch.load(run_dir / ckpt, map_location=device, weights_only=False)
    model = RefCOCOGrounding(
        rgb_ckpt=None, text_model=run_args.get("text_model", "roberta-base"),
        img_size=tuple(run_args["size"]), T=run_args.get("T", 4),
        rgb_T=run_args.get("rgb_T", 1), depth=run_args.get("depth", 2),
        attn_type=run_args.get("attn_type", "spatial_softmax"),
        freeze_rgb=False, freeze_text=not run_args.get("train_text", False),
        head_type=run_args.get("head_type", "pooled_mlp"),
        attn_map=run_args.get("attn_map", "last"),
        pos_std=run_args.get("pos_std", 0.02),
        text_unfreeze_last=(0 if run_args.get("train_text") else
                            run_args.get("text_unfreeze_last", 0)),
    ).to(device)
    report = model.load_state_dict(blob["model"], strict=False)
    print(f"loaded {run_dir / ckpt} (epoch {blob.get('epoch')}, "
          f"head_type={run_args.get('head_type', 'pooled_mlp')})")
    if report.missing_keys or report.unexpected_keys:
        print(f"  WARNING: missing {report.missing_keys}, "
              f"unexpected {report.unexpected_keys} -- checkpoint predates a code "
              f"change; those parameters are at RANDOM INIT and these numbers are "
              f"not comparable to a matching run")
    return model, run_args, blob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory, e.g. runs/refcoco_b1")
    ap.add_argument("--ckpt", default="best.pth", help="filename inside --run")
    ap.add_argument("--split", default="val",
                    help="val / testA / testB (refcocog: val / test)")
    ap.add_argument("--dataset", default=None,
                    help="override the run's dataset (refcoco / refcoco+ / refcocog)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="strided subset for a quick check; 0 = the whole split")
    ap.add_argument("--no-blind", action="store_true",
                    help="skip the caption-blind control (it doubles runtime)")
    ap.add_argument("--out", default=None, help="output json (default: <run>/eval_<split>.json)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run)
    model, run_args, _ = load_run(run_dir, args.ckpt, device)
    model.eval()

    name = args.dataset or run_args["dataset"]
    if args.split not in SPLITS[name]:
        raise SystemExit(f"{name} has no split {args.split!r}; it has {SPLITS[name]}")
    if args.split == run_args.get("val_split") and args.ckpt == "best.pth":
        print(f"  NOTE: '{args.split}' is the split best.pth was SELECTED on -- this is "
              f"not a held-out number. Use testA/testB to report.")

    amp_ctx = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
               if device == "cuda" else contextlib.nullcontext)
    collate = make_collate(build_tokenizer())
    ds = RefCOCO(name, args.split, size=tuple(run_args["size"]), augment=None)

    idxs = (list(range(len(ds))) if not args.limit else
            list(range(0, len(ds), max(1, len(ds) // args.limit))))

    def loader(source):
        src = Subset(source, idxs) if args.limit else source
        return DataLoader(src, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, collate_fn=collate, pin_memory=True)

    print(f"\n{name}/{args.split}: {len(idxs)} of {len(ds)} samples")
    m = evaluate(model, loader(ds), device, amp_ctx)

    if not args.no_blind:
        neg_caps, is_hard = hard_negative_captions(ds)
        keep = [is_hard[i] for i in idxs]
        blind = evaluate(model, loader(_CaptionOverride(ds, neg_caps)), device, amp_ctx,
                         keep=keep)
        # delta compares the correct-caption score on the SAME subset the blind score
        # used -- only samples that have a real same-image negative -- so the two means
        # are over the same population
        m_hard = evaluate(model, loader(ds), device, amp_ctx, keep=keep)
        m["blind_mIoU"] = blind["mIoU"]
        m["mIoU_on_hard_subset"] = m_hard["mIoU"]
        m["caption_delta"] = m_hard["mIoU"] - blind["mIoU"]
        m["n_hard"] = int(sum(keep))
        m["_delta_note"] = ("negatives are a caption for a DIFFERENT object in the SAME "
                            "image; delta compares only the n_hard samples that have one")

    m["_split"] = f"{name}/{args.split}"
    m["_ckpt"] = str(run_dir / args.ckpt)
    print("\n=== results ===")
    for k, v in m.items():
        if k.startswith("_"):
            continue
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")

    out = Path(args.out) if args.out else run_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(m, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
