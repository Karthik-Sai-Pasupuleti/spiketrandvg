"""The protected experiments: recorded, never hill-climbed.

    uv run python tools/protected.py --run runs/probe_00 --what all

Three tables the benchmark paper does not publish, plus the health check that has to
run after any encoder change:

  attributes   Acc@0.5 / Acc@0.75 with the caption reduced to ONE of Talk2Event's four
               annotated attribute groups. This is the table the four-sub-query design
               exists to support: if `relation_viewer` alone localises better than
               `appearance` alone, the model is reading geometry out of the caption, and
               if they are equal it is reading none of them in particular.
  sizes        Acc@0.75 bucketed by object size, against the arithmetic ceiling. IoU
               0.75 on a box of side s allows a centre error of about 0.143*s per axis,
               so a 40 px object permits 5.7 px and a 150 px object permits 21 px.
               Whether the failure is uniform across sizes decides whether resolution
               is implicated at all.
  firing       Mean firing rate of every spiking layer. Below 1% a LIF is dead and its
               residual branch is silently the identity; healthy is 5-35%.

Runs on `val`. It never opens `test`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.ops import box_iou

from spiketrandvg.dataloader import Talk2Event, make_t2e_collate
from spiketrandvg.model import Talk2EventGrounding, cxcywh_to_xyxy_norm
from spiketrandvg.textencoder import ATTRIBUTES, build_tokenizer


class _Override(Dataset):
    """Same cubes and boxes, each paired with a supplied caption."""

    def __init__(self, ds, captions):
        self.ds, self.captions = ds, captions

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        cube, _cap, box, _spans, rec = self.ds[i]
        return cube, self.captions[i], box, [], rec


def build(run_dir: Path, ckpt: str, device: str):
    ra = json.loads((run_dir / "args.json").read_text())
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
        qk_lif=ra.get("qk_lif", "binary"), attn_prior=ra.get("attn_prior", False),
        attn_prior_gain=ra.get("attn_prior_gain", 0.0),
        pos_ratio=ra.get("pos_ratio"),
    ).to(device)
    blob = torch.load(run_dir / ckpt, map_location=device, weights_only=False)
    rep = model.load_state_dict(blob["model"], strict=False)
    if rep.missing_keys or rep.unexpected_keys:
        print(f"  WARNING missing {rep.missing_keys} unexpected {rep.unexpected_keys}")
    model.eval()
    return model, ra, blob


@torch.no_grad()
def ious_for(model, ds, idxs, bs, device):
    dl = DataLoader(Subset(ds, idxs), batch_size=bs, shuffle=False, num_workers=6,
                    collate_fn=make_t2e_collate(build_tokenizer()), pin_memory=True)
    out = []
    for cube, ids, mask, gt, _l, _c, _r in dl:
        cube, ids, mask, gt = (t.to(device, non_blocking=True)
                               for t in (cube, ids, mask, gt))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(cube, ids, mask)
        out.append(box_iou(cxcywh_to_xyxy_norm(o["box"].float()),
                           cxcywh_to_xyxy_norm(gt)).diagonal().cpu())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pth")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--what", default="all",
                    choices=["all", "attributes", "sizes", "firing"])
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("protected experiments run on val; test is opened once, by hand")

    device, run = "cuda", Path(args.run)
    model, ra, blob = build(run, args.ckpt, device)
    ds = Talk2Event(args.split, t_steps=ra.get("T", 5))
    idxs = (list(range(0, len(ds), max(1, len(ds) // args.limit))) if args.limit
            else list(range(len(ds))))
    print(f"\n{run}/{args.ckpt} (epoch {blob.get('epoch')}) on {args.split}: "
          f"{len(idxs)} of {len(ds)} samples")

    base = None
    if args.what in ("all", "attributes", "sizes"):
        base = ious_for(model, ds, idxs, args.batch_size, device)
        print(f"\nfull caption            n {len(base):5d}  mIoU {base.mean():.4f}  "
              f"Acc@0.5 {(base >= 0.5).float().mean():.4f}  "
              f"Acc@0.75 {(base >= 0.75).float().mean():.4f}")

    if args.what in ("all", "attributes"):
        print("\n=== per-attribute: the caption reduced to ONE attribute group ===")
        print(f"{'attribute':18s} {'n':>5s} {'covered':>8s} {'mIoU':>7s} "
              f"{'Acc@0.5':>8s} {'Acc@0.75':>9s}")
        for a in ATTRIBUTES:
            caps, keep = [], []
            for i in range(len(ds)):
                _c, attrs = ds._caption_and_attrs(ds.items[i], i)
                phrases = [p for p in (attrs.get(a) or []) if p and p.strip()]
                caps.append(", ".join(phrases) if phrases else "")
                keep.append(bool(phrases))
            sub = [i for i in idxs if keep[i]]
            if not sub:
                print(f"{a:18s} {'0':>5s}  no annotated phrases"); continue
            t = ious_for(model, _Override(ds, caps), sub, args.batch_size, device)
            print(f"{a:18s} {len(t):5d} {len(sub)/len(idxs):8.1%} {t.mean():7.4f} "
                  f"{(t >= 0.5).float().mean():8.4f} {(t >= 0.75).float().mean():9.4f}")

    if args.what in ("all", "sizes"):
        print("\n=== Acc@0.75 by object size (centre tolerance ~0.143 * min side) ===")
        side = torch.tensor([min(ds.items[i]["bbox"]["w"], ds.items[i]["bbox"]["h"])
                             for i in idxs], dtype=torch.float32)
        edges = [0, 25, 40, 60, 90, 140, 10**9]
        print(f"{'min side px':>14s} {'n':>5s} {'tol px':>7s} {'mIoU':>7s} "
              f"{'Acc@0.5':>8s} {'Acc@0.75':>9s}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (side >= lo) & (side < hi)
            if m.sum() == 0:
                continue
            t = base[m]
            lab = f"{lo}-{hi if hi < 10**8 else ''}"
            print(f"{lab:>14s} {int(m.sum()):5d} {0.143*side[m].median():7.1f} "
                  f"{t.mean():7.4f} {(t >= 0.5).float().mean():8.4f} "
                  f"{(t >= 0.75).float().mean():9.4f}")

    if args.what in ("all", "firing"):
        print("\n=== firing rate per spiking layer (<1% dead, healthy 5-35%) ===")
        from spikingjelly.clock_driven import neuron as sj_neuron
        rates: dict[str, list[float]] = defaultdict(list)
        hooks = []

        levels: dict[str, list[float]] = defaultdict(list)

        def mk(name):
            def h(_m, _i, o):
                if isinstance(o, torch.Tensor):
                    rates[name].append((o.detach() > 0).float().mean().item())
                    # I-LIF emits {0,1,2,3,4}, so "fraction nonzero" is not the health
                    # measure a binary LIF's is -- a layer at 70% nonzero but mean level
                    # 0.8 is firing sparsely in amplitude. Report both.
                    levels[name].append(o.detach().float().mean().item())
            return h

        for name, mod in model.named_modules():
            cls = type(mod).__name__
            if isinstance(mod, sj_neuron.BaseNode) or cls == "mem_update":
                hooks.append(mod.register_forward_hook(mk(f"{name} [{cls}]")))
        ious_for(model, ds, idxs[:40], args.batch_size, device)
        for h in hooks:
            h.remove()
        dead = 0
        print(f"  {'nonzero':>8s} {'mean lvl':>9s}  layer")
        for k in sorted(rates, key=lambda k: sum(rates[k]) / len(rates[k])):
            r = sum(rates[k]) / len(rates[k])
            lv = sum(levels[k]) / len(levels[k])
            # a binary LIF's max level is 1.0, an I-LIF's is 4.0; flag on the fraction of
            # the available range actually used, not on the nonzero count
            hi = 4.0 if "mem_update" in k else 1.0
            flag = ("  DEAD" if r < 0.01 else
                    "  SATURATED" if lv > 0.6 * hi else "")
            dead += r < 0.01
            print(f"  {r:8.4f} {lv:9.4f}  {k}{flag}")
        print(f"  {len(rates)} spiking layers, {dead} below 1%")


if __name__ == "__main__":
    main()
