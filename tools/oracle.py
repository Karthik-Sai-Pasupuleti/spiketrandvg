"""Where does the error live: the centre, the size, or the map?

    uv run python tools/oracle.py --run runs/probe_01 --limit 400

Five numbers on the same predictions:

  model              as it stands
  pred cx,cy + TRUE w,h     an oracle on size
  TRUE cx,cy + pred w,h     an oracle on the centre
  MAP centre + pred w,h     the centre read off the attention map's own marginals --
                            NO oracle, this is exactly what `--attn-prior` would feed
                            the head, so it is a direct prediction of whether the prior
                            can pay off, at 3 minutes instead of a 24-minute probe
  MAP argmax + pred w,h     the same, hard argmax instead of the expectation

Runs on `val`. It never opens `test`.
"""

from __future__ import annotations

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou

from spiketrandvg.dataloader import Talk2Event, make_t2e_collate
from spiketrandvg.model import cxcywh_to_xyxy_norm
from spiketrandvg.textencoder import build_tokenizer
from tools.protected import build


def report(name, pred, gt):
    i = box_iou(cxcywh_to_xyxy_norm(pred), cxcywh_to_xyxy_norm(gt)).diagonal()
    print(f"  {name:28s} mIoU {i.mean():.4f}  Acc@0.5 {(i >= 0.5).float().mean():.4f}  "
          f"Acc@0.75 {(i >= 0.75).float().mean():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pth")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("diagnostic; it does not open the test split")

    device = "cuda"
    model, ra, blob = build(Path(args.run), args.ckpt, device)
    model.return_map = True                      # ask forward() for the attention map
    ds = Talk2Event(args.split, t_steps=ra.get("T", 5))
    idxs = (list(range(0, len(ds), max(1, len(ds) // args.limit))) if args.limit
            else list(range(len(ds))))
    dl = DataLoader(Subset(ds, idxs), batch_size=args.batch_size, shuffle=False,
                    num_workers=6, collate_fn=make_t2e_collate(build_tokenizer()),
                    pin_memory=True)

    P, G, MC, MA = [], [], [], []
    V = model.box_head.n_slots
    centres = (torch.arange(V, dtype=torch.float32, device=device) + 0.5) / V
    with torch.no_grad():
        for cube, ids, mask, gt, _l, _c, _r in dl:
            cube, ids, mask = (t.to(device, non_blocking=True) for t in (cube, ids, mask))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(cube, ids, mask)
            P.append(o["box"].float().cpu()); G.append(gt)
            # the map's own opinion, via the same marginals --attn-prior would build
            lp = model._attn_position_prior(o["attn"]) if model.attn_prior else None
            if lp is None:                       # gain is 0 or absent: build it directly
                g = getattr(model, "attn_prior_gain", None)
                model.attn_prior_gain = torch.ones(1, device=device)
                lp = model._attn_position_prior(o["attn"])
                if g is None:
                    del model.attn_prior_gain
                else:
                    model.attn_prior_gain = g
            p = lp.float().softmax(-1)           # (B,2,V)
            MC.append((p * centres).sum(-1).cpu())
            MA.append(centres[p.argmax(-1)].cpu())

    P, G = torch.cat(P), torch.cat(G)
    MC, MA = torch.cat(MC), torch.cat(MA)
    print(f"\n{args.run}/{args.ckpt} on {args.split}: {len(P)} samples\n")
    report("model", P, G)
    report("pred centre + TRUE size", torch.cat([P[:, :2], G[:, 2:]], -1), G)
    report("TRUE centre + pred size", torch.cat([G[:, :2], P[:, 2:]], -1), G)
    report("MAP centre + pred size", torch.cat([MC, P[:, 2:]], -1), G)
    report("MAP argmax + pred size", torch.cat([MA, P[:, 2:]], -1), G)
    report("MAP centre + TRUE size", torch.cat([MC, G[:, 2:]], -1), G)
    d = (P[:, :2] - G[:, :2]).abs().mul(torch.tensor([640., 480.])).norm(dim=-1)
    dm = (MC - G[:, :2]).abs().mul(torch.tensor([640., 480.])).norm(dim=-1)
    print(f"\n  centre error px: model median {d.median():.1f}, map median {dm.median():.1f}"
          f"   (IoU 0.75 needs ~6.7 px)")


if __name__ == "__main__":
    main()
