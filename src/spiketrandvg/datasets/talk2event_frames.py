"""Talk2Event as a plain object-detection set: one sample per event frame, all objects.

`Talk2EventDataset` yields one sample per (object, caption) pair, which is the right unit
for grounding but the wrong one for training a detector. This module re-keys the same
annotations by `event_path` and pools every object visible in each frame -- each item's
own `bbox` plus every entry in its `others` list -- so a frame appears once with the
complete set of boxes.

Measured on `meta_data_v10`:

    train   4433 frames, 10321 boxes (2.33 per frame)
    test    1134 frames,  3127 boxes (2.76 per frame)

The point of this set is pretraining. Localisation is learnable from events alone, with no
language involved; a detector trained here gives the grounding model a backbone that
already knows where objects are, leaving cross-attention with only the selection problem.
"""

from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from spiketrandvg.datasets.event_augment import AugmentConfig, augment_sample
from spiketrandvg.datasets.events_voxel_cube import T_STEPS, talk2event_cube

__all__ = ["CLASSES", "Talk2EventFrames", "frame_index", "detection_collate"]

CLASSES = ("pedestrian", "rider", "car", "bus", "truck", "bicycle", "motorcycle", "train")

_WS = Path(os.environ.get("T2E_WS", Path(__file__).resolve().parents[4]))
DATA_ROOT = Path(os.environ.get("T2E_DATA_ROOT", _WS / "dataset" / "talk2event"))


def frame_index(split: str, root: Path | None = None) -> list[tuple[str, list[tuple]]]:
    """split -> sorted [(event_path, [(x0, y0, x1, y1, class_idx), ...])].

    Boxes go through a `set` before sorting: the same object is described by several
    captions and also appears in every sibling item's `others`, so the raw lists contain
    heavy duplication.
    """
    root = root or DATA_ROOT
    frames: dict[str, set] = {}
    pattern = str(root / f"meta_data_v10/{split}/*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no annotation files at {pattern}")
    for f in files:
        for item in json.load(open(f)):
            boxes = frames.setdefault(item["event_path"], set())
            pairs = [(item["class"], item["bbox"])]
            pairs += [(o["class"], o["bbox"]) for o in item.get("others", [])]
            for cls, b in pairs:
                boxes.add((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"], CLASSES.index(cls)))
    return [(p, sorted(v)) for p, v in sorted(frames.items())]


class Talk2EventFrames(Dataset):
    """(T, 2, H, W) event cube -> every annotated box in that frame.

    Returns `(cube, boxes, labels, path)` with boxes as (N, 4) xyxy in PIXELS. Pixel units
    rather than normalised ones because the YOLO head's DFL works in feature-grid cells,
    which are defined by the stride in pixels.
    """

    def __init__(self, split: str, root: Path | None = None, limit: int | None = None,
                 augment: AugmentConfig | None = None):
        self.root = root or DATA_ROOT
        self.index = frame_index(split, self.root)[:limit]
        self.split = split
        # None = no augmentation, which is mandatory for eval and was the state of the
        # first two training runs (see event_augment for why that mattered)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.index)

    @property
    def num_boxes(self) -> int:
        return sum(len(b) for _, b in self.index)

    def _load(self, i):
        path, boxes = self.index[i]
        raw = np.load(self.root / path)["events"].astype(np.float32)
        cube = talk2event_cube(torch.from_numpy(raw))          # (T, 2, H, W)
        arr = torch.tensor(boxes, dtype=torch.float32)         # (N, 5)
        return cube, arr[:, :4].contiguous(), arr[:, 4].long(), path

    def __getitem__(self, i):
        cube, boxes, labels, path = self._load(i)
        if self.augment is not None:
            extra = None
            if random.random() < self.augment.mosaic:
                # three more frames, drawn from anywhere in the split -- that is the point,
                # it composes scenes the 47 drives never contain. Loaded only when mosaic
                # actually fires, since each one is another .npz read.
                extra = [self._load(random.randrange(len(self.index)))[:3] for _ in range(3)]
            cube, boxes, labels = augment_sample(cube, boxes, labels, self.augment, extra)
        return cube, boxes, labels, path


def detection_collate(batch):
    """Pad the ragged box lists to the batch maximum, the layout TaskAlignedAssigner wants.

    -> cube (T, B, 2, H, W)
       boxes (B, n_max, 4) xyxy px, zero-padded
       labels (B, n_max) long, zero-padded
       mask (B, n_max, 1) float, 1 where a box is real
       paths list[str]
    """
    cubes, boxes, labels, paths = zip(*batch)
    b = len(batch)
    n_max = max(len(x) for x in boxes)
    pad_b = torch.zeros(b, n_max, 4)
    pad_l = torch.zeros(b, n_max, dtype=torch.long)
    mask = torch.zeros(b, n_max, 1)
    for i, (bx, lb) in enumerate(zip(boxes, labels)):
        n = len(bx)
        pad_b[i, :n], pad_l[i, :n], mask[i, :n] = bx, lb, 1.0
    return torch.stack(cubes, dim=1), pad_b, pad_l, mask, list(paths)
