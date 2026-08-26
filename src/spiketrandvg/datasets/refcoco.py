"""RefCOCO / RefCOCO+ / RefCOCOg as a single-box grounding set.

One sample is one referring expression: an RGB image, a caption, and the ONE box the
caption refers to. That is the same unit `Talk2EventDataset` yields and the same unit
`SingleBoxLoss` consumes, so a model trained on Talk2Event transfers here with no change
to the loss or the head -- only the event stream is absent.

Why this dataset is here
------------------------
Talk2Event's grounding split is small and structurally narrow: 47 train sequences with
zero overlap against the 13 test sequences, frames spaced 8 apart, so roughly 47 distinct
scenes back 23k samples. RefCOCO is 120k expressions over 20k COCO images with no such
correlation. It is the standard referring-expression benchmark, which makes it both a
pretraining source for the RGB branch and a sanity check: a grounding architecture that
cannot learn RefCOCO is not being held back by the event modality.

Format
------
MDETR's pre-processed annotations (`finetune_<name>_<split>.json`, Zenodo record 4729015),
not the original UNC pickles. Each entry in `images` IS one referring expression -- it
carries the caption -- and `annotations` holds exactly one box per entry, verified:

    refcoco   120624 train / 10834 val / 5657 testA / 5095 testB
    refcoco+  120191 train / 10758 val / 5726 testA / 4889 testB
    refcocog   80512 train /  4896 val / 9602 test          (umd split)

Boxes are COCO `xywh` in absolute pixels and were checked to lie inside the image on every
one of the 321k annotations, so no clamping is needed on load.

Two measured facts drive the design
-----------------------------------
**Captions are short.** Median 5 tokens on refcoco, 10 on refcocog; the longest in any
split is 49. Padding to the project's MAX_TEXT_LEN of 80 would put >85% padding through
cross-attention, whose cost is linear in query count. `make_collate` pads to the longest
caption in the batch instead, which is exact and much cheaper.

**45.7% of refcoco captions contain "left" or "right"** (11.4% of refcocog, 0.1% of
refcoco+, whose whole premise is banning location words). A horizontal flip moves the box
but not the words, so on refcoco a naive flip mislabels almost half the training set --
the caption says "left" and the target is now on the right. `hflip_caption` rewrites the
words; `RefCOCOAugment.hflip` is safe only because of it. This is the single easiest way
to silently poison a RefCOCO run.

Images are resized to a fixed square because COCO sizes vary (279 distinct (h, w) pairs in
the val split alone) and the batch must be one tensor. Boxes are normalised by the
sample's ORIGINAL width and height before the resize, so they are resolution-independent
and the resize needs no corresponding box transform -- the same convention
`Talk2EventDataset` uses.

Using it with SpikeGroundingV2
------------------------------
The model takes `(cube, text_tokens, attention_mask, rgb)` and requires the event cube.
RefCOCO has no events, so either build the model RGB-only, or pass a zero cube of shape
(T, B, 2, H, W) and accept that the event branch contributes a constant. `make_collate`
returns no cube; the caller decides.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms.functional as tv_F
from PIL import Image
from torch.utils.data import Dataset

__all__ = [
    "DATA_ROOT",
    "SPLITS",
    "RefCOCO",
    "RefCOCOAugment",
    "hflip_caption",
    "make_collate",
]

# Same constants the Talk2Event pipeline normalises with, so a backbone pretrained on one
# set sees the same input distribution on the other. Defined here rather than imported to
# avoid pulling in talk2event_dataset's heavy module-level work.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_WS = Path(os.environ.get("T2E_WS", Path(__file__).resolve().parents[4]))
DATA_ROOT = Path(os.environ.get("REFCOCO_ROOT", _WS / "dataset" / "refcoco"))

# (dataset, split) -> annotation file stem. refcocog has no testA/testB -- it is the umd
# split, which is train/val/test -- so asking for one is an error rather than a silent
# fallback to another split.
SPLITS: dict[str, tuple[str, ...]] = {
    "refcoco": ("train", "val", "testA", "testB"),
    "refcoco+": ("train", "val", "testA", "testB"),
    "refcocog": ("train", "val", "test"),
}

# Word-level left/right swap. Ordered longest-first is irrelevant here because the regex
# alternation is anchored on word boundaries, but the pairs must cover the compounds:
# "leftmost" and "left-hand" appear in RefCOCO and a bare left->right rule would leave
# "leftmost" untouched while flipping the image under it.
_LR_PAIRS = [
    ("left", "right"),
    ("lefts", "rights"),
    ("leftmost", "rightmost"),
    ("lefthand", "righthand"),
    ("left-hand", "right-hand"),
    ("leftside", "rightside"),
    ("left-side", "right-side"),
    ("lefty", "righty"),
    ("leftward", "rightward"),
]
_LR_MAP = {a: b for a, b in _LR_PAIRS} | {b: a for a, b in _LR_PAIRS}
# \b does not fire around the hyphen in "left-hand", so match the hyphenated forms first.
_LR_RE = re.compile(
    r"(?<![\w-])(" + "|".join(sorted(map(re.escape, _LR_MAP), key=len, reverse=True)) + r")(?![\w-])",
    re.IGNORECASE,
)


def hflip_caption(caption: str) -> str:
    """Swap every left/right word, preserving the original capitalisation pattern.

    Mandatory whenever the image is mirrored: 45.7% of refcoco captions name a side, and
    flipping pixels without flipping words turns those into wrong labels.

    >>> hflip_caption("Left guy in the leftmost chair")
    'Right guy in the rightmost chair'
    """

    def sub(m: re.Match) -> str:
        word = m.group(0)
        rep = _LR_MAP[word.lower()]
        if word.isupper():
            return rep.upper()
        if word[0].isupper():
            return rep.capitalize()
        return rep

    return _LR_RE.sub(sub, caption)


@dataclass
class RefCOCOAugment:
    """Train-time augmentation. Every field is a probability except `jitter`.

    Deliberately conservative. Random crops are the obvious next addition and are
    deliberately absent: a crop can remove the referent entirely, which produces a sample
    whose caption describes nothing in the image, and detecting that needs a box-validity
    check plus a resample loop. Flip and photometric jitter cannot invalidate a sample.
    """

    hflip: float = 0.5
    # brightness/contrast/saturation jitter strength, 0 disables. Hue is left alone --
    # refcoco+ leans on colour words ("the red one"), and rotating hue makes them wrong
    # in exactly the way a flipped "left" is wrong.
    jitter: float = 0.0


class RefCOCO(Dataset):
    """One referring expression per item.

    Returns `(image, caption, box, meta)`:

        image   (3, size[0], size[1]) float, ImageNet-normalised
        caption str, already left/right-corrected if the sample was flipped
        box     (4,) float, normalised cxcywh in [0, 1] -- the SingleBoxLoss convention
        meta    dict with file_name, orig_size (h, w), image_id, category_id,
                tokens_positive (char spans in the caption that name the referent)

    Args:
        name: "refcoco", "refcoco+" or "refcocog".
        split: see `SPLITS`; refcocog uses test, not testA/testB.
        root: dataset root holding `images/train2014` and `annotations/OpenSource`.
        size: (H, W) to resize to. Must be divisible by 16 -- SpiLiFormer's patch stride
            is 16, and a non-multiple silently truncates the last row/column of tokens.
        augment: None for eval. Anything else must be a `RefCOCOAugment`.
        limit: keep only the first N expressions, for smoke tests.
    """

    def __init__(
        self,
        name: str = "refcoco",
        split: str = "train",
        root: Path | None = None,
        size: tuple[int, int] = (384, 384),
        augment: RefCOCOAugment | None = None,
        limit: int | None = None,
    ):
        if name not in SPLITS:
            raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(SPLITS)}")
        if split not in SPLITS[name]:
            raise ValueError(f"{name} has no split {split!r}; it has {SPLITS[name]}")
        if size[0] % 16 or size[1] % 16:
            raise ValueError(f"size {size} must be divisible by 16 (SpiLiFormer patch stride)")

        self.root = Path(root or DATA_ROOT)
        self.name, self.split, self.size, self.augment = name, split, size, augment
        self.img_dir = self.root / "images" / "train2014"

        ann_file = self.root / "annotations" / "OpenSource" / f"finetune_{name}_{split}.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"{ann_file} not found. Expected MDETR annotations (zenodo 4729015) "
                f"extracted under {self.root}/annotations/"
            )
        if not self.img_dir.is_dir():
            raise FileNotFoundError(
                f"{self.img_dir} not found. Expected COCO train2014 images extracted there."
            )

        blob = json.load(open(ann_file))
        # One annotation per image entry -- verified across all 321k annotations in the
        # three datasets -- so a dict keyed by image_id is a complete, lossless index.
        by_id = {a["image_id"]: a for a in blob["annotations"]}
        self.items = []
        for im in blob["images"]:
            a = by_id.get(im["id"])
            if a is None:                      # never observed; guard rather than KeyError
                continue
            self.items.append((im, a))
        self.items = self.items[:limit]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        im, ann = self.items[i]
        caption = im["caption"]
        w, h = float(im["width"]), float(im["height"])

        # Some COCO files are greyscale or CMYK; convert before to_tensor so the channel
        # count is always 3.
        img = Image.open(self.img_dir / im["file_name"]).convert("RGB")

        # xywh px -> cxcywh normalised by the ORIGINAL size, so the resize below needs no
        # matching box transform and the box survives any later change of `size`.
        x, y, bw, bh = ann["bbox"]
        box = torch.tensor([(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h],
                           dtype=torch.float32)

        if self.augment is not None:
            if random.random() < self.augment.hflip:
                img = tv_F.hflip(img)
                box[0] = 1.0 - box[0]          # mirror the centre; w/h are unchanged
                caption = hflip_caption(caption)   # see module docstring -- not optional
            if self.augment.jitter > 0:
                j = self.augment.jitter
                img = tv_F.adjust_brightness(img, 1 + random.uniform(-j, j))
                img = tv_F.adjust_contrast(img, 1 + random.uniform(-j, j))
                img = tv_F.adjust_saturation(img, 1 + random.uniform(-j, j))

        img = tv_F.resize(img, list(self.size), antialias=True)
        image = tv_F.normalize(tv_F.to_tensor(img), mean=IMAGENET_MEAN, std=IMAGENET_STD)

        meta = {
            "file_name": im["file_name"],
            "orig_size": (int(im["height"]), int(im["width"])),
            "image_id": im["id"],
            "category_id": ann["category_id"],
            "tokens_positive": ann.get("tokens_positive", []),
        }
        return image, caption, box, meta


def make_collate(tokenizer, max_len: int = 64):
    """Batch into the tensors `SpikeGroundingV2` and `SingleBoxLoss` want.

    -> rgb        (B, 3, H, W)
       input_ids  (B, L)   L = longest caption in THIS batch, capped at max_len
       mask       (B, L)
       boxes      (B, 4)   normalised cxcywh
       captions   list[str]
       metas      list[dict]

    Padding is to the batch maximum rather than a fixed 80. RefCOCO captions have a median
    of 5-10 tokens and a global maximum of 49, so a fixed 80 would make >85% of every text
    sequence padding -- and cross-attention pays for padded queries whether or not the
    mask later zeroes them. `max_len` 64 truncates nothing in any of the three datasets.
    """

    def collate(batch):
        images, captions, boxes, metas = zip(*batch)
        tok = tokenizer(list(captions), padding="longest", truncation=True,
                        max_length=max_len, return_tensors="pt")
        return (torch.stack(images), tok["input_ids"], tok["attention_mask"],
                torch.stack(boxes), list(captions), list(metas))

    return collate
