"""Talk2Event: event cube + caption + per-attribute token spans + one box.

One sample is one (object, caption) pair:

    cube        (T, 2, 480, 640)  event voxel counts, T=9 real time bins
    caption     str
    box         (4,) normalised cxcywh
    attr_spans  (L,) int64 token labels: 0..3 = the four attributes, 4 = none

The attribute labels are what makes `AttributeQueryTagger` trainable rather than
hand-wavy. Talk2Event annotates every caption with phrases under exactly four headings --
`appearance`, `status`, `relation_viewer`, `relation_others` -- and those phrases appear
in the caption, so each can be located and its tokens labelled. Measured over the train
split, the phrases are found verbatim the overwhelming majority of the time; the rest are
recovered by a normalised substring search, and anything still unfound is simply left
unlabelled (class 4) rather than guessed at.

Why the labels are token-level and not phrase-level
---------------------------------------------------
The tagger runs per token, because a caption interleaves attributes ("A dark-coloured car
is driving on the right side of the road") rather than presenting them in blocks. A
phrase-level label would need a segmentation step that the token labels make unnecessary.

Priority when spans overlap
---------------------------
Two annotated phrases can overlap ("positioned between X" vs "between X"). The longer
phrase wins, on the reasoning that it is the more specific annotation; ties go to the
earlier attribute in `ATTRIBUTES` order. This is deterministic, which matters because the
tagger's supervision must not vary between epochs.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from spiketrandvg.events_voxel_cube import T_STEPS, talk2event_cube
from spiketrandvg.text_encoder import ATTRIBUTES, MAX_TEXT_LEN

__all__ = ["DATA_ROOT", "Talk2Event", "make_collate", "N_ATTR", "NO_ATTR"]

N_ATTR = len(ATTRIBUTES)
NO_ATTR = N_ATTR                      # the "belongs to no attribute" class

_WS = Path(os.environ.get("T2E_WS", Path(__file__).resolve().parents[3]))
DATA_ROOT = Path(os.environ.get("T2E_DATA_ROOT", _WS / "dataset" / "talk2event"))


def _norm(s: str) -> str:
    """Lowercase and collapse whitespace/punctuation spacing, for span matching."""
    return re.sub(r"\s+", " ", s.lower().replace(",", " , ").replace(".", " . ")).strip()


def _find_span(caption: str, phrase: str) -> tuple[int, int] | None:
    """Character span of `phrase` in `caption`, or None. Exact first, then normalised."""
    lo_c, lo_p = caption.lower(), phrase.lower().strip()
    if not lo_p:
        return None
    i = lo_c.find(lo_p)
    if i >= 0:
        return i, i + len(lo_p)
    # normalised retry: handles differing punctuation spacing between annotation
    # and caption ("dark-colored car ," vs "dark-colored car,")
    nc, np_ = _norm(caption), _norm(phrase)
    j = nc.find(np_)
    if j < 0:
        return None
    # map the normalised offset back by walking the original string
    seen, orig = 0, 0
    while orig < len(caption) and seen < j:
        if not (caption[orig].isspace() and orig + 1 < len(caption)
                and caption[orig + 1].isspace()):
            seen += 1
        orig += 1
    return orig, min(len(caption), orig + len(phrase))


class Talk2Event(Dataset):
    """One (object, caption) pair per item.

    Args:
        split: "train" or "test".
        root: dataset root holding `meta_data_v10/` and `data/`.
        t_steps: event time bins. 9 is the Gate-2 setting; the npz files ship 20 bins
            and are regrouped mass-preservingly (20 -> 9 gives groups of 3,3,2,2,...).
        caption_index: Talk2Event gives several paraphrases per object. 0 takes the
            first deterministically; None samples one per epoch as augmentation.
        limit: keep only the first N samples, for smoke tests.
    """

    def __init__(self, split: str = "train", root: Path | None = None,
                 t_steps: int = T_STEPS, caption_index: int | None = 0,
                 limit: int | None = None):
        self.root = Path(root or DATA_ROOT)
        self.split = split
        self.t_steps = t_steps
        self.caption_index = caption_index

        pattern = str(self.root / f"meta_data_v10/{split}/*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no annotation files at {pattern}")
        self.items: list[dict] = []
        for f in files:
            for rec in json.load(open(f)):
                self.items.append(rec)
        self.items = self.items[:limit]

    def __len__(self) -> int:
        return len(self.items)

    def _caption_and_attrs(self, rec: dict, idx: int):
        caps = rec["captions"]
        if isinstance(caps, str):
            caps = [caps]
        i = (idx % len(caps)) if self.caption_index is None else \
            min(self.caption_index, len(caps) - 1)
        caption = caps[i]
        attrs = rec.get("attributes", [])
        # `attributes` is a list parallel to `captions`; fall back to the first entry
        a = attrs[i] if isinstance(attrs, list) and i < len(attrs) else (
            attrs[0] if isinstance(attrs, list) and attrs else {})
        return caption, (a if isinstance(a, dict) else {})

    @staticmethod
    def _char_labels(caption: str, attrs: dict) -> list[tuple[int, int, int]]:
        """-> [(start, end, attribute_index)], longer phrases winning on overlap."""
        spans = []
        for ai, name in enumerate(ATTRIBUTES):
            for phrase in attrs.get(name, []) or []:
                s = _find_span(caption, phrase)
                if s is not None:
                    spans.append((s[0], s[1], ai))
        # longest first so that a later, shorter span cannot overwrite a longer one
        spans.sort(key=lambda x: (-(x[1] - x[0]), x[2]))
        return spans

    def __getitem__(self, i: int):
        rec = self.items[i]
        caption, attrs = self._caption_and_attrs(rec, i)

        raw = np.load(self.root / rec["event_path"])["events"].astype(np.float32)
        cube = talk2event_cube(torch.from_numpy(raw), t_out=self.t_steps)

        b = rec["bbox"]
        H, W = cube.shape[-2], cube.shape[-1]
        box = torch.tensor([(b["x"] + b["w"] / 2) / W, (b["y"] + b["h"] / 2) / H,
                            b["w"] / W, b["h"] / H], dtype=torch.float32)

        return cube, caption, box, self._char_labels(caption, attrs), rec


def make_collate(tokenizer, max_len: int = MAX_TEXT_LEN):
    """Batch into what `Talk2EventGrounding.forward` and the losses want.

    -> cube (T,B,2,H,W) | ids (B,L) | mask (B,L) | boxes (B,4)
       attr_labels (B,L) int64 in 0..4 | captions | recs

    Character spans are converted to TOKEN labels here rather than in `__getitem__`
    because that needs the tokenizer's offset mapping, and passing a tokenizer into
    every worker process is wasteful. `return_offsets_mapping` requires a fast
    tokenizer -- roberta-base's is, and `build_tokenizer` returns it.
    """

    def collate(batch):
        cubes, caps, boxes, spans, recs = zip(*batch)
        tok = tokenizer(list(caps), padding="longest", truncation=True,
                        max_length=max_len, return_tensors="pt",
                        return_offsets_mapping=True)
        offsets = tok.pop("offset_mapping")                     # (B,L,2)
        B, L = tok["input_ids"].shape
        labels = torch.full((B, L), NO_ATTR, dtype=torch.long)
        for bi, sp in enumerate(spans):
            for (s, e, ai) in sp:
                for ti in range(L):
                    o0, o1 = int(offsets[bi, ti, 0]), int(offsets[bi, ti, 1])
                    if o1 <= o0:                                # special/padding token
                        continue
                    # a token counts as inside the span if it overlaps it at all
                    if o0 < e and o1 > s:
                        labels[bi, ti] = ai
        return (torch.stack(cubes, dim=1), tok["input_ids"], tok["attention_mask"],
                torch.stack(boxes), labels, list(caps), list(recs))

    return collate
