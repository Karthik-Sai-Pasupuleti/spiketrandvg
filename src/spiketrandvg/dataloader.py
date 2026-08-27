"""Data loading: RefCOCO/+/g referring expressions, and Talk2Event event streams.

Two datasets, one module, because they pose the same question in different modalities:
given an image (or an event stream) and a sentence, which object does the sentence mean?

  event voxel cubes -> `talk2event_cube`, `T_STEPS`     generic, any pre-binned source
  Talk2Event        -> `Talk2Event`, `make_t2e_collate` events, driving scenes
  RefCOCO/+/g       -> `RefCOCO`, `make_collate`        RGB, 120k expressions

Section order matters: the cube helpers come first because `Talk2Event` uses `T_STEPS` as
a default argument, which Python evaluates when the class body runs.

Both loaders emit boxes in **normalised cxcywh**, the convention the models and losses use
end to end, so no unit conversion happens anywhere in a training loop.
"""

from __future__ import annotations

from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset
import glob
import json
import numpy as np
import os
import random
import re
import torch
import torchvision.transforms.functional as tv_F

from spiketrandvg.textencoder import ATTRIBUTES, MAX_TEXT_LEN


# ====================================================================================
# from events_voxel_cube.py
# ====================================================================================

__all__ = [
    "events_to_coords_feats",
    "coords_feats_to_cube",
    "events_to_voxel_cube",
    "prebinned_to_voxel_cube",
    "rebin_time",
    "time_group_sizes",
    "talk2event_cube",
    "T_FILE",
    "T_STEPS",
    "POLARITIES",
]

# --- project configuration -------------------------------------------------- #
# Talk2Event .npz files are binned into 10 time bins x 2 polarities = 20 channels.
T_FILE = 10
POLARITIES = 2

# Simulation steps the network runs. 5 divides 10 exactly, so each step merges two
# file bins: equal step durations, integer counts, and no interpolation assumption.
# (T=4 was considered and rejected: 10/4 = 2.5, which forces either unequal step
# durations or fractional splitting of a bin whose sub-structure is already lost.)
# 5 timesteps over the file's 10 native time bins: an exact 2-bins-per-step grouping,
# so the rebin is mass-preserving with no uneven groups (T=9 needed [3,3,2,2,2,2,2,2,2]).
# Each step keeps the 2 polarity channels, giving (5, 2, 480, 640) -- 5 x 2 = the 10
# planes the raw file carries per polarity pair.
T_STEPS = 5


def events_to_coords_feats(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    *,
    T: int,
    tbin: int,
    sample_size: int,
    height: int,
    width: int,
    spatial_quant: tuple[int, int] = (1, 1),
    legacy_offset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw events -> the sparse (coords, feats) pair SFOD caches on disk.

    Args:
        t: timestamps in microseconds, ascending. Made relative to t[0], as upstream does.
        x, y: pixel coordinates.
        p: polarity, 0/1 (0 is treated as negative).
        T: number of macro timesteps.
        tbin: micro time bins per macro step.
        sample_size: total sample duration in microseconds.
        height, width: sensor resolution before spatial quantization.
        spatial_quant: (y_scale, x_scale) downscale factors, upstream's [1, 1].
        legacy_offset: reproduce the data-dependent channel offset bug.

    Returns:
        coords: (N, 3) int16 of (t_step, y_q, x_q)
        feats:  (N, 2*tbin) bool one-hot over (polarity, micro-bin)
    """
    if T <= 0 or tbin <= 0:
        raise ValueError(f"T and tbin must be positive, got T={T}, tbin={tbin}")
    if t.size == 0:
        return torch.zeros((0, 3), dtype=torch.int16), torch.zeros((0, 2 * tbin), dtype=torch.bool)

    t = t.astype(np.int64) - int(t[0])  # upstream: events['t'] -= events['t'][0]
    ys, xs = spatial_quant
    t_quant = sample_size // T
    quant = np.array([t_quant, ys, xs], dtype=np.int64)
    qh, qw = height // ys, width // xs

    coords = np.stack([t, y.astype(np.int64), x.astype(np.int64)], axis=1)
    coords = np.floor(coords / quant)
    coords[:, 0] = np.clip(coords[:, 0], 0, T - 1)  # guard t beyond the window
    coords[:, 1] = np.clip(coords[:, 1], 0, qh - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, qw - 1)

    # --- micro-bin (tbin) channel index, jointly with polarity ---
    tbin_size = t_quant / tbin
    tbin_coords = (t % t_quant) // tbin_size

    polarity = p.copy().astype(np.int8)
    polarity[p == 0] = -1
    ch = polarity * (tbin_coords + 1)
    ch[ch > 0] -= 1
    ch = ch + (int((tbin_coords + 1).max()) if legacy_offset else tbin)
    ch = np.clip(ch, 0, 2 * tbin - 1)

    feats = torch.nn.functional.one_hot(
        torch.from_numpy(ch.astype(np.int64)), 2 * tbin
    ).to(torch.bool)
    return torch.from_numpy(coords.astype(np.int16)), feats


def coords_feats_to_cube(
    coords: torch.Tensor,
    feats: torch.Tensor,
    *,
    T: int,
    C: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Sparse (coords, feats) -> dense (T, C, H, W) float32 cube of event counts.

    Mirrors GEN1DetectionDataset.__getitem__: build a sparse COO tensor of shape
    (T, H, W, C), coalesce (which SUMS duplicates into counts), densify, then
    permute to channel-second.
    """
    if coords.numel() == 0:
        return torch.zeros((T, C, height, width), dtype=torch.float32)
    cube = torch.sparse_coo_tensor(
        coords.t().to(torch.int64),
        feats.to(torch.float32),
        size=(T, height, width, C),
        check_invariants=False,  # indices are clamped above; skip the check + its warning
    )
    return cube.coalesce().to_dense().permute(0, 3, 1, 2).contiguous()


def events_to_voxel_cube(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    *,
    T: int,
    tbin: int,
    sample_size: int,
    height: int,
    width: int,
    spatial_quant: tuple[int, int] = (1, 1),
    legacy_offset: bool = False,
) -> torch.Tensor:
    """Raw events -> dense (T, 2*tbin, H, W) cube. Convenience wrapper."""
    coords, feats = events_to_coords_feats(
        t, x, y, p, T=T, tbin=tbin, sample_size=sample_size,
        height=height, width=width, spatial_quant=spatial_quant,
        legacy_offset=legacy_offset,
    )
    ys, xs = spatial_quant
    return coords_feats_to_cube(
        coords, feats, T=T, C=2 * tbin, height=height // ys, width=width // xs
    )


# --------------------------------------------------------------------------- #
# Talk2Event bridge
# --------------------------------------------------------------------------- #
def talk2event_cube(events: torch.Tensor, t_out: int | None = None) -> torch.Tensor:
    """Talk2Event npz array -> the project's canonical cube: (T_STEPS, 2, H, W).

    Applies the configuration decided once at the top of this module, so call
    sites do not repeat the bin counts. For (20, 480, 640) input this returns
    (5, 2, 480, 640): each step sums two adjacent file bins.

    Mass-preserving and exact -- with T_FILE divisible by T_STEPS, the "group"
    and "uniform" rebin modes are identical, so no mode choice arises.
    """
    return prebinned_to_voxel_cube(
        events, T=T_FILE, polarities=POLARITIES, t_out=t_out or T_STEPS
    )


def prebinned_to_voxel_cube(
    events: torch.Tensor,
    *,
    T: int = T_FILE,
    polarities: int = POLARITIES,
    polarity_major: bool = True,
    t_out: int | None = None,
    rebin_mode: str = "group",
) -> torch.Tensor:
    """Talk2Event's pre-binned (P*T, H, W) count array -> (T, P, H, W) cube.

    Talk2Event ships NO raw event streams -- every .npz holds an already-binned
    dense array of shape (20, 480, 640) uint8 counts, so the functions above
    cannot be applied to it. This reshapes that array into the same
    (T, C, H, W) cube layout the SNN consumes.

    The channel order is POLARITY-MAJOR, i.e. index = p * T + t (verified on the
    real data: per-channel event sums step up between channel 9 and 10). Pass
    polarity_major=False for an interleaved (t-major) file.

    Note this yields C = 2 (one channel per polarity per step), not 2*tbin --
    the file's own binning already fixes the temporal resolution at T bins, so
    there are no sub-bins left to split.

    T is the number of bins IN THE FILE (10 for Talk2Event) and must satisfy
    polarities * T == events.shape[0]. To run the network at fewer steps, leave T
    at the file's value and set t_out; see rebin_time for the two merge modes.
    """
    if events.dim() != 3:
        raise ValueError(f"expected (C, H, W), got {tuple(events.shape)}")
    c, h, w = events.shape
    if c != polarities * T:
        raise ValueError(
            f"channel count {c} != polarities({polarities}) * T({T}); "
            "check T against the file's binning"
        )
    if polarity_major:
        # (P*T, H, W) -> (P, T, H, W) -> (T, P, H, W)
        cube = events.reshape(polarities, T, h, w).permute(1, 0, 2, 3).contiguous()
    else:
        # interleaved: (T*P, H, W) -> (T, P, H, W)
        cube = events.reshape(T, polarities, h, w).contiguous()
    if t_out is not None and t_out != T:
        cube = rebin_time(cube, t_out, mode=rebin_mode)
    return cube


# --------------------------------------------------------------------------- #
# Temporal rebinning
# --------------------------------------------------------------------------- #
def time_group_sizes(t_in: int, t_out: int) -> list[int]:
    """Split t_in bins into t_out contiguous groups, as evenly as possible.

    Extra bins go to the earliest groups (numpy.array_split convention), e.g.
    10 -> 4 gives [3, 3, 2, 2].
    """
    if not 0 < t_out <= t_in:
        raise ValueError(f"need 0 < t_out <= t_in, got t_out={t_out}, t_in={t_in}")
    base, rem = divmod(t_in, t_out)
    return [base + 1] * rem + [base] * (t_out - rem)


def rebin_time(cube: torch.Tensor, t_out: int, mode: str = "group") -> torch.Tensor:
    """Merge a cube's time axis down to t_out steps. Input (T, C, H, W).

    Total event mass is preserved exactly by both modes (they only regroup counts).

    mode="group" (default)
        Sum contiguous groups of whole input bins, sizes from time_group_sizes().
        Counts stay integral and no sub-bin structure is invented, but when t_in is
        not divisible by t_out the output bins span unequal durations -- for 10->4
        that is [3, 3, 2, 2] bins, i.e. 30/30/20/20 ms if each input bin is 10 ms.

    mode="uniform"
        Every output bin covers exactly t_in/t_out input bins, splitting a boundary
        bin's counts proportionally between the two output bins it straddles. All
        output bins span equal durations (25 ms each for 10->4), at the cost of
        fractional counts and an implicit assumption that events are spread evenly
        inside an input bin.

    Which to prefer: "group" if you want untouched integer counts and can live with
    uneven step durations; "uniform" if equal time steps matter to the temporal
    dynamics. When t_in is divisible by t_out the two are identical.
    """
    if cube.dim() != 4:
        raise ValueError(f"expected (T, C, H, W), got {tuple(cube.shape)}")
    t_in = cube.shape[0]
    if t_out == t_in:
        return cube
    if mode == "group":
        chunks = torch.split(cube, time_group_sizes(t_in, t_out), dim=0)
        return torch.stack([c.sum(0) for c in chunks], dim=0).contiguous()
    if mode == "uniform":
        # weight[j, i] = overlap between output window j and input bin i
        step = t_in / t_out
        w = torch.zeros(t_out, t_in, dtype=cube.dtype)
        for j in range(t_out):
            lo, hi = j * step, (j + 1) * step
            for i in range(int(lo), min(int(np.ceil(hi)), t_in)):
                w[j, i] = max(0.0, min(hi, i + 1) - max(lo, i))
        return torch.einsum("ji,ichw->jchw", w, cube).contiguous()
    raise ValueError(f"unknown mode {mode!r}, expected 'group' or 'uniform'")


# ====================================================================================
# from talk2event.py
# ====================================================================================

__all__ = ["T2E_DATA_ROOT", "T2E_VAL_SEQ_FILE", "Talk2Event", "make_t2e_collate",
           "N_ATTR", "NO_ATTR", "t2e_val_sequences"]

N_ATTR = len(ATTRIBUTES)
NO_ATTR = N_ATTR                      # the "belongs to no attribute" class

_WS = Path(os.environ.get("T2E_WS", Path(__file__).resolve().parents[3]))
T2E_DATA_ROOT = Path(os.environ.get("T2E_DATA_ROOT", _WS / "dataset" / "talk2event"))

# The held-out validation split. Talk2Event ships train/test only, so selecting
# `best.pth` on `test` is model selection on the test set -- every Talk2Event number
# recorded before 2026-08-27 was a best-of-N chosen that way. This file names 8 of the 47
# TRAIN sequences as `val`; the split is BY SEQUENCE, so no driving scene appears in both
# and `val` is a domain shift to unseen streets exactly as `test` is. It is written once
# and frozen -- regenerating it would silently change what every recorded number means.
T2E_VAL_SEQ_FILE = Path(__file__).with_name("talk2event_val_sequences.txt")


def t2e_val_sequences(path: Path | None = None) -> frozenset[str]:
    """The frozen `val` sequence names, from `talk2event_val_sequences.txt`."""
    f = Path(path or T2E_VAL_SEQ_FILE)
    if not f.exists():
        raise FileNotFoundError(
            f"{f} is missing. It defines the train/val split and is not regenerable "
            f"without invalidating every number measured against it.")
    names = frozenset(ln.strip() for ln in f.read_text().splitlines()
                      if ln.strip() and not ln.startswith("#"))
    if not names:
        raise ValueError(f"{f} names no sequences")
    return names


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
        split: "train", "val" or "test". `train` and `val` both read the shipped
            train annotations and partition them BY SEQUENCE on
            `talk2event_val_sequences.txt` -- 39 sequences (6535 samples) against 8
            (1140), with no scene in common. `test` is the shipped test split and is
            not touched during model selection.
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
        self.root = Path(root or T2E_DATA_ROOT)
        self.split = split
        self.t_steps = t_steps
        self.caption_index = caption_index

        # One annotation file per driving sequence, so the train/val partition is a
        # filter on file names -- there is no per-sample decision to get wrong, and no
        # way for a frame of a val scene to reach train.
        if split in ("train", "val"):
            src, val_seqs = "train", t2e_val_sequences()
            want_val = split == "val"
        elif split == "test":
            src, val_seqs, want_val = "test", frozenset(), False
        else:
            raise ValueError(f"unknown split {split!r}; expected train, val or test")

        pattern = str(self.root / f"meta_data_v10/{src}/*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no annotation files at {pattern}")
        if src == "train":
            unknown = val_seqs - {Path(f).stem for f in files}
            if unknown:
                raise ValueError(f"{T2E_VAL_SEQ_FILE} names sequences that do not exist "
                                 f"in the train split: {sorted(unknown)}")
            files = [f for f in files if (Path(f).stem in val_seqs) == want_val]
        self.sequences = tuple(Path(f).stem for f in files)
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


def make_t2e_collate(tokenizer, max_len: int = MAX_TEXT_LEN):
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


# ====================================================================================
# from dataset.py
# ====================================================================================

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

_WS = Path(os.environ.get("T2E_WS", Path(__file__).resolve().parents[3]))
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
