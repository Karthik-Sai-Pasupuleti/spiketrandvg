"""Detection augmentation for event cubes.

Why this is the highest-value change available, measured rather than assumed:

The detector reaches mAP@0.5 = 0.97 on train and 0.38 on test. Inspecting the split
explains most of that gap and points here:

* train and test share **zero** driving sequences (47 vs 13, no overlap), so test is a
  domain shift to unseen streets, not held-out frames of seen ones;
* annotated frames within a sequence are spaced a constant 8 apart, so the nominal 4433
  training frames are really **47 distinct scenes** sampled densely -- trivially
  memorisable by a 23M-parameter network;
* there was previously **no augmentation of any kind**.

Only the third is fixable by training choices, and it is the one that attacks the second:
every transform here manufactures scene diversity the 47 drives do not contain.

Geometry conventions
--------------------
Cubes are (T, 2, H, W) event counts, boxes (N, 4) xyxy in PIXELS, labels (N,). Every
transform returns the same layout at the same H, W, so the collate function and the
strides downstream are unaffected. Boxes are clipped to the frame and then dropped if they
survive with less than `min_area_frac` of their pre-transform area or fewer than
`min_side` pixels on a side -- otherwise a sliver of a truncated car becomes a label that
is impossible to predict and the loss learns noise.

Event-specific note
-------------------
Spatial resizing uses **area** interpolation, not bilinear. Event counts are extensive
quantities; area interpolation conserves total count under downscaling, whereas bilinear
smears sparse impulses and inflates apparent activity.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

__all__ = ["AugmentConfig", "augment_sample", "mosaic"]


class AugmentConfig:
    """Probabilities and ranges. All defaults are the ones the training script uses."""

    def __init__(
        self,
        hflip: float = 0.5,
        scale_jitter: float = 0.5,
        scale_range: tuple[float, float] = (0.6, 1.5),
        translate: float = 0.1,
        event_dropout: float = 0.3,
        dropout_frac: float = 0.15,
        mosaic: float = 0.5,
        min_area_frac: float = 0.25,
        min_side: float = 4.0,
    ):
        self.hflip = hflip
        self.scale_jitter = scale_jitter
        self.scale_range = scale_range
        self.translate = translate
        self.event_dropout = event_dropout
        self.dropout_frac = dropout_frac
        self.mosaic = mosaic
        self.min_area_frac = min_area_frac
        self.min_side = min_side


def _filter(boxes, labels, orig_area, H, W, cfg):
    """Clip to frame, then drop boxes that survived too small to be learnable."""
    if boxes.numel() == 0:
        return boxes, labels
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= cfg.min_side) & (h >= cfg.min_side) & ((w * h) >= cfg.min_area_frac * orig_area)
    return boxes[keep], labels[keep]


def _hflip(cube, boxes, labels):
    W = cube.shape[-1]
    cube = torch.flip(cube, dims=[-1])
    if boxes.numel():
        boxes = boxes.clone()
        x0 = boxes[:, 0].clone()
        boxes[:, 0] = W - boxes[:, 2]
        boxes[:, 2] = W - x0
    return cube, boxes, labels


def _scale_translate(cube, boxes, labels, cfg):
    """Resize by a random factor, then place into the original frame at a random offset.

    Scale and translation are handled together because both are just a choice of where the
    resized content lands on a fixed-size canvas.
    """
    T, C, H, W = cube.shape
    s = random.uniform(*cfg.scale_range)
    nh, nw = max(8, int(round(H * s))), max(8, int(round(W * s)))
    resized = F.interpolate(cube.reshape(T * C, 1, H, W), size=(nh, nw),
                            mode="area").reshape(T, C, nh, nw)

    out = torch.zeros_like(cube)
    max_dx, max_dy = int(cfg.translate * W), int(cfg.translate * H)
    # dst_x is where the resized content's left edge sits on the canvas
    dst_x = (W - nw) // 2 + random.randint(-max_dx, max_dx)
    dst_y = (H - nh) // 2 + random.randint(-max_dy, max_dy)

    # intersect [dst, dst+n) with [0, W) in canvas space
    sx0, sy0 = max(0, -dst_x), max(0, -dst_y)                 # crop offset in resized
    cx0, cy0 = max(0, dst_x), max(0, dst_y)                   # paste offset in canvas
    cw = min(nw - sx0, W - cx0)
    ch = min(nh - sy0, H - cy0)
    if cw > 0 and ch > 0:
        out[..., cy0:cy0 + ch, cx0:cx0 + cw] = resized[..., sy0:sy0 + ch, sx0:sx0 + cw]

    if boxes.numel():
        orig_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        b = boxes.clone() * s
        b[:, 0::2] += dst_x
        b[:, 1::2] += dst_y
        boxes, labels = _filter(b, labels, orig_area, H, W, cfg)
    return out, boxes, labels


def _event_dropout(cube, frac):
    """Zero a random fraction of active sites -- sensor-noise-like regularisation."""
    mask = (torch.rand_like(cube) >= frac).to(cube.dtype)
    return cube * mask


def mosaic(samples, cfg):
    """Four samples -> one, at native scale.

    Built on a 2H x 2W canvas and then cropped back to H x W around a jittered centre, so
    objects keep their original pixel size. Halving each frame into a 2x2 grid instead
    would shrink every object by 2x, and Talk2Event's median box is already ~1.3% of the
    frame -- the resolution is doing real work here and must not be given away.
    """
    (cube0, _, _), *_ = samples
    T, C, H, W = cube0.shape
    canvas = torch.zeros(T, C, 2 * H, 2 * W, dtype=cube0.dtype)
    all_boxes, all_labels = [], []
    for k, (cube, boxes, labels) in enumerate(samples[:4]):
        oy, ox = (k // 2) * H, (k % 2) * W
        canvas[..., oy:oy + H, ox:ox + W] = cube
        if boxes.numel():
            b = boxes.clone()
            b[:, 0::2] += ox
            b[:, 1::2] += oy
            all_boxes.append(b)
            all_labels.append(labels)

    boxes = torch.cat(all_boxes) if all_boxes else torch.zeros(0, 4)
    labels = torch.cat(all_labels) if all_labels else torch.zeros(0, dtype=torch.long)

    # crop an HxW window centred near the canvas middle, so all four tiles contribute
    cx = W + random.randint(-W // 2, W // 2)
    cy = H + random.randint(-H // 2, H // 2)
    x0 = min(max(cx - W // 2, 0), 2 * W - W)
    y0 = min(max(cy - H // 2, 0), 2 * H - H)
    out = canvas[..., y0:y0 + H, x0:x0 + W].contiguous()

    if boxes.numel():
        orig_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        b = boxes.clone()
        b[:, 0::2] -= x0
        b[:, 1::2] -= y0
        boxes, labels = _filter(b, labels, orig_area, H, W, cfg)
    return out, boxes, labels


def augment_sample(cube, boxes, labels, cfg, extra=None):
    """Apply the configured pipeline to one sample.

    `extra` is three additional (cube, boxes, labels) tuples, or None. Mosaic runs iff
    `extra` is supplied -- the caller owns that coin flip, so the dataset can skip loading
    three extra .npz files (4x the I/O) on the samples where mosaic will not fire.
    Geometric transforms run after mosaic so they apply to the composed scene.
    """
    if extra and len(extra) >= 3:
        cube, boxes, labels = mosaic([(cube, boxes, labels), *extra[:3]], cfg)

    if random.random() < cfg.hflip:
        cube, boxes, labels = _hflip(cube, boxes, labels)
    if random.random() < cfg.scale_jitter:
        cube, boxes, labels = _scale_translate(cube, boxes, labels, cfg)
    if random.random() < cfg.event_dropout:
        cube = _event_dropout(cube, cfg.dropout_frac)
    return cube, boxes, labels
