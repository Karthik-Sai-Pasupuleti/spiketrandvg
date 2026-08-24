"""Reader for jAER AEDAT files, as shipped by CIFAR10-DVS.

Why this exists: spikingjelly 0.0.0.0.14 provides `load_events`, but its
`load_raw_events` calls `np.fromstring(data, dtype='>u4')`, which NumPy 2 removed
("The binary mode of fromstring is removed, use frombuffer instead"). Rather than
patch an installed dependency, the few lines of parsing are reimplemented here with
`np.frombuffer`. The bit masks are taken from spikingjelly's own CIFAR10-DVS
definition so the decoded coordinates match it exactly.

Format: an ASCII header of '#'-prefixed lines, then a flat sequence of 32-bit
big-endian words alternating (address, timestamp). The address packs x, y and
polarity into bit fields.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Bit layout for CIFAR10-DVS, from CIFAR10DVS.load_origin_data in
# spikingjelly/datasets/cifar10_dvs.py (NOT the module-level defaults in that file,
# which are for a different sensor layout and decode this data as 8x1).
X_MASK, X_SHIFT = 0xFE, 1
Y_MASK, Y_SHIFT = 0x7F00, 8
POLARITY_MASK = 0x1          # no shift: polarity is the low bit

DVS128_HW = (128, 128)
_MAX_COORD = 127

# The same source applies an axis swap with inversion after decoding:
#     x = 127 - y_raw,  y = 127 - x_raw,  p = 1 - p_raw
# See the comment there and jackd/events-tfds#1 for why this orientation is the
# one that matches the reference MATLAB reader.

__all__ = ["read_aedat", "aedat_to_voxel_cube"]


def _skip_header(fp) -> int:
    """Return the byte offset just past the '#' comment header."""
    pos = 0
    line = fp.readline()
    while line:
        try:
            text = line.decode().strip()
        except UnicodeDecodeError:
            break               # binary payload started mid-line
        if not text or text[0] != "#":
            break
        pos += len(line)
        line = fp.readline()
    return pos


def read_aedat(path: str | Path) -> dict[str, np.ndarray]:
    """Parse one .aedat file into a raw asynchronous event stream.

    Returns a dict with equal-length arrays:
        t  int64  timestamps in microseconds, made relative to the first event
        x  int16  column, 0..127 for DVS128
        y  int16  row
        p  int8   polarity, 0 or 1

    Events are returned in file order, which is chronological.
    """
    path = Path(path)
    with path.open("rb") as fp:
        offset = _skip_header(fp)
        fp.seek(offset)
        raw = fp.read()

    words = np.frombuffer(raw, dtype=">u4")
    if words.size % 2:
        words = words[:-1]      # trailing partial record, ignore it
    addr, ts = words[0::2], words[1::2]

    x_raw = ((addr & X_MASK) >> X_SHIFT).astype(np.int16)
    y_raw = ((addr & Y_MASK) >> Y_SHIFT).astype(np.int16)
    p_raw = (addr & POLARITY_MASK).astype(np.int8)

    # axis swap + inversion, and polarity inversion (see note above)
    x = (_MAX_COORD - y_raw).astype(np.int16)
    y = (_MAX_COORD - x_raw).astype(np.int16)
    p = (1 - p_raw).astype(np.int8)
    t = ts.astype(np.int64)
    if t.size:
        t = t - t[0]
    return {"t": t, "x": x, "y": y, "p": p}


def aedat_to_voxel_cube(
    path: str | Path,
    *,
    T: int,
    tbin: int = 1,
    height: int = DVS128_HW[0],
    width: int = DVS128_HW[1],
    sample_size: int | None = None,
):
    """Convenience: .aedat file -> (T, 2*tbin, H, W) voxel cube.

    sample_size (microseconds) defaults to the file's own duration, so the T bins
    span exactly the recording.
    """
    from spiketrandvg.datasets.events_voxel_cube import events_to_voxel_cube

    ev = read_aedat(path)
    if sample_size is None:
        sample_size = int(ev["t"].max()) + 1 if ev["t"].size else 1
    return events_to_voxel_cube(
        ev["t"], ev["x"], ev["y"], ev["p"],
        T=T, tbin=tbin, sample_size=sample_size, height=height, width=width,
    )
