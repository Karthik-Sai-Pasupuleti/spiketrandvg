"""Event voxel-cube construction, extracted from SFOD.

Source: repositories/SFOD/datasets/gen1_od_dataset.py (GEN1DetectionDataset.create_data
+ __getitem__), itself modified from loiccordone/object-detection-with-spiking-neural-networks.
Lifted out of the GEN1 Dataset class so the representation can be used independently of
Prophesee .dat loading.

The cube
--------
A voxel cube has shape (T, C, H, W) with C = 2 * tbin:

  T     macro timesteps  -- the SNN's simulation steps
  tbin  micro time bins within each macro step
  C     2 * tbin channels, encoding (polarity, micro-bin) jointly
  H, W  spatial dims (optionally downscaled by quantization_size)

Values are event COUNTS: duplicates at the same (t, y, x, c) accumulate via
sparse coalesce(), so a voxel holds how many events fell in that cell.

Channel layout (with the constant offset, i.e. legacy_offset=False):
  channels [0, tbin)      -> negative polarity, micro-bin index REVERSED
  channels [tbin, 2*tbin) -> positive polarity, micro-bin index in order

Two upstream defects, preserved but flagged
-------------------------------------------
1. `feats` was computed from a one-hot of raw polarity and then immediately
   overwritten by the tbin one-hot (gen1_od_dataset.py:160 vs :182). Dead code,
   dropped here.
2. The channel offset was `(tbin_coords + 1).max()` -- DATA DEPENDENT. When a
   sample's events do not span all `tbin` micro-bins, the offset shrinks and the
   positive-polarity block slides down into the negative block, so the same
   channel index means different things in different samples. Verified: with
   tbin=2 and all events in the first micro-bin, positive events land on channel
   1 and negative on channel 0, whereas a fully-populated sample puts positives
   on 2..3 and negatives on 0..1. Fixed here by offsetting with the constant
   `tbin`; pass legacy_offset=True to reproduce the original behaviour exactly.
"""

from __future__ import annotations

import numpy as np
import torch

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
T_STEPS = 9


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
