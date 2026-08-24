"""CIFAR10-DVS as voxel cubes, with an on-disk cache.

CIFAR10-DVS ships 10,000 .aedat recordings (10 classes x 1000), each ~1.3 s of raw
events from a 128x128 DVS. Parsing and voxelizing one costs ~26 ms, so ~4.3 minutes
per epoch would be spent on decoding alone. Instead every sample is voxelized once
into a single uint8 memmap (1.53 GiB for T=5), which then loads at memory-bandwidth
speed. uint8 is safe: the busiest voxel observed holds 37 events, far below 255.

There is no official train/test split for this dataset. The convention in the
literature is 9:1; this module builds a deterministic, class-stratified split so
results are reproducible and every class is represented in both halves.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from spiketrandvg.datasets.aedat import aedat_to_voxel_cube

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
SENSOR_HW = (128, 128)


def default_root() -> Path:
    return Path(__file__).resolve().parents[4] / "dataset" / "cifar10_dvs"


def _voxelize_one(args) -> tuple[int, np.ndarray]:
    idx, path, T, tbin = args
    cube = aedat_to_voxel_cube(path, T=T, tbin=tbin,
                              height=SENSOR_HW[0], width=SENSOR_HW[1])
    return idx, cube.numpy().astype(np.uint8)


class CIFAR10DVSVoxel(Dataset):
    """Voxel-cube CIFAR10-DVS. Items are (cube float32 (T, 2*tbin, 128, 128), label)."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        T: int = 5,
        tbin: int = 1,
        train_fraction: float = 0.9,
        seed: int = 0,
        workers: int = 8,
        rebuild: bool = False,
    ):
        if split not in ("train", "test", "all"):
            raise ValueError(f"split must be train/test/all, got {split!r}")
        self.root = Path(root) if root else default_root()
        self.T, self.tbin, self.split = T, tbin, split

        extract = self.root / "extract"
        if not extract.is_dir():
            raise FileNotFoundError(
                f"{extract} not found -- run tools/download_cifar10_dvs.py first"
            )

        # stable file ordering => stable indices across runs
        self.files = sorted(extract.rglob("*.aedat"), key=lambda p: (p.parent.name, p.name))
        if not self.files:
            raise FileNotFoundError(f"no .aedat files under {extract}")
        self.labels_all = np.array(
            [CLASSES.index(p.parent.name) for p in self.files], dtype=np.int64
        )

        self.cache_path = self.root / f"voxel_T{T}_tbin{tbin}.npy"
        self.meta_path = self.cache_path.with_suffix(".json")
        if rebuild or not (self.cache_path.exists() and self.meta_path.exists()):
            self._build_cache(workers)
        self._check_meta()

        self.data = np.load(self.cache_path, mmap_mode="r")
        self.indices = self._split_indices(train_fraction, seed)

    # -- cache -----------------------------------------------------------------
    def _build_cache(self, workers: int) -> None:
        n = len(self.files)
        C = 2 * self.tbin
        shape = (n, self.T, C, *SENSOR_HW)
        print(f"[cifar10-dvs] building cache {shape} uint8 "
              f"({np.prod(shape) / 2**30:.2f} GiB) with {workers} workers")
        out = np.lib.format.open_memmap(
            self.cache_path, mode="w+", dtype=np.uint8, shape=shape
        )
        jobs = [(i, str(p), self.T, self.tbin) for i, p in enumerate(self.files)]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for idx, cube in pool.map(_voxelize_one, jobs, chunksize=16):
                out[idx] = cube
                done += 1
                if done % 1000 == 0:
                    print(f"  {done}/{n}", flush=True)
        out.flush()
        del out
        self.meta_path.write_text(json.dumps({
            "n": n, "T": self.T, "tbin": self.tbin,
            "sensor_hw": list(SENSOR_HW), "classes": list(CLASSES),
            "files": [p.name for p in self.files],
        }))
        print(f"[cifar10-dvs] cache written to {self.cache_path}")

    def _check_meta(self) -> None:
        meta = json.loads(self.meta_path.read_text())
        if meta["n"] != len(self.files) or meta["T"] != self.T or meta["tbin"] != self.tbin:
            raise RuntimeError(
                f"cache at {self.cache_path} was built for "
                f"n={meta['n']} T={meta['T']} tbin={meta['tbin']}, "
                f"but this dataset wants n={len(self.files)} T={self.T} tbin={self.tbin}. "
                "Pass rebuild=True."
            )

    # -- split -----------------------------------------------------------------
    def _split_indices(self, train_fraction: float, seed: int) -> np.ndarray:
        if self.split == "all":
            return np.arange(len(self.files))
        rng = np.random.default_rng(seed)
        train, test = [], []
        for c in range(len(CLASSES)):          # stratify: same ratio within each class
            idx = np.flatnonzero(self.labels_all == c)
            rng.shuffle(idx)
            cut = int(round(len(idx) * train_fraction))
            train.append(idx[:cut])
            test.append(idx[cut:])
        chosen = np.concatenate(train if self.split == "train" else test)
        chosen.sort()
        return chosen

    # -- Dataset ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        cube = torch.from_numpy(np.asarray(self.data[j])).float()
        return cube, int(self.labels_all[j])

    @property
    def num_classes(self) -> int:
        return len(CLASSES)
