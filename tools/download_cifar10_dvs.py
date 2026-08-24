"""Download, verify and extract CIFAR10-DVS into the workspace dataset/ directory.

CIFAR10-DVS (Li et al., 2017) is CIFAR-10 replayed on a monitor and recorded with a
DVS128 event camera: 10 classes x 1000 samples, 128x128 resolution, stored as .aedat
(AER format). Distributed by figshare as one zip per class; URLs and md5 checksums come
from spikingjelly's own dataset definition, so integrity is verifiable.

Layout produced:
    dataset/cifar10_dvs/
        download/   the 10 .zip archives (kept, so re-extraction needs no re-download)
        extract/    <class>/*.aedat

Re-runnable: an archive whose md5 already matches is not downloaded again, and a class
directory that already holds the expected file count is not re-extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

WS = Path(__file__).resolve().parents[2]
ROOT = WS / "dataset" / "cifar10_dvs"
DL, EX = ROOT / "download", ROOT / "extract"
EXPECTED_PER_CLASS = 1000


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, expect_md5: str) -> str:
    """Download `url` to `dest` unless it is already present and intact."""
    if dest.exists():
        if md5sum(dest) == expect_md5:
            return "cached"
        print(f"    md5 mismatch on existing file, re-downloading", flush=True)
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")
    t0, last = time.time(), [0.0]

    def hook(blocks, bsize, total):
        done = blocks * bsize
        now = time.time()
        if now - last[0] > 5 or done >= total > 0:
            last[0] = now
            pct = f"{100 * done / total:5.1f}%" if total > 0 else "  ?  "
            mb = done / 2**20
            rate = mb / max(now - t0, 1e-6)
            print(f"    {pct}  {mb:7.1f} MiB  {rate:5.1f} MiB/s", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    got = md5sum(tmp)
    if got != expect_md5:
        tmp.unlink()
        raise RuntimeError(f"md5 mismatch for {dest.name}: got {got}, want {expect_md5}")
    tmp.rename(dest)
    return "downloaded"


def extract(zip_path: Path, out_root: Path) -> tuple[str, int]:
    """Extract one class archive; skip when the class dir already looks complete."""
    cls = zip_path.stem
    target = out_root / cls
    if target.is_dir():
        n = len(list(target.rglob("*.aedat")))
        if n >= EXPECTED_PER_CLASS:
            return "cached", n
        shutil.rmtree(target)  # partial extraction -> start clean

    out_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_root)
    # archives may nest the class dir at any depth; normalise to extract/<class>/
    if not target.is_dir():
        found = next((p for p in out_root.rglob(cls) if p.is_dir()), None)
        if found and found != target:
            found.rename(target)
    return "extracted", len(list(target.rglob("*.aedat")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*", default=None, help="subset of class names")
    ap.add_argument("--keep-zips", action="store_true", default=True)
    args = ap.parse_args()

    DL.mkdir(parents=True, exist_ok=True)
    EX.mkdir(parents=True, exist_ok=True)

    resources = CIFAR10DVS.resource_url_md5()
    if args.classes:
        wanted = set(args.classes)
        resources = [r for r in resources if Path(r[0]).stem in wanted]
        if not resources:
            print(f"no classes matched {sorted(wanted)}", file=sys.stderr)
            return 1

    print(f"CIFAR10-DVS -> {ROOT}")
    print(f"{len(resources)} class archives\n")

    total_files, failures = 0, []
    for i, (name, url, md5) in enumerate(resources, 1):
        cls = Path(name).stem
        print(f"[{i}/{len(resources)}] {cls}")
        try:
            how = fetch(url, DL / name, md5)
            size = (DL / name).stat().st_size / 2**20
            print(f"    archive {how} ({size:.0f} MiB), md5 ok")
            how2, n = extract(DL / name, EX)
            print(f"    {how2}: {n} .aedat files")
            if n != EXPECTED_PER_CLASS:
                print(f"    WARNING expected {EXPECTED_PER_CLASS}, found {n}")
            total_files += n
        except Exception as e:  # keep going; report at the end
            print(f"    FAILED: {type(e).__name__}: {e}")
            failures.append((cls, repr(e)))

    print(f"\ntotal .aedat files: {total_files}")
    if failures:
        print(f"{len(failures)} class(es) failed:")
        for cls, err in failures:
            print(f"  {cls}: {err}")
        return 1
    print(f"download dir: {DL}\nextract dir : {EX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
