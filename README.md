# spiketrandvg

Hybrid spiking transformer for **referring-expression grounding**: given an image and a
sentence like *"the man in the red shirt on the left"*, predict the single bounding box
the sentence refers to.

Vision is a spiking neural network (SpiLiFormer); the language encoder is a conventional
ANN (roberta-base, frozen); they are joined by spike-driven cross-attention. Trained and
evaluated on RefCOCO / RefCOCO+ / RefCOCOg.

Every model component is loaded from a **frozen** reference repository under
`../repositories/`. Nothing in those repos is edited — `forks.py` executes individual
files under unique aliases and stubs the imports the used code paths never touch.

---

## Contents

- [Quick start](#quick-start)
- [Setup](#setup) · [1. Dependencies](#1-dependencies) · [2. Frozen repos](#2-frozen-reference-repositories) · [3. Pretrained weights](#3-pretrained-weights) · [4. Dataset](#4-dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Diagnostics](#diagnostics)
- [Architecture](#architecture)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Things that will bite you](#things-that-will-bite-you)

---

## Quick start

Assuming setup is done:

```bash
cd spiketrandvg

# 2-minute smoke test: 2 training steps, then the full eval path
uv run python -m spiketrandvg.train --run-name smoke \
    --limit-train 20 --epochs 1 --batch-size 5 --max-iters 2 --patience 0

# real training run (~21 min/epoch on an RTX 5090)
uv run python -m spiketrandvg.train --run-name myrun --epochs 60 --batch-size 16

# evaluate on a held-out test split
uv run python -m spiketrandvg.eval --run runs/myrun --split testA
```

---

## Setup

### 1. Dependencies

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). A CUDA GPU is required —
the spiking layers use CUDA kernels and there is no practical CPU path.

```bash
uv sync
```

Hardware this was developed on: **RTX 5090, 32 GiB**. Batch 16 at 384×384 peaks at
**17.4 GiB**, so a 24 GiB card works; below that, drop to `--batch-size 8` (9.4 GiB).

### 2. Frozen reference repositories

Three repos must exist under `../repositories/` (i.e. a sibling of `spiketrandvg/`):

| repo | supplies |
|---|---|
| `SpiLiFormer` | the spiking vision backbone |
| `CMSF` | `Dynamic_Threshold_LIFNode`, `Spiking_GFNN`, `RepeatTextEncoder` |
| `SpikeYOLO` | `mem_update` (the integer-LIF used in the box head) |

Check they resolve:

```bash
uv run python -c "from spiketrandvg import forks; print(forks.fork_status())"
# {'spiliformer': 'ok', 'cmsf': 'ok', 'spikeyolo': 'ok', ...}
```

`forks.py` also carries loaders for repos this pipeline no longer uses (`e3dsnn`,
`sdt2`, `sfod`, `talk2event`); those may report anything without affecting training.

Override the workspace root with `T2E_WS=/path/to/workspace` if your layout differs.

### 3. Pretrained weights

The vision backbone starts from SpiLiFormer's ImageNet checkpoint:

```
ckpts/spiliformer/checkpoint_spiliformer_T4_224.pth      # 830 MB, ImageNet 85.82% @ T=4
```

Training without it (`--rgb-ckpt ""`) is possible but throws away the only pretrained
visual prior available. roberta-base downloads automatically from HuggingFace on first
run.

### 4. Dataset

RefCOCO is **annotations only** — the images come from COCO. Three downloads, ~14 GB:

```bash
mkdir -p ../dataset/refcoco/{images,annotations}

# COCO train2014 images (12.6 GB) -- all of RefCOCO/+/g index into these
wget -c -O ../dataset/refcoco/images/train2014.zip \
    http://images.cocodataset.org/zips/train2014.zip
unzip -q ../dataset/refcoco/images/train2014.zip -d ../dataset/refcoco/images/

# MDETR pre-processed annotations (1.0 GB, Zenodo record 4729015)
wget -c -O ../dataset/refcoco/annotations/mdetr_annotations.tar.gz \
    https://zenodo.org/records/4729015/files/mdetr_annotations.tar.gz
tar xzf ../dataset/refcoco/annotations/mdetr_annotations.tar.gz \
    -C ../dataset/refcoco/annotations/
```

Expected layout (`dataset.py` finds this automatically; override with `REFCOCO_ROOT`):

```
dataset/refcoco/
├── images/train2014/                    82,783 .jpg
└── annotations/OpenSource/
    ├── finetune_refcoco_{train,val,testA,testB}.json
    ├── finetune_refcoco+_{train,val,testA,testB}.json
    └── finetune_refcocog_{train,val,test}.json
```

Verify:

```bash
uv run python -c "
from spiketrandvg.dataset import RefCOCO
ds = RefCOCO('refcoco','val'); print(len(ds), 'val samples'); print(ds[0][1])"
# 10834 val samples
# bowl behind the others can only see part
```

**Split sizes** (expressions, not images):

| dataset | train | val | testA | testB |
|---|---|---|---|---|
| refcoco | 120,624 | 10,834 | 5,657 | 5,095 |
| refcoco+ | 120,191 | 10,758 | 5,726 | 4,889 |
| refcocog | 80,512 | 4,896 | 9,602 (`test`) | — |

`refcoco+` bans absolute location words; `refcocog` has longer, more descriptive captions.

The original UNC pickles (`refs(unc).p`) are **not** needed — the MDETR JSON format is
what `dataset.py` reads.

---

## Training

```bash
uv run python -m spiketrandvg.train --run-name myrun --epochs 60 --batch-size 16
```

Everything trains except roberta-base: the SpiLiFormer backbone, its projection, the text
projection, cross-attention, and the box head — **66.19 M trainable of 195.7 M**.

Each epoch prints, and appends a row to `runs/<name>/log.tsv`:

```
EPOCH   6 | loss  1.728 (l1 0.0961 ciou 0.6240) train-IoU 0.420 (mode gap +0.0118)
          | mIoU 0.3870  Acc@0.25 0.6917  Acc@0.5 0.3475  Acc@0.75 0.0554  Acc@0.9 0.0037
          | blind 0.1385 delta +0.2486 | 146.1 min
```

### Key flags

| flag | default | notes |
|---|---|---|
| `--epochs` | 20 | 20 **underfits**: the cosine schedule hits zero at train-IoU 0.47. Use 60–100. |
| `--batch-size` | 16 | 17.4 GiB. `24` gives no speedup and triggers allocator OOM warnings. |
| `--lr` / `--encoder-lr` | 5e-4 / 1e-5 | From-scratch modules vs the pretrained backbone. A shared 5e-4 destroys the ImageNet prior before the 1.3 M fusion learns to use it. |
| `--patience` | 8 | Use **20+** for long runs — most gains land in the final LR anneal. |
| `--keep-all-best` | off | Archive a weights-only `best_ep<N>_miou<X>.pth` per new best (~748 MB each) instead of only overwriting `best.pth`. |
| `--limit-train N` | all | Overfit test on the first N samples. |
| `--max-iters N` | — | Smoke test: N steps, then straight through the full eval path. |
| `--resume <ckpt>` | — | Restores model, optimiser, epoch and step. |
| `--head-type` | `pooled_mlp` | **Leave it.** `attn_softargmax` collapses — see [Results](#results). |
| `--attn-type` | `spatial_softmax` | **Leave it.** `cmsf_linear` makes grounding impossible — see [Architecture](#architecture). |

### Outputs

```
runs/<name>/
├── args.json            every flag, for exact reproduction
├── log.tsv              one row per epoch (16 columns)
├── best.pth             full checkpoint (model + optimiser), highest val mIoU
├── last.pth             full checkpoint, most recent epoch
├── best_ep*.pth         only with --keep-all-best
└── final_metrics.json   full val split, best checkpoint
```

---

## Evaluation

`train.py` selects `best.pth` on `--val-split`, so reporting on that same split is model
selection on your own test set. `eval.py` exists to touch a test split exactly once,
after selection is done:

```bash
uv run python -m spiketrandvg.eval --run runs/myrun --split testA
```

```
refcoco/testA: 5657 of 5657 samples

=== results ===
  mIoU                   0.4075
  Acc@0.25               0.7115
  Acc@0.5                0.3891
  Acc@0.75               0.0852
  blind_mIoU             0.1674
  caption_delta          0.2385
```

Writes `runs/<name>/eval_testA.json`. Useful flags: `--split`, `--ckpt`, `--limit N`
(strided subset for a quick check), `--no-blind` (skips the control, halves runtime).

It warns if you point it at the split the checkpoint was selected on, and if the
checkpoint predates a code change and is loading parameters at random init.

### Metrics

- **mIoU** — mean Intersection-over-Union between predicted and true box.
- **Acc@X** — fraction of samples with IoU ≥ X. `Acc@0.5` is the standard bar in the
  literature; `Acc@0.75` measures precise localisation.
- **caption_delta** — `mIoU(correct caption) − mIoU(wrong caption)`. **The metric that
  matters most.** A model with delta ≈ 0 is a detector ignoring the text — a failure this
  project shipped for 85 epochs before catching.

The blind control's negative is a caption for a **different object in the same image**,
precomputed per sample. Do **not** replace this with a caption shuffle inside the batch:
56.7 % of the pairs that forms on RefCOCO name the *same* object, because each object has
~2.8 expressions stored consecutively. A paraphrase is not a negative — the same bug
elsewhere turned a true +0.22 into a reported +0.017.

---

## Diagnostics

```bash
uv run python tools/diagnose.py --run runs/myrun --ckpt best.pth
```

Reports four things, each tied to a failure mode already observed here:

1. **Firing rate per spiking layer** — <1 % means a LIF is dead and its residual branch is
   silently the identity. Healthy is roughly 5–35 %.
2. **Positional RMS ratio** — `RMS(pos) / RMS(lateral output)`. Below ~0.05, the
   positional embedding is thresholded away before it can inform the attention map.
3. **Attention perplexity** — effective number of positions each caption token attends.
   Near the key count (576) = uniform, carrying no location.
4. **Train mIoU, train mode vs eval mode** — a large gap means the eval path is broken and
   nothing else can be trusted.

---

## Architecture

```
  RGB (B,3,384,384)                       caption ids (B,L)
        │                                        │
  ┌─────▼──────────┐                     ┌───────▼────────┐
  │ VisionEncoder  │  SPIKING            │  TextEncoder   │  ANN
  │ SpiLiFormer    │  64.19M  trains     │  roberta-base  │  124.6M  FROZEN
  │ /16 → 24×24    │                     │  + 0.20M proj  │
  └─────┬──────────┘                     └───────┬────────┘
    576 × 256                                L × 256
        │                                        │
        └────────► SpatialBlock × 2 ◄────────────┘
              softmax cross-attention, 1.32M
              Q = TEXT,  K = V = VISION
                        │
              masked mean over caption tokens
                        │
              spiking MLP head, 0.14M
                        │
              ONE box (B,4) normalised cxcywh
```

**Why text queries vision.** The reverse direction produces 576 outputs needing to be
pooled back down. Querying with the caption makes the output caption-shaped — a handful of
tokens that have absorbed visual context — and makes softmax attention affordable.

**Why softmax, not CMSF's linear attention.** `cmsf_linear` forms `kᵀv`, summing over
*every* key position: the spatial index is marginalised away and no text query can ask
*where*. Grounding becomes impossible. This is why `--attn-type` should stay at
`spatial_softmax`.

**How spiking it is.** Measured on a trained checkpoint: q, k, v and the block output are
exactly binary `{0,1}`; the softmax weights in between are real-valued. So both large
matmuls are spike-driven, with one analog operation at the core — the honest price of
being able to localise. 100 LIF neurons in total across three variants
(`MultiStepLIFNode` ×88, CMSF's `Dynamic_Threshold_LIFNode` ×10, SpikeYOLO's integer
`mem_update` ×2), simulated over `T=4` timesteps.

---

## Results

**`refcoco_b1`** — full 120,624-sample train split, 20 epochs, `pooled_mlp`. Selected on
val, then evaluated **once** on the test splits:

| split | n | mIoU | Acc@0.25 | Acc@0.5 | Acc@0.75 | caption_delta |
|---|---|---|---|---|---|---|
| val | 10,834 | 0.4046 | 70.8 % | 38.2 % | 8.2 % | +0.2609 |
| **testA** | 5,657 | **0.4075** | 71.2 % | 38.9 % | 8.5 % | +0.2385 |
| **testB** | 5,095 | **0.3889** | 70.2 % | 33.5 % | 6.3 % | — |

Val and test agree closely, so the model generalises and the val numbers are not inflated
by selection. A caption_delta of +0.24 against same-image negatives means it genuinely
reads the expression.

**Head-type A/B** (100 samples, 200 epochs, one flag apart):

| head | mIoU | Acc@0.75 | caption_delta | outcome |
|---|---|---|---|---|
| `pooled_mlp` | 0.2513 | 0.52 % | +0.1141 | trains normally |
| `attn_softargmax` | 0.2009 | 0.05 % | +0.00005 | **collapsed** to a near-constant box |

`attn_softargmax` reads the box centre off the attention map instead of a binarised
vector. It failed because that map is statistically uniform — perplexity 575.8–576.0 out
of 576 keys, even after 199 epochs. The idea is untested rather than refuted; it becomes
testable once the map carries location.

### Known limitation

Accuracy collapses with the IoU threshold: 71 % at 0.25, 39 % at 0.5, 8.5 % at 0.75. The
model finds roughly the right region but rarely the precise box. `diagnose.py` traces this
to the near-uniform attention map, with the positional signal (RMS ratio 0.005–0.018) as
the leading suspect — position is a ~1 % perturbation on features that then pass a firing
threshold. **`Acc@0.75` is the metric to watch for any fix aimed at this.**

---

## Repository layout

```
spiketrandvg/
├── src/spiketrandvg/
│   ├── model.py           SpatialCrossAttention, SpatialBlock, RefCOCOGrounding, SingleBoxLoss
│   ├── vision_encoder.py  SpiLiFormer + lateral projection + positional embedding
│   ├── text_encoder.py    roberta-base + projection
│   ├── dataset.py         RefCOCO/+/g loader, augmentation, collate
│   ├── train.py           training entry point
│   ├── eval.py            evaluation entry point
│   └── forks.py           loads the frozen repos under ../repositories/
├── tools/diagnose.py      checkpoint diagnostics
├── docs/research-log.md   lab notebook: findings, experiment record, dead ends
├── runs/                  per-run args.json, log.tsv, checkpoints, metrics
└── ckpts/                 pretrained donor weights
```

`docs/research-log.md` documents the earlier event-camera (Talk2Event) phase of this
project. That code was removed when the repo was narrowed to RefCOCO; it is recoverable
from git history.

---

## Things that will bite you

**Never evaluate a bf16-trained model in fp32.** These spiking layers are not
dtype-agnostic — precision decides which membranes cross threshold. Measured on identical
weights: **bf16 0.656 IoU vs fp32 0.060**. An earlier conclusion that "this architecture
cannot fit 16 samples" was purely this mismatch. `predict()` and both entry points
autocast correctly; anything custom must too.

**LIF membranes persist between calls.** They are not stateless like ReLU. Every forward
pass must zero all 100 of them first — that is what `model.reset()` does, called at the
top of `forward`. Any new code path that skips it leaks one sample's charge into the next.

**Do not use `functional.reset_net(self)`.** It calls `.reset()` on every submodule
including the caller, and recurses until the stack blows. Reset neurons directly.

**Do not zero-initialise `fc2.weight`.** `grad_in = grad_out @ W = 0` starves the entire
fusion stack of gradient. It sits at `std=1e-3` for this reason; it cost a full optimiser
run to diagnose.

**Do not add gradient checkpointing.** Recomputation re-runs stateful LIF neurons whose
membranes have already advanced, silently producing different activations.

**`--attn-type cmsf_linear` cannot ground.** It marginalises the spatial index away. It
exists for comparison only.

**Horizontal flip must rewrite the caption.** 45.7 % of RefCOCO captions contain
"left"/"right". `dataset.py`'s `hflip_caption` swaps them (including `leftmost`,
`RIGHT-hand`, while leaving `leftover` alone). A naive flip mislabels almost half the
training set.

**`pgrep -f <pattern>` matches its own shell** when the pattern appears in the command
line. It has killed the invoking command here more than once. Kill by PID.
