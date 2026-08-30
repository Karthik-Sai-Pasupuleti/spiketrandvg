# spiketrandvg

**Spike-driven referring-expression grounding on event streams.** Given an event-camera
recording and a sentence like *"a dark-coloured car driving on the right side of the road"*,
predict the single bounding box that sentence refers to.

The vision pathway is a spiking neural network; the language encoder is a conventional ANN
(roberta-base, frozen); they meet in spike-driven cross-attention. Primary dataset is
**Talk2Event**; RefCOCO is supported as an RGB control.

Every model component is loaded from a **frozen** reference repository under
`../repositories/`. Nothing in those repos is edited — `utils.py` executes individual files
under unique aliases and stubs the imports the used code paths never touch.

> **Read `docs/research-log.md` before changing anything.** It records 46 runs, and several
> of the defaults below look wrong until you know which measurement produced them.

---

## Contents

- [Quick start](#quick-start) · [Setup](#setup) · [Architecture](#architecture)
- [Training](#training) · [Evaluation](#evaluation) · [Inference](#inference)
- [Diagnostics](#diagnostics) · [Results](#results) · [Repository layout](#repository-layout)
- [Things that will bite you](#things-that-will-bite-you)

---

## Quick start

```bash
cd spiketrandvg

# 3-minute smoke test: 2 steps, then the full eval path
uv run python -m spiketrandvg.train --task talk2event --run-name smoke \
    --limit-train 8 --epochs 1 --batch-size 2 --max-iters 2 \
    --eval-samples 8 --train-eval-samples 8 --patience 0

# the finalised configuration (~14 min/epoch on one RTX 5090)
uv run python -m spiketrandvg.train --task talk2event --run-name myrun \
    --val-split val --epochs 60 --patience 25 --batch-size 4 \
    --event-backbone metaspikformer \
    --event-ckpt ckpts/spikeformer_v2_weights/55M_kd.pth --keep-all-best

# report once, on the untouched test split
uv run python -m spiketrandvg.eval --run runs/myrun --split test
```

---

## Setup

### 1. Dependencies

Python 3.12 with [uv](https://docs.astral.sh/uv/). **A CUDA GPU is required** — the spiking
layers use CUDA kernels and SpikeLM's linear layers hard-code `.cuda()`.

```bash
uv sync
```

Developed on an **RTX 5090, 32 GiB**. The finalised config peaks at ~27 GiB at batch 4; the
small `spiliformer_dvs` backbone runs the whole model in ~3 GiB.

### 2. Frozen reference repositories

Expected under `../repositories/` (a sibling of `spiketrandvg/`):

| repo | supplies |
|---|---|
| `Spike-Driven-Transformer-V2` | Meta-SpikeFormer event backbone |
| `SpiLiFormer` | alternative backbones — ImageNet (RGB) and CIFAR10-DVS (event-native) |
| `SpikeYOLO` | `mem_update` — the integer I-LIF neuron |
| `CMSF` | `Dynamic_Threshold_LIFNode`, `Spiking_GFNN`, `RepeatTextEncoder` |
| `SpikeLM` | spiking BERT (experimental — see [Results](#results)) |
| `Talk2Event` | the official evaluation protocol, for reference |

```bash
uv run python -c "from spiketrandvg import utils; print(utils.fork_status())"
```

Override the workspace root with `T2E_WS=/path/to/workspace`.

### 3. Pretrained weights

```
ckpts/spikeformer_v2_weights/55M_kd.pth              # Meta-SpikeFormer, ImageNet
ckpts/spiliformer/checkpoint_spiliformer_T4_224.pth  # SpiLiFormer, ImageNet (RGB path)
```

The Meta-SpikeFormer checkpoint is **worth +0.11 mIoU and +17 points of Acc@0.5** over the
same backbone from scratch. Its 3-channel RGB stem is averaged onto the 2 event-polarity
channels automatically. roberta-base downloads from HuggingFace on first run.

### 4. Dataset

Talk2Event is expected at `../dataset/talk2event/`:

```
dataset/talk2event/
├── meta_data_v10/{train,test}/*.json     annotations
└── data/{train,test}/<sequence>/event/*.npz
```

```bash
uv run python -c "
from spiketrandvg.dataloader import Talk2Event
for s in ('train','val','test'): print(s, len(Talk2Event(s)))"
# train 6535 · val 1140 · test 2555
```

**Talk2Event ships train and test only.** Selecting checkpoints on `test` is model selection
on the test set, so `talk2event_val_sequences.txt` carves 8 of the 47 train sequences out as
`val`, **by sequence**, so no driving scene appears on both sides. That file is **frozen** —
regenerating it silently changes what every recorded number means.

Each sample is one *(object, caption)* pair. The `.npz` holds 10 native time bins × 2
polarities, regrouped into **5 timesteps × 2 channels** — exact, mass-preserving.

---

## Architecture

```
  events (5, B, 2, 480, 640)                caption ids (B, L)
            │                                       │
            │                            roberta-base, FROZEN (ANN)
            │                                       │
            │                            AttributeQueryTagger
            │                          → 4 sub-queries: appearance,
            │                            status, viewer, others
            │                                       │
            │  ◄──── ThresholdModulator: per-channel firing thresholds
            ▼
   Meta-SpikeFormer · 5 real timesteps · I-LIF integer spikes (0–4)
            │                    taps: s8 → 60×80×256, s16 → 30×40×512
            │
  ══════ ACCUMULATOR — sum spikes over T ══════
            │            ↑ binary ends here, real numbers resume
   lateral 1×1 + learnable 2D position → ~6000 vision tokens
            │
   softmax cross-attention (Q = the 4 sub-queries, K/V = vision)
            │   the map is SUPERVISED toward the true box
            │   and ROUTED into the head as a coordinate prior
            │
   SlotBoxHead: each coordinate a 1000-way choice over the
   480×640 annotation frame + bounded residual
            │
        ONE box (B, 4) normalised cxcywh
```

**Neurons.** 81 integer I-LIF (`mem_update`, emits 0–4, integrates across T with a soft
reset) and 8 binary. The event encoder is entirely I-LIF; the cross-attention has I-LIF on
q/k/v/proj and binary in its gated MLP; the softmax between them is necessarily analog. The
text encoder and box head contain no spiking neurons.

**The accumulator is the hinge.** Above it everything is binary or small-integer and pays
the precision cost; below it everything is an ordinary real-valued problem.

**No RGB branch, deliberately.** Talk2Event's own baselines are frame-only 55.47 vs
event-only 31.96 mAcc, so fusing would bury the contribution under a frame encoder. When
fusion is added for completeness the RGB branch should be a conventional ANN —
`VisionEncoder` is kept for that.

---

## Training

```bash
uv run python -m spiketrandvg.train --task talk2event --run-name myrun \
    --val-split val --epochs 60 --patience 25 --batch-size 4 \
    --event-backbone metaspikformer \
    --event-ckpt ckpts/spikeformer_v2_weights/55M_kd.pth --keep-all-best
```

`--task refcoco` (the default) runs the RGB control instead, with its own flags.

Each epoch prints and appends a row to `runs/<name>/log.tsv`:

```
EPOCH  18 | loss  3.589 (box 0.921 slot 2.310 tag 0.358) map 0.412 mass 0.691
          train-IoU 0.729 (mode gap +0.0147)
        | mIoU 0.4897  Acc@0.25 0.7316  Acc@0.5 0.5877  Acc@0.75 0.2263
        | blind 0.1139 delta +0.3473 | perplex 9.9/6000 box_mass 66.6x
```

### Key flags

| flag | default | notes |
|---|---|---|
| `--event-backbone` | `spiliformer_dvs` | **Use `metaspikformer`** — 54.7M with an ImageNet checkpoint, vs 1.7M event-native with none. Worth +0.11 mIoU. |
| `--event-ckpt` | — | The single biggest lever in the whole study. |
| `--epochs` / `--patience` | 20 / 8 | Converges by ~epoch 20 on Talk2Event. 60 with patience 25 is safe; 200 early-stopped at 44. |
| `--map-weight` | 1.0 | Supervises **where** the attention map points. Do not set to 0. |
| `--attn-prior` | on | **Routes** the map into the coordinate head. Do not disable. |
| `--pos-ratio` | 0.5 | Positional amplitude in the vision keys. |
| `--qk-lif` | `ilif` | Integer spikes on q/k. Binary gives ≤33 distinct logits over 6000 keys. |
| `--T` | 5 | T=10 costs 2× compute for −0.002 mIoU. |
| `--text-backbone` | `roberta` | `spikelm` reaches 98% spiking but **destroys grounding** — see Results. |
| `--no-condition` | off | Ablates language→threshold conditioning. |
| `--keep-all-best` | off | Archives a weights-only snapshot per new best (~780 MB each). |
| `--max-iters N` | — | Smoke test: N steps, then straight through the full eval. |
| `--resume <ckpt>` | — | Restores model, optimiser, epoch and LR schedule. |

### Outputs

```
runs/<name>/
├── args.json           every flag, for exact reproduction
├── log.tsv             one row per epoch
├── best.pth            full checkpoint (model + optimiser), highest val mIoU
├── last.pth            most recent epoch
└── final_metrics.json  full val split, best checkpoint
```

---

## Evaluation

`train.py` selects `best.pth` on `--val-split`, so reporting on that same split is model
selection on your own test set. `eval.py` exists to touch a test split exactly once:

```bash
uv run python -m spiketrandvg.eval --run runs/myrun --split test
```

```
talk2event/test: 2555 of 2555 samples

=== results ===
  mIoU                   0.4442
  Acc@0.25               0.6568
  Acc@0.5                0.5076
  Acc@0.75               0.2262
  blind_mIoU             0.0976
  caption_delta          0.3016
```

Writes `runs/<name>/eval_test.json`. The run records its own task, so `eval.py` infers
whether it is a Talk2Event or RefCOCO checkpoint. Useful flags: `--ckpt`, `--limit N`
(strided subset), `--no-blind` (skips the control, halves runtime).

### Metrics, and which one is official

- **mIoU** — mean IoU between predicted and true box.
- **Acc@X** — fraction of samples with IoU ≥ X.
- **caption_delta** — `mIoU(correct caption) − mIoU(wrong caption)`. **The metric that
  matters most.** Delta ≈ 0 means a detector ignoring the text, which this project shipped
  twice before catching it.

> **The official Talk2Event protocol evaluates at IoU ∈ [0.9, 0.95]**, not 0.5
> (`repositories/Talk2Event/test.py`, `T2E_Metric`). Published event-only numbers —
> EventRefer mIoU 76.46 / mAcc 31.96 — are on *that* scale. Comparing this model's Acc@0.5
> against them is not a like-for-like comparison; quote `Acc@0.9` when citing the benchmark.

The blind control's negative is a caption for a **different object in the same event
frame**, chosen by object identity and precomputed per sample. Do **not** replace it with a
batch shuffle: Talk2Event stores each object's paraphrases consecutively, so ~50% of the
pairs a shuffle forms describe the same box. That bug turned a true +0.22 delta into +0.017.

---

## Inference

There is no separate inference script — `model.predict()` is the whole path. This snippet is
tested end to end:

```python
import json, torch, numpy as np
from spiketrandvg.model import Talk2EventGrounding
from spiketrandvg.dataloader import talk2event_cube
from spiketrandvg.textencoder import build_tokenizer

dev, run = "cuda", "runs/final_200"
a    = json.load(open(f"{run}/args.json"))
blob = torch.load(f"{run}/best.pth", map_location=dev, weights_only=False)

# event_ckpt=None: the checkpoint overwrites every backbone weight anyway
model = Talk2EventGrounding(
    event_ckpt=None, event_backbone=a["event_backbone"], img_size=tuple(a["size"]),
    T=a["T"], depth=a["depth"], n_slots=a["n_slots"],
    ilif=not a.get("no_ilif", False),
    condition_encoder=not a.get("no_condition", False),
).to(dev)
model.load_state_dict(blob["model"], strict=False)
model.eval()

raw  = torch.from_numpy(np.load("path/to/000123.npz")["events"].astype(np.float32))
cube = talk2event_cube(raw, t_out=a["T"]).unsqueeze(1).to(dev)   # (T, 1, 2, 480, 640)

tk = build_tokenizer()
t  = tk(["the car on the right side of the road"], padding="longest",
        truncation=True, max_length=80, return_tensors="pt").to(dev)

box = model.predict(cube, t["input_ids"], t["attention_mask"])[0]   # (4,) cxcywh in [0,1]
H, W = a["size"]
cx, cy, w, h = box.tolist()
print(f"xyxy px: {int((cx-w/2)*W)}, {int((cy-h/2)*H)}, "
      f"{int((cx+w/2)*W)}, {int((cy+h/2)*H)}")
```

Three things that matter:

- **`unsqueeze(1)`** — the cube is `(T, B, C, H, W)`; the batch axis is second, not first.
- **Boxes come out normalised cxcywh in [0, 1]**, so multiply by the *annotation* frame
  (480×640), not the feature grid.
- **`predict()` autocasts to bf16** and calls `model.eval()` for you. Do not evaluate these
  layers in fp32 — see below.

---

## Diagnostics

```bash
uv run python -m spiketrandvg.utils --run runs/myrun --ckpt best.pth
```

Reports four things, each tied to a failure this project actually hit: per-layer **firing
rates** (<1% = dead, silently identity-passing its residual), **positional RMS ratio**
(<0.05 = position thresholded away), **attention perplexity** (near the key count = uniform
map carrying no location), and **train mIoU in train vs eval mode** (a large gap invalidates
every other number).

`tools/oracle.py` substitutes ground-truth centre or size to localise the remaining error.

---

## Results

Finalised configuration, `runs/final_200`, best epoch 18:

| | val (selection) | **test (held out)** |
|---|---|---|
| mIoU | 0.4897 | **0.4442** |
| Acc@0.25 | 73.2% | 65.7% |
| Acc@0.5 | 58.8% | 50.8% |
| Acc@0.75 | 22.6% | 22.6% |
| Acc@0.9 *(official scale)* | 2.3% | **1.9%** |
| caption_delta | +0.347 | **+0.302** |

Against the matched-budget baseline (pre-search architecture, same 60 epochs): **+45.8%
mIoU, 2.6× Acc@0.5, 6× Acc@0.75.** Roughly 75% of localisation is caption-attributable on
unseen scenes.

**Against the published benchmark, on its own metric, this is not competitive** — EventRefer
reports mIoU 76.46 / mAcc 31.96 event-only against this model's 44.42 / 1.9. See
`docs/research-log.md` and the study record for the full assessment.

### Two experiments whose negative results are load-bearing

- **`--text-backbone spikelm`** takes the model from 30% to 98% spiking, and **destroys the
  grounding**: caption_delta +0.325 → −0.006, mIoU 0.4651 → 0.2554. SpikeLM is a
  quantisation-aware *training framework* and ships no weights; transplanting roberta
  tensors leaves its 456 clip-value parameters uncalibrated, and the output becomes a random
  hash — measured, paraphrase pairs score *less* similar than unrelated ones.
- **`--no-condition`** costs only −0.007 to −0.018 mIoU. Language→threshold conditioning is
  the weakest kept component.

---

## Repository layout

```
spiketrandvg/
├── src/spiketrandvg/
│   ├── model.py           attention, heads, losses, both grounding models
│   ├── visionencoder.py   EventEncoder (SpikeFormer / SpiLiFormer-DVS) + VisionEncoder (RGB)
│   ├── textencoder.py     roberta + attribute tagger + SpikingTextEncoder
│   ├── dataloader.py      Talk2Event + RefCOCO + event voxel cubes
│   ├── train.py           training entry point (both tasks)
│   ├── eval.py            evaluation entry point
│   └── utils.py           fork loading + checkpoint diagnostics
├── tools/                 oracle.py, protected.py, temperature_response.py
├── docs/research-log.md   lab notebook — 46 runs, findings §1–25
├── runs/                  per-run args.json, log.tsv, checkpoints, metrics
└── ckpts/                 pretrained donor weights
```

---

## Things that will bite you

**Never evaluate a bf16-trained model in fp32.** These layers are not dtype-agnostic —
precision decides which membranes cross threshold. Measured on identical weights: **bf16
0.656 IoU vs fp32 0.060.** An earlier "this architecture cannot fit 16 samples" conclusion
was purely this mismatch.

**A dead spiking layer passes its residual straight through, silently.** At the default
BatchNorm gain two attention neurons fire at exactly 0.00%, collapsing `x + attn(x,y,y)` to
`x`. A 41-step fit reached IoU 0.996 that way — entirely through the language path, with the
event encoder inert at zero gradient. Check firing rates before trusting a loss curve.

**LIF membranes persist between calls.** `model.reset()` runs at the top of `forward`; any
new code path that skips it leaks one sample's charge into the next.

**`no_grad` around a frozen encoder severs language conditioning.** The gains enter *inside*
the encoder's forward, so wrapping it gives the modulator `grad is None` on every step while
it still counts as trainable.

**Do not use `functional.reset_net(self)`.** It calls `.reset()` on every submodule
including the caller and recurses until the stack blows.

**Do not zero-initialise `fc2.weight`.** `grad_in = grad_out @ W = 0` starves the whole
fusion stack.

**Gradient checkpointing is unavailable.** Recomputation re-runs stateful LIF neurons whose
membranes have already advanced, silently producing different activations.

**`pgrep -f <pattern>` matches its own shell** when the pattern is in the command line. It
has killed the invoking command here more than once, and produced a false "training stopped"
reading. Kill by PID.
