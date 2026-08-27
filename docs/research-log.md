# SpikeTransDVG — research log

Fully spiking language-driven object grounding on Talk2Event. Every component comes from a
frozen fork under `repositories/`; nothing in those repos is edited.

**Status as of 2026-08-24.** Stage-1 detection pretraining is complete and the augmented
backbone is the best artefact produced so far. Stage-2 grounding on that backbone has not
started and is blocked on one design decision (see [Open decisions](#open-decisions)).

---

## 1. Results at a glance

Detection, full 1134-frame Talk2Event test split (3127 boxes):

| run | config | mAP@0.5 | mAP@0.75 | wall clock |
|---|---|---|---|---|
| `det_t5` | 40 ep, no augmentation | 0.4633 | 0.3142 | 133 min |
| **`det_aug`** | **80 ep, augmentation** | **0.5307** | **0.3292** | **274 min** |

Best sampled-eval epochs: `det_t5` 0.4856 / 0.3502 (ep 37–38); `det_aug` 0.5569 / 0.3839
(both ep 67).

Grounding, full 7665-sample test split:

| run | mIoU | Acc@0.25 | Acc@0.5 | Acc@0.75 | Acc@0.9 |
|---|---|---|---|---|---|
| `grounding_t5` | 0.2225 | 36.5% | 19.8% | 5.3% | 0.26% |

Published reference (Talk2Event / EventRefer, arXiv 2507.17664, Table 2, **event-only**):
EventRefer mAcc 31.96% / mIoU 76.46%; best baseline EvRT-DETR 29.34% / 75.66%. Our
grounding mIoU of 0.2225 is far below that — see [finding 3](#3-the-grounding-model-never-used-the-caption).

---

## 2. Architecture

```
events (T=5, B, 2, 480, 640)                    caption (B, L)
        |                                              |
  EventVisionEncoder                          TextEmbedder
  Meta-SpikeFormer (SDTv2), ImageNet init      HuggingFace encoder + projection
        |  taps s4 / s8 / s16b                        |  (B, L, 256) tokens
        +---------------> CrossModalFusion <----------+
                          CMSF spiking cross-attention, per scale
                                   |
                             SpikingPAN neck
                             SpikeYOLO MS_ConvBlock / MS_StandardConv
                                   |
                             SingleBoxHead
                             spiking MLP -> 4 numbers (+ FiLM on the sentence vector)
                                   |
                          ONE box, (B, 4) normalised cxcywh
```

Detection pretraining (stage 1) replaces the left branch entirely with SpikeYOLO:

```
events -> DetectionBackbone (SpikeYOLO, snn_yolov8 topology)  taps s4/s8/s16/s32
       -> PAN neck -> SpikeDetect (anchor-free, DFL reg_max=16)  ->  8-class boxes
```

### Source modules and where they come from

| component | fork | loader |
|---|---|---|
| Meta-SpikeFormer vision encoder | `Spike-Driven-Transformer-V2` | `forks.load_metaspikformer()` |
| CMSF spiking cross-attention | `CMSF` | `forks.load_cmsf()` |
| SpikeYOLO blocks, `SpikeDetect`, `TaskAlignedAssigner`, `bbox_iou` | `SpikeYOLO` | `forks.load_spikeyolo()` |
| E-3DSNN sparse-3D backbone (explored, not used) | `E-3DSNN` | `forks.load_e3dsnn()` |

Two loader details worth knowing:

* SpikeYOLO's head imports `ultralytics.utils.tal`, and importing that package for real
  runs the fork's `__init__` (builds YOLO/SAM machinery, writes `~/.config`). The import is
  stubbed and `make_anchors` / `dist2bbox` / `bbox2dist` / `TaskAlignedAssigner` are loaded
  from `tal.py` in isolation and rebound on the module object.
* `bbox_iou` lives in `metrics.py`, which pulls matplotlib and `ultralytics.utils.LOGGER`.
  Its body uses only `torch` and `math` (verified), so it is extracted by text and executed
  in an isolated namespace.

### Own code

| file | role |
|---|---|
| `models/event_encoder.py` | Meta-SpikeFormer wrapper, multi-scale taps |
| `models/text_embedder.py` | HuggingFace encoder + projection to fusion width |
| `models/fusion.py` | CMSF cross-attention adapted to dense feature maps |
| `models/grounding_model.py` | `SpikingPAN`, `SingleBoxHead`, `SpikeTransDVG` |
| `models/grounding_loss.py` | `SingleBoxLoss` (L1 + CIoU) |
| `models/spikeyolo_detector.py` | `DetectionBackbone`, `SpikeYOLODetector` |
| `models/detection_loss.py` | YOLOv8 BCE + CIoU + DFL with the real TAL assigner |
| `datasets/talk2event_frames.py` | Talk2Event re-keyed to frame-level detection |
| `datasets/event_augment.py` | mosaic / hflip / scale+translate / event dropout |
| `tools/train_grounding.py` | stage-2 grounding trainer |
| `tools/train_spikeyolo_det.py` | stage-1 detection trainer |

---

## 3. Dataset facts

`Talk2EventDataset` yields one sample per **(object, caption)** pair — 23,025 train /
7,665 test. `talk2event_frames.frame_index()` re-keys the same annotations by
`event_path`, pooling each item's `bbox` with its `others` list:

| split | frames | boxes | boxes/frame | captions/frame |
|---|---|---|---|---|
| train | 4,433 | 10,321 | 2.33 | 5.19 (median 3, max 21) |
| test | 1,134 | 3,127 | 2.76 | — |

### The split is by driving sequence, with zero overlap

* **47 train sequences, 13 test sequences, 0 shared.** Test is a domain shift to unseen
  streets (`thun_02_a`, `interlaken_01_a`, `zurich_city_12–15`), not held-out frames of
  seen scenes.
* Annotated frames within a sequence are spaced a **constant 8 apart** (median = min = 8).
  The nominal 4,433 training frames are really **47 distinct scenes** sampled densely.
* `thun_02_a` alone is 457 frames — 40% of the test set from one drive.

### Class priors differ sharply between splits

| class | train | test |
|---|---|---|
| car | 82.8% | 64.1% |
| pedestrian | 4.2% | 13.6% |
| rider | 0.9% | 7.1% |
| bicycle | 2.3% | 8.0% |
| motorcycle | 1.8% | 0.5% |

mAP averages over classes, so `rider` (≈93 training instances) is weighted equally with
`car`. A meaningful slice of the train→test drop is this mismatch, independent of
overfitting. Box sizes are comparable (median 0.0107 vs 0.0135 of frame).

---

## 4. Findings

### 1. CMSF's cross-attention is dead at PyTorch's default init

`SpikingCrossAttention` computes `(q @ (kᵀv)) * 0.125` then thresholds at 0.5, so the raw
spike-count product must reach **4** before one output spike appears. With q/k/v
BatchNorms at gain 1.0 those neurons fire under 1%, the branch returns exactly zero, and
`SCA_Block`'s `x + attn(x, y, y)` collapses to `x` — the model ignores the caption with no
error and a healthy-looking loss.

Measured (Talk2Event captions, BN calibrated, 160×224, T=5):

| `attn_bn_gain` | k firing | attention firing | caption sensitivity |
|---|---|---|---|
| 1.0 | 0.6% | **0.0%** | **0.0000** — text-blind |
| 3.0 | 2.6% | 26.3% | 1.0816 |
| 6.0 | 4.0% | 34.6% | 1.0798 |
| 10.0 | 7.1% | 39.8% | 1.0221 |

Default is **3.0** (threshold-dependent BN, tdBN — Zheng et al. AAAI 2021).

### 2. BatchNorm at batch size 1 destroys a regression head

At batch 1 a BatchNorm's "batch" is the T=5 timesteps of *one* sample, so it subtracts
that sample's own channel means — deleting the per-sample signal a regressor needs — then
eval swaps in pooled running stats and scrambles the mapping. Fitting 8 samples on frozen
features:

| head normalisers | train loss | eval mean IoU | predictions |
|---|---|---|---|
| BatchNorm1d/2d | 3.25 → 0.54 | **0.000** | uncorrelated (gt cx 0.255 → pred 0.68) |
| GroupNorm + LayerNorm | 3.26 → 0.31 | **0.71–0.84** | gt 0.516 → pred 0.506 |

Falling loss with collapsing IoU is what makes it dangerous. `SingleBoxHead` uses
GroupNorm/LayerNorm exclusively.

Related: an **uncalibrated** BatchNorm makes the whole eval path return an all-zero
pyramid. `SpikeTransDVG.calibrate_bn()` exists for that and must run before a fresh
model's first eval.

### 3. The grounding model never used the caption

The finished `grounding_t5` run was evaluated with **deliberately wrong captions**:

| | mIoU | Acc@0.25 | Acc@0.5 |
|---|---|---|---|
| correct captions | 0.2195 | 35.5% | 18.3% |
| **wrong captions** | **0.2185** | 36.6% | 18.6% |

A wrong caption costs **0.001 mIoU**. The language branch contributed nothing; the model
predicted a plausible salient box from events alone. It does clear the trivial floor
(constant mean box = 0.0609 mIoU), so the vision branch worked — it was doing detection,
not grounding.

Oracle swap localises the error to **centre placement**:

| | mIoU |
|---|---|
| model as-is | 0.2195 |
| predicted centre + **true size** | 0.2407 |
| **true centre** + predicted size | **0.4814** |

Median centre error 0.11 of frame (~61 px) = 0.83× the size of the object being located.
Size is nearly solved (predicted median 0.1436 vs GT 0.1242).

Train/test mAP by IoU threshold on the same checkpoint:

| IoU | train | test | test/train |
|---|---|---|---|
| 0.50 | 0.8305 | 0.3478 | 0.42 |
| 0.75 | 0.7610 | 0.2013 | 0.26 |
| 0.90 | 0.1809 | 0.0315 | 0.17 |

**A FiLM head conditioned on the sentence vector did not fix it** (mean IoU 0.064 vs
baseline 0.074 on a same-frame overfit set). The failure is upstream of the head.

### 4. The grounding model cannot overfit 16 samples; the detector can

Same-frame overfit set (4 frames × 4 captions, each caption a different object — fittable
only by reading the caption):

| | train-mode IoU | eval-mode IoU |
|---|---|---|
| grounding model, 200 steps | 0.025–0.137 | 0.026–0.108 |

Loss fell 3.42 → 0.36 while IoU stayed ~0.07 in **both** modes, so this is not the
BatchNorm train/eval gap — the model genuinely fails to fit.

The detector, by contrast, saturates 16 frames in 90 seconds:

| step | mAP@0.5 | mAP@0.75 | mAP@0.9 |
|---|---|---|---|
| 150 | 0.650 | 0.073 | 0.005 |
| **300** | **1.000** | **0.957** | 0.366 |

So architecture, loss, TAL assigner, DFL decoding and mAP evaluation are all correct. The
entire detection train→test gap is generalisation, not capacity.

### 5. mAP@0.9 is unreachable at 480×640

Even with the answers memorised, mAP@0.9 plateaus at ~0.37 on train and 0.03 on test.
IoU 0.9 on a median ~62×66 px box permits ~2 px error per edge, and DFL quantises to
feature cells. This is a resolution ceiling, not a training failure.

### 6. Augmentation delays memorisation rather than preventing it, but still wins

| epoch | AUG test | AUG train | gap | | NO-AUG test | NO-AUG train | gap |
|---|---|---|---|---|---|---|---|
| 0 | 0.0728 | 0.0769 | +0.004 | | 0.0935 | 0.1608 | +0.067 |
| 10 | 0.3631 | 0.7511 | +0.388 | | 0.3475 | 0.8873 | +0.540 |
| 23 | 0.4522 | 0.9052 | +0.453 | | — | — | — |
| 41 | 0.4637 | 0.9755 | +0.512 | | — | — | — |
| 67 | **0.5569** | 0.9925 | +0.436 | | — | — | — |

Train mAP crossed 0.90 at epoch 23 with augmentation versus epoch 9 without, and both end
near 0.99. Augmentation bought ~14 epochs of delay, not a lower plateau — expected, given
only 47 distinct scenes. It still improved final test mAP@0.5 by 14.5%.

**Most gains land in the final LR anneal.** `det_aug`'s mAP@0.75 went 0.244 (ep 24) →
0.3839 (ep 67). Twice during that run the evidence looked like mosaic was permanently
harming tight localisation and the run should be abandoned; both readings were wrong.
Judge these runs at the end of the cosine, not mid-schedule.

---

## 5. Experiment record

### `grounding_t5` — end-to-end grounding

```
uv run python tools/train_grounding.py --run-name grounding_t5 --epochs 2 --accum 8 \
  --warmup 300 --lr 3e-4 --backbone-lr 3e-5 --eval-every 250 --eval-samples 600
```

2 epochs, 5,756 optimiser steps, 4h 08m. Batch 1 (hard ceiling), bf16 autocast
(**required** — fp32 with a trainable backbone OOMs at batch 1 on 32 GiB, >31.3 GiB),
74.9M trainable / 124.1M frozen (the text encoder). Peaked on its final evals, i.e. the
schedule ran out before the model did. Result: mIoU 0.2225 — but see finding 3.

### `det_t5` — SpikeYOLO detection, no augmentation

```
uv run python tools/train_spikeyolo_det.py --run-name det_t5 --epochs 40 \
  --batch-size 4 --accum 2 --lr 1e-3 --warmup 300 --eval-frames 300
```

40 epochs, 133 min, 23.12M params (12.62M backbone), width 0.5, ~11 GiB at batch 4.
Loss monotone at all 40 epochs, 8.46 → ~1.5. **mAP@0.5 0.4633, mAP@0.75 0.3142.**

### `det_t5_long` — 300 epochs, no augmentation — ABANDONED

Killed at epoch 15. The 300-epoch cosine keeps LR near peak for hundreds of epochs, and
with a memorised training set that does not improve test performance; it fell behind the
40-epoch run by epoch 7. Retained only as evidence that schedule shape, not epoch count,
drives the result.

### `det_aug` — SpikeYOLO detection + augmentation

```
uv run python tools/train_spikeyolo_det.py --run-name det_aug --epochs 80 \
  --batch-size 4 --accum 2 --lr 1e-3 --warmup 300 --augment --patience 25 \
  --eval-frames 300 --train-eval-every 1 --train-eval-frames 150
```

80 epochs, 274 min. **mAP@0.5 0.5307, mAP@0.75 0.3292** on the full split; best sampled
0.5569 / 0.3839 at epoch 67. Still setting bests at epoch 67 of 80.

**Artefact: `runs/det_aug/backbone.pth`** — 50.8 MB, 12.6M backbone weights plus channel
spec `{s4:64, s8:128, s16:256, s32:512}`. This is what stage 2 consumes.

---

## 6. Augmentation

`datasets/event_augment.py`, train split only:

| transform | p | notes |
|---|---|---|
| mosaic | 0.5 | 4 frames on a 2H×2W canvas, random-cropped back to 480×640 |
| hflip | 0.5 | |
| scale + translate | 0.5 | scale ∈ [0.6, 1.5], shift ±10%, zero-padded/cropped |
| event dropout | 0.3 | zeroes 15% of active sites |

Two event-specific decisions:

* **Mosaic builds at 2× canvas and crops**, rather than tiling four half-size frames.
  Tiling shrinks every object 2×; the median box is already ~1.3% of the frame and IoU@0.9
  needs ~2 px accuracy per edge. Resolution is load-bearing here.
* **Area interpolation, not bilinear.** Event counts are extensive quantities — area
  conserves total count, bilinear smears sparse impulses and inflates apparent activity.

### Validation

A coordinate bug in augmentation is silent. The check is content-based: mean event density
inside boxes vs frame average.

| augmentation | density ratio | boxes/frame | kept | out-of-bounds |
|---|---|---|---|---|
| none (baseline) | 1.22 | 1.38 | 100% | 0 |
| hflip | **1.22** | 1.38 | 100% | 0 |
| scale/translate | 1.66 | 1.38 | 100% | 0 |
| event dropout | **1.22** | 1.38 | 100% | 0 |
| mosaic | 2.12 | 2.28 | 165% | 0 |
| full pipeline | 1.84 | 1.67 | 120% | 0 |

hflip and event dropout reproducing the baseline **exactly** is the correctness signal —
the ratio is mathematically invariant under both, so any coordinate error would show.

Trap avoided: the train-mAP overfit probe must use a **separate un-augmented** view of
train. Subsetting the augmented dataset turns the train/test gap into a measurement of the
augmentation instead.

---

## 7. Traps and gotchas

* **`pkill -f <pattern>` matches its own shell** when the pattern appears in the command
  line — it killed the invoking command twice here, once mid-heredoc. Kill by PID.
* **Zero-initialising a head's output layer cuts the graph.** With `fc2.weight = 0` the
  gradient reaching its input is `grad_out @ W = 0`, so the backbone, fusion and neck all
  receive exactly zero gradient. Use a small non-zero init (1e-3).
* **`sj_functional.reset_net(self)` recurses** if the module also defines `reset()` —
  the helper calls `.reset()` on every submodule including the caller. Reset neurons
  directly.
* **SpikeYOLO's `BNAndPadLayer` reads `running_mean`/`running_var` while the same BN is
  updating them**, so its train-mode output legitimately drifts between identical calls
  until the statistics converge. Eval mode is deterministic.
* **SpikeLM's `BertConfig` carries dropout 0.1** — train-mode comparisons of text features
  are nondeterministic. Freezing the encoder holds it in eval and removes this.
* **`SpikeDetect.forward` mutates the list it is given.** Pass a fresh `list(...)`.
* **Gradient checkpointing is unavailable** for these models: recomputation re-runs
  stateful LIF neurons whose membranes have already advanced, silently producing different
  activations.
* **Stale metric carry-over.** An early version of the train-mAP probe computed the
  train/test `gap` every epoch against a train score up to 4 epochs old. Recompute or emit
  NaN — never carry.

---

## 7b. SpikeLM removed (2026-08-24)

The SpikeLM text encoder was deleted: `models/text_encoder.py`, `forks.load_spikelm()`,
and `notebooks/spikelm_text_encoder.ipynb`. `models/text_embedder.py` replaces it with a
stock HuggingFace encoder plus a projection, and **the text encoder is no longer part of
the grounding model** -- `SpikeGroundingV2.forward` takes `text_tokens` of shape
(B, L, d_model) and the caller supplies them.

Why: SpikeLM was 124.3M parameters of roberta-base transplanted by name into a spiking
BERT (197/199 tensors). SpikeLM ships no weights, so nothing about it was spike-pretrained.
Two runs of the same architecture differing only in the freeze flags settled it:

| encoders | epochs | caption delta |
|---|---|---|
| frozen | 85 | +0.0009 |
| unfrozen | 2 | **+0.051** |

A caption-blind model at delta +0.0009 against mIoU 0.21 means a deliberately wrong caption
costs 0.4% of the prediction. The fusion and head were not the blocker; the encoders were.

Also refuted this session: the hypothesis that event cameras cannot see what the captions
describe. Over 4,929 same-frame caption pairs, **97.0%** differ by a position word, 58.6%
by a motion word, **99.4%** by some event-observable cue, and only **0.4%** by colour
alone. The discriminating information is in the events.

## 8. Open decisions

### Stage 2: which head?

The instruction was "train only the cross-attention", but the grounding head is randomly
initialised, so that is only literal if the head is pretrained too.

* **(A) Frozen `SpikeDetect` head — only cross-attention trains** (~2.3M params). Fusion
  learns to re-rank the detector's 6,300 anchors so the referred object scores highest;
  output is the top-scoring box. Inherits localisation from a backbone measured at
  mAP@0.75 = 0.33. Cost: internally multi-anchor, which walks back the earlier
  "predict only 1 bbox" requirement.
* **(B) Frozen backbone, trainable fusion + single-box MLP head.** Keeps one box by
  construction, but retains the coordinate regression that finding 3 identified as the
  failure.

**Recommendation: (A)**, more strongly now that the backbone is demonstrably good.

### Other open items

* `notebooks/spiketrandvg_architecture.ipynb` has stale executed outputs from the old
  25,200-anchor DFL grounding head; the builder script is updated but it needs re-running
  (~6 min, GPU free).
* `runs/det_t5_long` is a dead 15-epoch stub, safe to delete.
* **Test-set early stopping is model selection on test.** `best.pth` is chosen by test
  mAP throughout. For publication, hold out ~400 of the 4,433 train frames as validation,
  stop on that, and report test once.
* Class-balanced loss or sampling is untried and targets a distinct part of the gap
  (finding: `rider` is 0.9% of train boxes but weighted equally in mAP).
* Neither encoder is spike-pretrained — Meta-SpikeFormer is ImageNet, SpikeLM is
  roberta-base transplanted. Both are ANN-trained weights adapted to a spiking regime, not
  inheriting one. This belongs in any writeup.

---

## 9. Environment

* RTX 5090, 32 GiB (31.3 GiB usable), CUDA, PyTorch ≥ 2.13, Python 3.12
* `T=5` timesteps, 480×640 native resolution (no downsampling — see finding 5)
* bf16 autocast throughout
* Detection: batch 4 (~11–14 GiB). Grounding: batch 1 hard ceiling (17.1 GiB bf16; fp32
  OOMs)

---

# autoresearch run, 2026-08-27 (branch `autoresearch/20260827`)

Event-only grounding on Talk2Event, hill-climbing `val_acc75` by changing the spiking
vision encoder and the cross-attention module. Everything below is measured on this
branch unless it says otherwise.

## 10. Setup

### 10.1 There was no validation split, and there is now

`train.py` defaulted `--val-split` to `test` for `--task talk2event`, so early stopping
AND best-checkpoint selection both ran on the test set. The previously reported mIoU
0.2399 (`runs/t2e_100`) is a best-of-35 epochs selected on test, not a held-out number.

`talk2event_val_sequences.txt` carves 8 of the 47 train sequences out as `val`, BY
SEQUENCE, so no driving scene appears in both and val is a domain shift to unseen streets
exactly as test is. **6535 train / 1140 val / 2555 test**, verified 0 shared sequences and
0 shared frame paths. The rule that produced the list is in the file header; the file is
frozen, because regenerating it would silently change what every recorded number means.

The split behaves like test: the 8-epoch baseline reaches val mIoU 0.2359, against 0.2358
at epoch 5 of the old test-selected run.

### 10.2 Two leading indicators on the epoch line

`attn_perplexity` (effective number of vision positions a query attends, out of 6000
keys) and `pos_rms_ratio` (RMS(pos) / RMS(lateral output)), plus q/k firing rates and the
logit spread in `log.tsv`. Collected in eval only, one device sync per call.

## 11. The near-uniform attention map is arithmetic, not a training failure

In `SpatialCrossAttention`, q and k each pass Linear -> BN -> `Dynamic_Threshold_LIFNode`,
so they are **binary {0,1}** before the matmul, and the fusion runs at T=1. Per head,
`q.k` is a sum of `dh=32` Bernoulli terms.

Measured at init: q fires at 12.3%, k at 14.5%. So `q.k` has mean 0.57 and standard
deviation 0.75 counts, and after `self.scale = dh**-0.5 = 0.177` the logits spread by
**sigma = 0.13**. A softmax over N=6000 keys whose logits differ by 0.13 is the uniform
distribution: predicted perplexity `N*exp(-sigma^2)` = 5896, **measured 5980 of 6000**.

The consequence is not subtle. With a uniform map, `attn @ v` is the global average of all
6000 vision tokens, and the positional table is zero-mean, so **its average contributes
exactly nothing**. Position cannot reach `SlotBoxHead` at all. What the head sees is a
caption vector plus a global scene descriptor, which is precisely the observed signature:
mIoU 0.24 with caption_delta +0.15 (it does read the caption), and Acc@0.75 ~ 0 (there is
no localisation mechanism to speak of).

`dh**-0.5` is the right constant for ANALOG q/k -- unit-variance Gaussian entries make
`q.k` have variance dh. For binary q/k the derivation is off by more than an order of
magnitude.

## 12. Temperature buys a 14x sharper map and nothing else

`tools/temperature_response.py`, sweeping the scale on `probe_00/best.pth` over 570 val
samples, then training at the chosen value.

| scale | x default | logit sd | perplexity |
|---|---|---|---|
| 0.1768 | 1.0 | 0.53 | 4971.9 |
| 1.0 | 5.7 | 3.06 | 1852.9 |
| 4.0 | 22.6 | 12.01 | 681.0 |
| 16.0 | 90.5 | — | 548.1 |
| 32.0 | 181.0 | — | **548.1** |

**Perplexity saturates at 548 of 6000 and does not move again.** Binary `q.k` is an
integer overlap count, so at any large scale the softmax is a hard max over the ~548 keys
TIED at the maximum count. That floor belongs to the binary code, not to the temperature.

`probe_01` trained at scale 4.0 for 8 epochs. Perplexity 4967 -> 348, a 14x sharpening,
and every accuracy number unchanged within noise (last-3-epoch means: acc75 0.0032 vs
0.0038, acc50 0.1131 vs 0.1152, mIoU 0.2312 vs 0.2319, delta +0.1490 vs +0.1498).
Recorded as a **near-miss**: the leading indicator moved by an order of magnitude and the
metric did not, which localises the next question -- a sharper map is being thrown away
downstream.

## 13. Integer spikes break the tie floor

`mem_update` (SpikeYOLO's I-LIF) is a real LIF with a soft reset that quantises to
{0,1,2,3,4} instead of {0,1}. The event encoder already runs on it; the attention
projections were the last place still binary. Same checkpoint, same 127 val samples:

| q/k | scale | logit sd | perplexity |
|---|---|---|---|
| binary | 0.1768 | 0.53 | 4984.6 |
| binary | 4.0 | 12.01 | 679.2 (floor 548) |
| **I-LIF** | 0.1768 | 7.11 | 679.5 |
| **I-LIF** | 0.3536 | 14.14 | 333.3 |
| **I-LIF** | 1.0 | 39.58 | **123.3** |

At the *same* scale the map is 7x sharper, at matched logit spread it is 2x sharper, and
unlike the binary code it keeps sharpening past the floor. No analog path is added.

## 14. `val_acc75` is not yet a usable ranking metric, and this has to be said out loud

Acc@0.75 on this task is around 0.004, which on 1140 val samples is **4 to 5 samples**.
The standard error of a between-run difference at that base rate is 0.0030 -- 3.4 samples
-- so the protocol's 0.002 margin is **0.68 SE**, and picking the best of 8 epochs on it
is selecting the maximum of eight noisy draws. probe_01's best-epoch acc75 of 0.0070
against the baseline's 0.0044 looks like a win by the stated rule and is 0.9 SE.

Every comparison on this branch is therefore reported as a **mean over the last 3
epochs**, and until Acc@0.75 reaches a few percent the honest ranking signal is Acc@0.5
(SE 0.013), mIoU, and the two leading indicators. Recording this rather than hill-climbing
on it is the point.

## 15. The slot grid is not the constraint; 6.7 px is

Box statistics on the frozen splits (val, 1140 boxes): median width 67 px, median height
50 px. IoU 0.75 on a box of side s allows a centre error of about 0.143*s per axis, so the
**median centre tolerance is 6.7 px** and the 10th percentile is 4.6 px. At IoU 0.9 it is
2.5 px.

`--n-slots 1000` quantises to 0.32 px in x and 0.24 px in y, which caps IoU 0.75 only for
boxes below about 2.2 px -- **0.00% of train, val and test**. The slot head is not what is
limiting precision, and raising `--n-slots` would be wasted work. The stride-16 feature
grid, at 16 px per cell against a 6.7 px tolerance, is a real constraint on any head that
reads a cell index rather than a sub-cell expectation.

## 16. The model localises from viewer-relative geometry and almost nothing else

`tools/protected.py --what attributes`, `probe_00/best.pth`, 380 val samples. The caption
is reduced to ONE of Talk2Event's four annotated attribute groups, its phrases
comma-joined, and the model is scored on that alone.

| caption content | n | mIoU | Acc@0.5 | Acc@0.75 |
|---|---|---|---|---|
| full caption | 380 | 0.2411 | 0.1184 | 0.0026 |
| appearance only | 380 | 0.0501 | 0.0105 | 0.0000 |
| status only | 378 | 0.0532 | 0.0000 | 0.0000 |
| **relation_viewer only** | 378 | **0.1507** | **0.0608** | 0.0026 |
| relation_others only | 375 | 0.0731 | 0.0400 | 0.0027 |

A constant mean box scores ~0.06 on this data, so **appearance and status alone are at the
trivial floor** and `relation_viewer` -- "on the left side of the road", "in front of the
viewer" -- carries essentially all of the localisation. The model is not finding the
described object; it is decoding a coarse position out of the sentence.

Caveat, stated because it cuts the other way: a comma-joined phrase list is out of
distribution for a model trained on full sentences, so each row understates that
attribute's contribution in context. The RANKING is the finding, not the absolute values.

Two consequences. It is the table the four-sub-query design exists to produce and the
benchmark paper does not publish. And it makes Lane A load-bearing: the one cue that works
is language that names a position, which can only be matched against a positional signal
in the keys -- the signal measured at RMS ratio 0.019-0.035, below the 0.05 floor, for the
entire baseline run.
