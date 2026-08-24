# spiketrandvg

Spiking transformer for language-driven object grounding on event streams (Talk2Event).

Every model component comes from a frozen reference repository under `../repositories/`.
Nothing in those repos is edited — they are loaded through `utils/forks.py`, which executes
individual files under unique aliases and stubs the imports the used code paths never
touch.

## Where things are

| path | contents |
|---|---|
| [`docs/research-log.md`](docs/research-log.md) | **Start here.** Results, findings, experiment record, open decisions |
| `src/spiketrandvg/models/` | encoders, fusion, necks, heads, losses |
| `src/spiketrandvg/datasets/` | Talk2Event grounding + frame-level detection, augmentation |
| `src/spiketrandvg/notebooks/` | exploratory notebooks (see caveat below) |
| `tools/` | training entry points |
| `runs/` | completed runs: `args.json`, `log.tsv`, `best.pth`, `final_metrics.json` |
| `ckpts/` | pretrained donor weights (Meta-SpikeFormer, E-3DSNN) |

## Current state

Stage-1 detection pretraining is done; stage-2 grounding on that backbone has not started.

| run | what | headline |
|---|---|---|
| `det_aug` | SpikeYOLO detection + augmentation, 80 ep | **mAP@0.5 0.5307, mAP@0.75 0.3292** |
| `det_t5` | same, no augmentation, 40 ep | mAP@0.5 0.4633, mAP@0.75 0.3142 |
| `grounding_t5` | end-to-end grounding, 2 ep | mIoU 0.2225 — but measured **caption-blind** |

`runs/det_aug/backbone.pth` is the artefact stage 2 consumes.

## Running things

```bash
# stage 1 — detection pretraining (produces backbone.pth)
uv run python tools/train_spikeyolo_det.py --run-name det_aug --epochs 80 \
    --batch-size 4 --accum 2 --lr 1e-3 --augment --patience 25

# stage 2 — end-to-end grounding
uv run python tools/train_grounding.py --run-name grounding --epochs 2 --accum 8
```

Both accept `--max-iters` for smoke tests and `--resume <ckpt>`.

## Two things that will bite you

**Call `calibrate_bn()` before a fresh grounding model's first eval.** Untrained BatchNorm
running statistics make BatchNorm the identity, which leaves every spiking neuron below
threshold, which returns an all-zero pyramid and a caption-independent prediction — with no
error anywhere.

**Do not set `attn_bn_gain=1.0`.** At PyTorch's default gain, CMSF's cross-attention emits
exactly zero and the model silently ignores the caption. Measured; see finding 1 in the
research log.

## Caveat on notebooks

`notebooks/spiketrandvg_architecture.ipynb` still holds executed outputs from an earlier
DFL-based grounding head and does not reflect the current `SingleBoxHead`. The other
notebooks (`e3dsnn_event_bbox`, `spikelm_text_encoder`, `visualize_events`,
`train_cifar10_dvs`) are standalone investigations and remain accurate.
