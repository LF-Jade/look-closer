# Look Closer: Patch-wise Supervision for AI-Generated Image Detection

<!-- TODO(arXiv): replace the placeholder id XXXX.XXXXX with the real arXiv id once the preprint is posted. -->
<p align="center">
<a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red" alt="Paper on arXiv"></a>
</p>

Reference implementation of the **multi-patch** pipeline studied in the paper
*Look Closer: Patch-wise Supervision for AI-Generated Image Detection*.

![Method overview: patch sampling, per-patch supervision, inference-time averaging](assets/method_overview.png)

A shared backbone classifies explicit RGB crops. **Every crop receives its own
classification loss** — there is no within-image pooling before the loss — and the
crop probabilities are averaged only at inference, into a single image score. No
handcrafted residual filter, no learned image-level fusion module, and no
truncated backbone are involved.

> **Verification.** `scripts/smoke_test.py` runs in CI on CPU, for ResNet-18 and
> Swin-T. It checks the pipeline's *behaviour* — sampling geometry, collation, both
> supervision modes, checkpoint round-trips, metric conventions, the schedule
> sequence, selection and gating, and error handling — not detection accuracy. See
> [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
>
> [![smoke test](https://github.com/LF-Jade/look-closer/actions/workflows/smoke.yml/badge.svg)](https://github.com/LF-Jade/look-closer/actions/workflows/smoke.yml)

## Contents

- [Install](#install)
- [Quick start (CPU, no benchmark, no downloads)](#quick-start-cpu-no-benchmark-no-downloads)
- [Train](#train)
- [Evaluate](#evaluate)
- [Input handling](#input-handling)
- [Data](#data)
- [Reported results](#reported-results)
- [Layout](#layout)
- [What is not included](#what-is-not-included)
- [Paper and citation](#paper-and-citation)
- [License](#license)

## Install

Run all commands from the repository root.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Quick start (CPU, no benchmark, no downloads)

```sh
python scripts/make_synthetic_data.py --out data/synthetic --count 16
python scripts/train.py --config configs/genimage_resnet50.yaml \
    --device cpu \
    --set data.train_manifest=data/synthetic/train.txt \
    --set model.backbone=resnet18 --set model.pretrained=false \
    --set training.epochs=1 --set training.batch_size=4 --set training.num_workers=0
python scripts/evaluate.py --config configs/eval_genimage.yaml \
    --device cpu --set eval.num_workers=0 \
    --set model.backbone=resnet18 \
    --checkpoint runs/genimage_resnet50/final.pth \
    --manifests data/synthetic/eval.txt
```

The synthetic images carry no detection signal, so any metric from them is
meaningless as a performance number — they exist only to exercise the code.

Self-check, which asserts the pipeline's behaviour rather than its accuracy:

```sh
python scripts/smoke_test.py
python scripts/smoke_test.py --backbone swin_tiny_patch4_window7_224
```

## Train

```sh
# the release's ResNet-50 configuration
python scripts/train.py --config configs/genimage_resnet50.yaml

# the Swin-T configuration
python scripts/train.py --config configs/genimage_swin_tiny.yaml

# any config value can be overridden
python scripts/train.py --config configs/genimage_resnet50.yaml \
    --set training.epochs=2 --set training.batch_size=8 --limit 64
```

The resolved configuration is written to `<output_dir>/resolved_config.yaml`.

Checkpoints are written every `training.save_interval` **steps** as
`checkpoint<N>.pth`, where `N` is a save counter; the real epoch and step are
stored inside the file. (The research code named these `epoch<N>.pth`, so a
30-epoch run with `save_interval: 2000` produced 151 files, and the historical
logs' "Epoch 151" line is the 151st save. `scan_checkpoints` still reads that
legacy name.)

## Evaluate

```sh
python scripts/evaluate.py --config configs/eval_genimage.yaml \
    --checkpoint runs/genimage_resnet50/checkpoint151.pth
```

`--scan <exp_dir>` selects by mean accuracy over the manifests you pass, following
the retained evaluator's selection approach; the single-manifest gate behaviour
differs as documented in [SCOPE.md](docs/SCOPE.md). **Pass
`--select-on <held-out manifest>` for any new work** — without it, selection
happens on the reported target sets and the scores are optimistic.

Both `eval.decision_threshold` and `eval.gate_threshold` are 0–1 ratios; the
historical logs printed the gate as a percentage, so `99.5` becomes `0.995`.
`gate_threshold: 0` disables gating. If every candidate fails the gate the command
exits non-zero.

## Input handling

Patches are **never upscaled** by this release, and neither setting of
`pre_crop_resize` restores a historical pipeline:

| `pre_crop_resize` | What it does | What it is not |
|---|---|---|
| **`null`** (default) | Crops at the source resolution into `sampling.patch_size` patches. This is the release's own default, and it matches the manuscript's description of the default multi-patch setting. | A reproduction of any historical run's transform pipeline. |
| `299` | Resizes the whole source image to 299×299 *before* cropping — the resize-before-crop path retained in the original sampler. | A reproduction of the manuscript's resolution-aligned control, which *also* enlarges each extracted patch to the backbone input size. That is not supported here: `model.input_size` is bound to `sampling.patch_size`. |

Four different things are involved and must not be conflated: what the manuscript
records, what the retained research snapshot executes, what this release chooses as
its default, and what has actually been run and verified here. See
[docs/SCOPE.md](docs/SCOPE.md#four-settings-that-must-not-be-conflated).

## Data

One image per line: `<image path><whitespace><0|1>`, resolved relative to the
working directory (absolute paths are also accepted). No datasets, manifests or
checkpoints are shipped; see [docs/DATA.md](docs/DATA.md) for the format and for
which public benchmarks the paper uses.

## Reported results

![Reported GenImage means across four backbones](assets/genimage_means.png)

Reported GenImage mean accuracies for the whole-image baselines and for PWS across
four backbones. These are the paper's reported numbers, reproduced here for
orientation only — they were not produced by this repository.

Part of the selection behind these rows used the evaluation sets they are reported
on, and the remaining selection histories are not fully recovered; the paper marks
the documented cases `T` and leaves the rest unqualified. Read the differences as
an account of what was run, not as the direction or magnitude of gains under a
common evaluation protocol — the paper's Section 5.1 states the full limitation.

## Layout

```
look_closer/          # the method
  sampler.py          # patch extraction (pre-crop resize, grid, capped subset)
  data.py             # manifest parsing + patch collation
  models.py           # shared backbone + shared head
  losses.py           # per-patch focal loss; image-level control loss
  engine.py           # shared train loop, image-level inference, checkpoint helpers
  utils.py            # seeding, config, optimiser, cosine schedule
configs/              # ResNet-50 and Swin-T recipes, evaluation config
scripts/              # train / evaluate / smoke_test / make_synthetic_data
baselines/            # image-level control (same loop, loss after logit averaging)
tools/                # manifest subsampling utility
docs/                 # data format, environment, scope and limitations
assets/               # figures used by this README
.github/workflows/    # CPU smoke test
CITATION.cff          # machine-readable citation
```

## What is not included

* **Detector checkpoints** — train your own with `scripts/train.py`.
* **The single-patch selection study (SPD)** — this repository covers the
  multi-patch pipeline only.
* **Datasets** — obtain them from their official sources; see
  [docs/DATA.md](docs/DATA.md).

For how the released configuration relates to the reported runs, and what the
historical checkpoint-selection records mean for the reported numbers, see
[docs/SCOPE.md](docs/SCOPE.md).

## Paper and citation

**Look Closer: Patch-wise Supervision for AI-Generated Image Detection**
Zhida Zhang, Tao Wu, Siyu Liu, Jie Cao

<!-- Update this line with the arXiv URL once the preprint is posted. -->
The preprint is being prepared for arXiv; the link will be added here when it is
posted. Until then, please cite this repository:

```bibtex
@misc{lookcloser2026,
  title        = {Look Closer: Patch-wise Supervision for AI-Generated Image Detection},
  author       = {Zhang, Zhida and Wu, Tao and Liu, Siyu and Cao, Jie},
  year         = {2026},
  note         = {Preprint},
  howpublished = {\url{https://github.com/LF-Jade/look-closer}}
}
```

## License

[MIT](LICENSE).
