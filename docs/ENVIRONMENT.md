# Environment

## Verified environment

The checks in `scripts/smoke_test.py` were first run in exactly this environment:

| Component | Version |
|---|---|
| OS | macOS 26.6.2 (arm64) |
| Python | 3.9.6 |
| torch | 2.8.0 |
| torchvision | 0.23.0 |
| timm | 1.0.30 |
| numpy | 2.0.2 |
| Pillow | 11.3.0 |
| PyYAML | 6.0.3 |
| Device | CPU |

Reproduce it with:

```sh
python -m venv .venv && . .venv/bin/activate
pip install "torch==2.8.0" "torchvision==0.23.0" "timm==1.0.30" \
            "numpy==2.0.2" "Pillow==11.3.0" "PyYAML==6.0.3"
python scripts/smoke_test.py
```

That local run predates the change of the `pre_crop_resize` default from `299` to
`null`, so it is not a statement about the current revision; see below for what
covers the current revision.

## What `requirements.txt` floors mean

The floors in `requirements.txt` are the versions the code is written against;
they are **not** an exhaustively tested range. `torch>=2.4` reflects the unified
`torch.amp.GradScaler(device)` API the trainer uses — earlier releases expose
`torch.cuda.amp.GradScaler()` instead.

## Current revision and CI

The current revision is executed by the CI workflow
[`.github/workflows/smoke.yml`](../.github/workflows/smoke.yml) on `ubuntu-latest`
with Python 3.11 and CPU-only PyTorch wheels, for `resnet18` and
`swin_tiny_patch4_window7_224`. Every push to `main` so far has passed, and the
workflow is the live record of whether the release default currently works.

Because CI runs `scripts/smoke_test.py` on the current revision, it exercises the
behavioural changes made after the local run above: wrap-around padding for images
smaller than one patch, whitespace-safe manifest parsing and missing-file handling,
the single-patch training-batch guard, relative-path generation, seed propagation,
and the evaluation limit/JSON paths. What has *not* been re-run is the original
local macOS / Python 3.9.6 combination against the current revision.

## Not covered by the checks

* **CUDA / mixed precision.** `training.amp` only takes effect on a CUDA device;
  the checks run on CPU, so the `torch.amp.GradScaler` path is constructed only
  when `--device cuda` is used and has not been exercised here.
* **Benchmark training.** No GPU training was performed for this release; see
  [SCOPE.md](SCOPE.md).
* **`timm` backbones other than the two exercised.** ResNet-18 and Swin-T run the
  full check suite. ResNet-34/50 and the `timm` Xception are instantiated and
  given a forward pass, but do not run the full suite.
* **The gate's "enabled and passing" branch.** The executed run covered the gate
  disabled and the gate rejecting every candidate; the partial-pass case was
  skipped because all candidates scored identically on the gate set.
