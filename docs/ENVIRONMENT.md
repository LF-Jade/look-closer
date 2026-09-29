# Environment

## Verified environment

The checks in `scripts/smoke_test.py` were run in exactly this environment:

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

## What `requirements.txt` floors mean

The floors in `requirements.txt` are the versions the code is written against;
they are **not** an exhaustively tested range. `torch>=2.4` reflects the unified
`torch.amp.GradScaler(device)` API the trainer uses — earlier releases expose
`torch.cuda.amp.GradScaler()` instead. Only the environment above has been run.

## Not covered by the checks

* **The current release default has not been executed.** The checks above were
  run when `pre_crop_resize` defaulted to `299`. The default was subsequently
  changed to `null` (source-resolution cropping), together with the sampler
  signature, the three configs and the `train.py` fallback. The `[1] sampler`
  block gained a direct assertion on the new default, and the main transform now
  builds with `pre_crop_resize: None`, but **the modified code has not been run**.
  Treat the last executed state as historical, not as evidence for the current
  default.
* **Subsequent fixes have only been reviewed statically.** These include cyclic
  padding for small images, whitespace-safe manifest parsing and missing-file
  handling, the single-patch training-batch guard, relative-path generation,
  seed propagation, and the evaluation limit/JSON paths. New assertions are
  included, but have not been run on this revision.
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
