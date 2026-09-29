# Look Closer — Patch-wise Supervision (PWS)

Reference implementation of the **multi-patch** pipeline studied in
*Look Closer: Patch-wise Supervision for AI-Generated Image Detection*.

The recipe is deliberately small:

1. split an image into explicit RGB crops;
2. classify **every crop independently** with one shared backbone and one shared
   head, and give **every crop its own classification loss** — there is no
   within-image pooling before the loss;
3. at inference, average the synthetic-class probabilities of an image's crops
   into a single image score and threshold it.

No handcrafted residual filter, no learned image-level fusion module, and no
truncated backbone are involved.

---

## Install

Run all commands from the repository root. This is a research reference
implementation: the current revision has passed static review, but its latest
behavioral changes have not been executed. See [verification details](docs/ENVIRONMENT.md).

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

Self-check (same conditions, asserts the pipeline's behaviour):

```sh
python scripts/smoke_test.py
python scripts/smoke_test.py --backbone swin_tiny_patch4_window7_224
```

## Train

```sh
# the release's ResNet-50 configuration
python scripts/train.py --config configs/genimage_resnet50.yaml

# the Swin-T recipe
python scripts/train.py --config configs/genimage_swin_tiny.yaml

# any config value can be overridden
python scripts/train.py --config configs/genimage_resnet50.yaml \
    --set training.epochs=2 --set training.batch_size=8 --limit 64
```

The resolved configuration is written to `<output_dir>/resolved_config.yaml`.

Checkpoints are written every `training.save_interval` **steps** as
`checkpoint<N>.pth`, where `N` is a save counter. The real epoch and step are
stored inside the file. (The research code named these `epoch<N>.pth`; a 30-epoch
run with `save_interval: 2000` therefore produced 151 files, and the historical
logs' "Epoch 151" line is the 151st save. `scan_checkpoints` still reads that
legacy name.)

## Evaluate

```sh
python scripts/evaluate.py --config configs/eval_genimage.yaml \
    --checkpoint runs/genimage_resnet50/checkpoint151.pth
```

`--scan <exp_dir>` selects by mean accuracy over the manifests you pass, following
the retained evaluator's selection approach. The single-manifest gate behavior
differs as documented in [SCOPE.md](docs/SCOPE.md). **Pass `--select-on <held-out manifest>` for any new
work** — without it, selection happens on the reported target sets and the scores
are optimistic. See [docs/SCOPE.md](docs/SCOPE.md).

Both `eval.decision_threshold` and `eval.gate_threshold` are 0–1 ratios; the
historical logs printed the gate as a percentage, so `99.5` becomes `0.995`.
`gate_threshold: 0` disables gating. If every candidate fails the gate the
command exits non-zero.

## Data

One image per line: `<image path><whitespace><0|1>`, resolved relative to the
working directory (absolute paths are also accepted). Run the documented commands
from the repository root. No datasets, manifests or checkpoints are shipped; see
[docs/DATA.md](docs/DATA.md).

## Input handling: read this before comparing with the paper

`pre_crop_resize` defaults to **`null`** — crop at the source resolution, which
matches the default multi-patch setting described in the manuscript. Setting it
to `299` selects the resize-before-crop ablation path retained in the original
sampler.

Neither setting is a full reproduction of a historical pipeline: the manuscript's
resolution-aligned control also *enlarges the extracted patches* to the backbone
input size afterwards, and this release does not support that (`model.input_size`
is bound to `sampling.patch_size`). See
[docs/SCOPE.md](docs/SCOPE.md#four-settings-that-must-not-be-conflated) for the
four settings that must be kept apart — manuscript, retained snapshot, this
release's default, and what has actually been verified here.

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
docs/                 # data format and scope/limitations
```

## Scope and limitations

Please read [docs/SCOPE.md](docs/SCOPE.md) before comparing against the paper:

* only the multi-patch PWS pipeline is included;
* **no detector checkpoints are provided** and reproducing the reported tables is
  not guaranteed — some original implementation and evaluation details are
  unavailable;
* the implementation used for the single-patch selection study (SPD) has **not
  been located in the retained materials** and is not included in this
  repository.

## License

[MIT](LICENSE). The license text is already in the repository, but the choice of
license is the authors' to confirm before public release.

## Citation

```bibtex
@misc{lookcloser2026,
  title  = {Look Closer: Patch-wise Supervision for AI-Generated Image Detection},
  author = {Zhang, Zhida and Wu, Tao and Liu, Siyu and Cao, Jie},
  year   = {2026},
  note   = {Preprint}
}
```
