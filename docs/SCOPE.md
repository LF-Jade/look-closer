# Scope, fidelity and limitations

This release implements the **multi-patch PWS pipeline** so that the method can be
read, run and extended. The paper reports the original research experiments;
this repository packages their core multi-patch method as a reference
implementation, rather than bundling every experimental variant and trained
checkpoint.

## What this repository covers

* patch extraction with an explicit, configurable pre-crop resize;
* per-patch classification with a shared backbone and a shared head;
* the per-patch focal objective, and the image-level control objective as a
  single switch in the same training loop;
* image-level inference by averaging synthetic-class probabilities;
* checkpoint writing (with the real epoch and step recorded inside the file) and
  the historical checkpoint-selection rule as an explicit legacy mode;
* a CPU end-to-end self-check that needs no data and no downloads.

## What it does not cover

**No detector checkpoints are provided.** Training and evaluating from scratch
requires the original benchmarks and compute. The reference implementation and
its CPU checks are separate from reproduction of the paper's benchmark tables;
the configuration differences relevant to that task are listed below.

Backbone initialisation is separate from this: `model.pretrained: true` downloads
third-party ImageNet weights for the backbone. That is an initialisation choice
for training from scratch, and it is unrelated to detector checkpoints, none of
which are distributed here.

**The single-patch selection study (SPD) is not included.** The implementation
used for that study has not been located in the retained materials. Running this
multi-patch model with one patch is *not* a reproduction of it: the single-patch
work used a complexity-based selection procedure and a different training
protocol, and the manuscript describes it as such.

**Configuration mapping for the original experiments.** The following distinctions
concern which settings this release implements and which run-level details have
been linked during preparation of the release. They are not a claim that the
original experiments were not performed or that their records do not exist.
In particular:

* the transform version behind the historical main-table runs is not identified.
  The manuscript records source-resolution cropping for the default multi-patch
  setting; the retained sampler's active path unconditionally resizes the source
  to 299×299 before cropping and carries no switch, and it does not enlarge
  patches afterwards. Preparation of this release has not linked each reported
  result to its exact transform version;
* the low-data subset labels (20% vs 5%) in the single-patch tables cannot be
  reconciled within the reviewed materials: the affected rows have not been
  linked to a particular manifest;
* the JPEG quality factor of the perturbation column cannot be verified, and the
  corresponding perturbation-evaluation script has not been located in the
  retained materials;
* checkpoint-selection thresholds have not been mapped for every reported run.

These are disclosed in the manuscript and in its evidence ledger rather than
worked around here.

## Four settings that must not be conflated

Throughout this repository, keep these apart — they are different claims:

| # | What it is | Where it is stated |
|---|---|---|
| 1 | what the **manuscript records** | the paper, plus [DATA.md](DATA.md) |
| 2 | what the **retained snapshot actually executes** | read from the original research code |
| 3 | what **this release chooses as its default** | `configs/`, module docstrings |
| 4 | what has actually been **run and verified here** | [ENVIRONMENT.md](ENVIRONMENT.md) |

Concretely for input handling: the release default is `pre_crop_resize: null`
(source-resolution cropping into 64×64 patches). That is setting 3 — a deliberate
release configuration. It is **not** a claim that the manuscript's complete
default pipeline has been restored, because the manuscript also specifies
backbone-specific transforms after cropping, and `model.input_size` is bound to
`sampling.patch_size` here, so independent post-crop enlargement is not
supported. Setting `pre_crop_resize: 299` selects setting 2's retained ablation
path; it likewise is not a reproduction of the manuscript's full
resolution-aligned control, which additionally enlarges patches.

## Checkpoint selection is optimistic by construction

`scripts/evaluate.py --scan` reproduces the retained evaluation implementation:
it scans the saved checkpoints and keeps the one with the highest **mean accuracy
over the manifests passed in**. The paper documents runs selected this way on
their target sets. Those were the reported target sets — there is no
independent validation split, and the first manifest can act as a gate via
`eval.gate_threshold`. Scores obtained this way are optimistic and must not be
read as an unbiased comparison. Pass `--select-on <held-out manifest>` for any new
experiment; that is the recommended path and the legacy scan is kept only for
fidelity to the original protocol.

Selection compatibility note: the retained evaluator applied the gate only in
its multi-manifest branch. This release applies a nonzero gate even to a single
selection manifest. In both implementations the multi-manifest mean includes
the first manifest. This interface is not an exact replay of every historical
selection setting.

## Naming

The method is called **PWS (patch-wise supervision)**. "DMPA" was the name used
in earlier experiment records and in the original figures; "IPS" was an
intermediate name during rewriting. Neither appears in the final manuscript.

## Backbone notes

* Backbones receive the patch tensor at `model.input_size`, which must equal
  `sampling.patch_size`; nothing is upscaled internally. ImageNet initialisation
  is an *initialisation choice*, the pretrained transform is never applied, and a
  model identifier such as `..._224` does not imply a 224×224 input.
* Swin takes its input size at construction time, so it is passed through rather
  than hard-coded.
* Xception historically came from the unmaintained `pretrainedmodels==0.7.4`
  package; here it is rebuilt on `timm`. `state_dict` keys therefore differ from
  the historical checkpoints, numerical equivalence is not claimed, and no such
  checkpoints are distributed.
* The classification head has one definition (with `BatchNorm1d`), matching the
  head that is active in the saved snapshot. An earlier variant without it is not
  reproduced: the two produce different `state_dict` keys, and a strict load
  raises rather than quietly succeeding.

## Verification boundary

The included checks validate the implementation on synthetic data — geometry,
collation, both supervision modes, checkpoint round-trips, metric conventions,
the schedule sequence, selection and gating, and error handling. Benchmark
training and performance reproduction are outside the validation performed for
this release.

Two things the checks do **not** cover, stated explicitly rather than implied:

* the mixed-precision path (`training.amp`) only activates on a CUDA device and
  has not been exercised — see [ENVIRONMENT.md](ENVIRONMENT.md);
* only ResNet-18 and Swin-T run the full check suite; ResNet-34/50 and the
  `timm` Xception are instantiated and given a forward pass, but not more.
