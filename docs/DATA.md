# Data format

The pipeline takes **manifest text files**. Each line is one image:

```
<image path><whitespace><label>
```

* `<image path>` may be absolute or relative to the working directory, and may
  contain spaces. It is read as everything except the final two characters.
* `<label>` is a single character: `0` for a real photograph, `1` for a
  synthetic image.
* Blank lines are ignored.

Example:

```
/data/images/progan/000123.png 0
/data/images/sd14/000456.png 1
```

A malformed line raises an error immediately. (The research loader used
`line[:-2]`/`line[-1]` without validation, so a malformed manifest produced wrong
labels silently; that behaviour is not reproduced.)

## Paths

Manifest entries are resolved **relative to the working directory**, and every
documented command is meant to be run from the repository root. Entries may also
be absolute, which is convenient for large datasets kept outside the repository
or on a mounted disk. Nothing here hard-codes an author-specific path.

## Pointing at your own data

This repository ships **no images, manifests or checkpoints**, and it deliberately
does not document the benchmark datasets themselves. They are public and are
versioned elsewhere; restating their composition here would only go stale and
would not help anyone download them. Obtain each from its official source and
write manifests in the format above.

The benchmarks the paper uses, and the role each plays:

| Benchmark | Role in the paper |
|---|---|
| GenImage | training source SD v1.4; evaluation over eight generator subsets |
| AIGCDetectBenchmark | trained on ProGAN; sixteen evaluation subsets |
| Chameleon | harder real/synthetic collection |
| DiffusionForensics | LSUN-Bedroom to ImageNet transfer study |

For a runnable path before you have any benchmark, use
`scripts/make_synthetic_data.py`.

## Subsampling a manifest

`tools/make_manifest.py` mirrors the research utility that created the reduced
manifests:

```sh
python tools/make_manifest.py data/DIFFall_train.txt --interval 5 --out data/DIFFall_train_sub5.txt
```

It keeps every *N*-th line. **The `N` is a line interval, not a percentage.** In
the saved manifests `sub5` therefore contains about 20% of the lines and `sub20`
about 5%. The manuscript flags the affected low-data rows as unreconciled for
exactly this reason.

## Sampler inputs

The sampler does not read manifests; it operates on a decoded RGB image. For each
image it:

1. optionally resizes the whole image to `pre_crop_resize` (default **`null`**,
   i.e. crop at the source resolution);
2. builds a grid of `patch_size` windows with the given `stride`, adding
   boundary-aligned windows when the grid does not divide evenly;
3. if more candidates exist than `max_patches`, draws a seeded subset.

Patches are passed to the backbone at `patch_size`; nothing enlarges them
afterwards.

### `pre_crop_resize`: three things that are not the same

| | setting |
|---|---|
| **Manuscript, default multi-patch** | crop at the source resolution, then the same backbone-specific transforms as the paired whole-image baseline |
| **Manuscript, resolution-aligned control** | resize the source into a fixed 224 or 299 canvas **before** cropping, **then enlarge** the extracted patches to the backbone input size |
| **Saved sampler, active path** | unconditionally resizes the source to 299×299 before cropping; its own comment calls it an ablation; **no post-crop enlargement** |
| **This release** | default `null` (source-resolution cropping); `299` available explicitly as the retained ablation path |

So `299` in this release is the **resize-before-crop ablation only** — it is the
first half of the manuscript's aligned control, and it is *not* a full
reproduction of it. `null` is a deliberate release configuration, not a claim to
have restored the manuscript's complete default pipeline.

### Candidate counts

With `patch_size=64` and `stride=32`, a 299×299 canvas yields 81 candidates
(capped to `max_patches`). Because the retained training loop never advanced the
sampler's epoch offset, the same subset was used for a given image on every
epoch; `resample_each_epoch: true` opts into per-epoch redrawing instead.

Patches are fed to the backbone at `patch_size` with no per-patch upscaling.
