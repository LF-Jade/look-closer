#!/usr/bin/env python
"""CPU end-to-end checks for the PWS pipeline.

This does **not** reproduce any paper number.  It runs on synthetic images with
randomly initialised weights and verifies:

1. sampler geometry, capping, and the default fixed-subset behaviour;
2. manifest parsing, and that a malformed manifest fails loudly;
3. training, checkpoint naming/metadata, strict reload, and both supervision modes;
4. image-level probability averaging, including variable patch counts per image;
5. average-precision conventions, including tied scores;
6. the learning-rate schedule sequence;
7. checkpoint selection: gate disabled, gate filtering, gate rejecting everything;
8. error handling for invalid configuration.

Run:  python scripts/smoke_test.py [--backbone resnet18]
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look_closer import (  # noqa: E402
    AllCheckpointsGatedError,
    DeepfakeDataset,
    DetectionModel,
    FocalLoss,
    PatchSampler,
    average_precision,
    build_optimizer,
    build_scheduler,
    cosine_lr_multiplier,
    evaluate_manifests,
    image_level_logits,
    load_checkpoint,
    predict_image_scores,
    scan_checkpoints,
    select_best_checkpoint,
    train_one_epoch,
)
from look_closer.models import build_backbone  # noqa: E402
from train import build_transform  # noqa: E402

FAILURES: list[str] = []
SKIPPED: list[str] = []
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def skip(name: str, why: str) -> None:
    print(f"  [skip] {name}  -- {why}")
    SKIPPED.append(name)


def expect_raises(name: str, fn, exc=Exception) -> None:
    try:
        fn()
    except exc:
        check(name, True)
    except Exception as other:  # noqa: BLE001
        check(name, False, f"raised {type(other).__name__}, expected {exc.__name__}")
    else:
        check(name, False, "no exception raised")


def make_images(root: str, count: int, size: int = 320, seed: int = 0,
                labels: list[int] | None = None) -> list[str]:
    """Synthetic images; when ``labels`` is given, brightness encodes the label.

    The brightness signal makes the task learnable, so checkpoints taken at
    different training steps genuinely differ. Without it the data is pure noise
    and every checkpoint sits at chance, which makes gates untestable.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    paths = []
    for index in range(count):
        if labels is None:
            low, high = 0, 255
        else:
            centre = 60 if labels[index] == 0 else 200
            low, high = max(centre - 30, 0), min(centre + 30, 255)
        coarse = rng.integers(low, high, size=(max(size // 16, 2), max(size // 16, 2), 3), dtype=np.uint8)
        image = Image.fromarray(coarse).resize((size, size), Image.BILINEAR)
        path = os.path.join(root, f"img{index:03d}.png")
        image.save(path)
        paths.append(path)
    return paths


def write_manifest(path: str, image_paths: list[str], labels: list[int]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for image_path, label in zip(image_paths, labels):
            handle.write(f"{image_path} {label}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default="resnet18")
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="look_closer_smoke_")
    print(f"workdir: {tmpdir}")
    device = torch.device("cpu")

    try:
        # ---------------------------------------------------------------- #
        print("\n[1] sampler")
        defaults = PatchSampler()
        check("release default is native-resolution cropping", defaults.pre_crop_resize is None,
              f"got {defaults.pre_crop_resize}")
        check("release default geometry",
              (defaults.patch_size, defaults.stride, defaults.max_patches) == (64, 32, 64),
              f"got {(defaults.patch_size, defaults.stride, defaults.max_patches)}")

        # the resize-before-crop ablation path retained from the saved sampler
        sampler = PatchSampler(patch_size=64, stride=32, max_patches=64, pre_crop_resize=299)
        probe = Image.open(make_images(os.path.join(tmpdir, "probe"), 1)[0]).convert("RGB")
        check("ablation path: 299x299 canvas yields 81 candidates",
              len(sampler._get_coordinates(299, 299)) == 81,
              f"got {len(sampler._get_coordinates(299, 299))}")
        patches = sampler(probe)
        check("cap of 64 patches applied", patches.shape == (64, 3, 64, 64),
              f"got {tuple(patches.shape)}")
        check("same subset on repeat call (historical default)", torch.equal(patches, sampler(probe)))
        resampler = PatchSampler(patch_size=64, stride=32, max_patches=64,
                                 pre_crop_resize=299, resample_each_epoch=True)
        resampler.set_epoch(3)
        check("resample_each_epoch=True changes the subset", not torch.equal(patches, resampler(probe)))
        expect_raises("invalid resize_mode raises", lambda: PatchSampler(resize_mode="bogus"), ValueError)
        expect_raises("non-positive patch_size raises",
                      lambda: PatchSampler(patch_size=0), ValueError)
        expect_raises("non-positive pre_crop_resize raises",
                      lambda: PatchSampler(pre_crop_resize=-1), ValueError)

        # Small sources now really reach the pad branch, because the release default
        # crops at the source resolution. A solid colour makes black fill detectable.
        solid = Image.new("RGB", (16, 16), (10, 20, 30))
        padder = PatchSampler(patch_size=64, stride=32, max_patches=100, pre_crop_resize=None)
        canvas = padder._adjust_image_size(solid)
        check("pad grows a 16x16 source to 64x64", canvas.size == (64, 64), f"got {canvas.size}")
        check("pad tiles rather than leaving black fill",
              set(canvas.getdata()) == {(10, 20, 30)}, f"got {sorted(set(canvas.getdata()))[:3]}")
        for size, expect in (((16, 80), (64, 80)), ((80, 16), (80, 64)), ((16, 16), (64, 64))):
            grown = padder._adjust_image_size(Image.new("RGB", size, (10, 20, 30)))
            check(f"pad handles a {size[0]}x{size[1]} source", grown.size == expect,
                  f"got {grown.size}, expected {expect}")
        check("a one-patch source yields exactly one patch",
              padder(solid).shape[0] == 1)

        # ---------------------------------------------------------------- #
        print("\n[2] data + transform")
        _, transform = build_transform(
            {"patch_size": 64, "stride": 32, "max_patches": 64,
             "pre_crop_resize": None, "resize_mode": "pad", "resample_each_epoch": False}
        )
        train_labels = [i % 2 for i in range(24)]
        train_paths = make_images(os.path.join(tmpdir, "images"), 24, seed=1, labels=train_labels)
        train_manifest = os.path.join(tmpdir, "train.txt")
        write_manifest(train_manifest, train_paths, train_labels)
        dataset = DeepfakeDataset(train_manifest, transform=transform)
        check("dataset length", len(dataset) == 24, f"got {len(dataset)}")
        items = [dataset[i] for i in range(2)]
        check("per-patch labels repeated", bool((items[0][1] == 0).all() and (items[1][1] == 1).all()))

        bad_manifest = os.path.join(tmpdir, "bad.txt")
        with open(bad_manifest, "w", encoding="utf-8") as handle:
            handle.write(f"{train_paths[0]} 7\n")  # label outside {0,1}
        expect_raises("malformed manifest raises", lambda: DeepfakeDataset(bad_manifest), ValueError)

        missing_manifest = os.path.join(tmpdir, "missing.txt")
        write_manifest(missing_manifest, [train_paths[0], "/nonexistent/x.png"], [0, 1])
        expect_raises("missing image raises by default",
                      lambda: DeepfakeDataset(missing_manifest, transform=transform),
                      FileNotFoundError)
        skipped = DeepfakeDataset(missing_manifest, transform=transform, on_missing="skip")
        check("missing images can be dropped explicitly",
              len(skipped) == 1 and skipped.num_declared == 2 and skipped.num_missing == 1,
              f"len={len(skipped)} declared={skipped.num_declared} missing={skipped.num_missing}")

        # A wide separator must not leave trailing whitespace on the path.
        spaced_manifest = os.path.join(tmpdir, "spaced.txt")
        with open(spaced_manifest, "w", encoding="utf-8") as handle:
            handle.write(f"{train_paths[0]}  1\n")
        check("a doubled separator still parses",
              len(DeepfakeDataset(spaced_manifest, transform=transform)) == 1)

        loader = DataLoader(dataset, batch_size=8, shuffle=False,
                            collate_fn=DeepfakeDataset.collate_fn)
        batch = next(iter(loader))
        check("collate concatenates patches", batch[0].shape[0] == int(batch[2].sum()),
              f"{batch[0].shape[0]} vs {int(batch[2].sum())}")
        check("patch counts per image", int(batch[2].sum()) == 8 * 64)

        # ---------------------------------------------------------------- #
        print("\n[3] training, checkpoints, both supervision modes")
        model = DetectionModel(backbone_name=args.backbone, pretrained=False).to(device)
        criterion = FocalLoss()
        optimizer = build_optimizer(model, 1e-3)
        out_dir = os.path.join(tmpdir, "run")
        os.makedirs(out_dir, exist_ok=True)

        log_path = os.path.join(out_dir, "train.log")
        global_step, counter, losses = 0, 1, []
        for epoch in range(1, 4):
            # epoch 3 switches the loss aggregation without touching anything else
            supervision = "patch" if epoch < 3 else "image"
            global_step, counter, loss = train_one_epoch(
                model, loader, criterion, optimizer, device,
                save_dir=out_dir, save_interval=3, epoch=epoch,
                checkpoint_counter=counter, global_step=global_step,
                log_path=log_path, supervision=supervision, config={"a": 1})
            losses.append(loss)
        check("both supervision modes produce finite losses",
              all(math.isfinite(value) for value in losses),
              " ".join(f"{v:.4f}" for v in losses))

        saved = sorted(p for p in os.listdir(out_dir) if p.endswith(".pth"))
        check("step-interval checkpoints written and counter continues across epochs",
              saved == ["checkpoint1.pth", "checkpoint2.pth", "checkpoint3.pth"], f"got {saved}")

        state, meta = load_checkpoint(os.path.join(out_dir, "checkpoint1.pth"))
        check("checkpoint metadata records real epoch and step",
              meta.get("epoch") == 1 and meta.get("global_step") == 3, f"got {meta}")
        check("scan_checkpoints finds them in order", len(scan_checkpoints(out_dir)) == 3)

        reloaded = DetectionModel(backbone_name=args.backbone, pretrained=False)
        reloaded.load_state_dict(state)
        check("checkpoint loads strictly into a fresh model", True)
        expect_raises("invalid supervision raises",
                      lambda: train_one_epoch(model, loader, criterion, optimizer, device,
                                              supervision="bogus"), ValueError)

        # ---------------------------------------------------------------- #
        print("\n[4] inference: probability averaging with variable patch counts")
        eval_paths = make_images(os.path.join(tmpdir, "eval_images"), 6, seed=7)
        eval_manifest = os.path.join(tmpdir, "eval.txt")
        write_manifest(eval_manifest, eval_paths, [0, 1, 0, 1, 1, 0])
        results = evaluate_manifests(reloaded, [eval_manifest], transform, device,
                                     batch_size=2, num_workers=0, decision_threshold=0.5)
        scores, labels = results[0]["scores"], results[0]["labels"]
        check("one score per image", len(scores) == 6, f"got {len(scores)}")
        check("accuracy in [0, 1]", 0.0 <= results[0]["accuracy"] <= 1.0)

        reloaded.eval()
        with torch.no_grad():
            reference = float(torch.softmax(
                reloaded(transform(Image.open(eval_paths[2]).convert("RGB"))), dim=1)[:, 1].mean())
        check("image score equals the mean patch probability",
              abs(reference - float(scores[2])) < 1e-5,
              f"{reference:.6f} vs {float(scores[2]):.6f}")

        # unequal patch counts per image exercise the grouping path
        ragged_sampler = PatchSampler(patch_size=64, stride=64, max_patches=100, pre_crop_resize=None)
        sizes = [128, 192, 320, 256, 64, 448]
        ragged_dir = os.path.join(tmpdir, "ragged")
        os.makedirs(ragged_dir, exist_ok=True)
        ragged_paths = []
        for index, side in enumerate(sizes):
            path = os.path.join(ragged_dir, f"r{index}.png")
            Image.open(eval_paths[0]).resize((side, side)).save(path)
            ragged_paths.append(path)
        ragged_manifest = os.path.join(tmpdir, "ragged.txt")
        write_manifest(ragged_manifest, ragged_paths, [0, 1, 0, 1, 1, 0])
        ragged_transform = transforms.Compose([
            transforms.Lambda(ragged_sampler),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        counts = [ragged_transform(Image.open(p).convert("RGB")).shape[0] for p in ragged_paths]
        check("ragged sampler yields unequal patch counts", len(set(counts)) > 1, f"counts={counts}")
        ragged = evaluate_manifests(reloaded, [ragged_manifest], ragged_transform, device,
                                    batch_size=4, num_workers=0)
        check("ragged grouping yields one score per image", len(ragged[0]["scores"]) == len(sizes))
        with torch.no_grad():
            ref_ragged = float(torch.softmax(
                reloaded(ragged_transform(Image.open(ragged_paths[0]).convert("RGB"))), dim=1)[:, 1].mean())
        check("ragged grouping averages only that image's patches",
              abs(ref_ragged - float(ragged[0]["scores"][0])) < 1e-5,
              f"{ref_ragged:.6f} vs {float(ragged[0]['scores'][0]):.6f}")

        # ---------------------------------------------------------------- #
        print("\n[5] average precision conventions")
        ap = lambda s, l: average_precision(np.array(s, dtype=float), np.array(l, dtype=int))  # noqa: E731
        check("tied scores: order does not matter",
              ap([0.5, 0.5], [1, 0]) == ap([0.5, 0.5], [0, 1]),
              f"{ap([0.5, 0.5], [1, 0])} vs {ap([0.5, 0.5], [0, 1])}")
        check("tied scores: all-tied two-of-four is 0.5", abs(ap([0.5] * 4, [1, 1, 0, 0]) - 0.5) < 1e-12)
        check("perfect ranking is 1.0", abs(ap([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) - 1.0) < 1e-12)
        check("reversed ranking is 5/12", abs(ap([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) - 5 / 12) < 1e-9,
              f"got {ap([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])}")
        check("single positive ranks first", abs(ap([0.9, 0.1], [1, 0]) - 1.0) < 1e-12)
        check("no positives gives nan", math.isnan(ap([0.5, 0.4], [0, 0])))
        check("empty input gives nan", math.isnan(ap([], [])))
        pairs = [(0.9, 1), (0.2, 1), (0.9, 0), (0.2, 0)]
        shuffled = [pairs[i] for i in (2, 0, 3, 1)]
        forward = ap([p[0] for p in pairs], [p[1] for p in pairs])
        backward = ap([p[0] for p in shuffled], [p[1] for p in shuffled])
        check("permuting (score, label) pairs does not change AP",
              abs(forward - backward) < 1e-12, f"{forward} vs {backward}")
        check("tied pairs give 0.5", abs(forward - 0.5) < 1e-12, f"got {forward}")

        # ---------------------------------------------------------------- #
        print("\n[6] learning-rate schedule")
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        sched = build_scheduler(opt, total_steps=10, name="cosine", warmup_fraction=0.2)
        check("cosine scheduler is not None", sched is not None)
        multipliers = []
        for _ in range(10):
            multipliers.append(opt.param_groups[0]["lr"])
            sched.step()
        expected = [cosine_lr_multiplier(step, 10, 2) for step in range(10)]
        check("warmup then cosine matches the reference sequence",
              all(abs(a - b) < 1e-9 for a, b in zip(multipliers, expected)),
              f"first three {[round(m, 4) for m in multipliers[:3]]}")
        check("warmup ramps up", multipliers[0] < multipliers[1] <= multipliers[2])
        check("decay is monotone after warmup", all(a >= b - 1e-12 for a, b in zip(multipliers[2:], multipliers[3:])))
        check("schedule is identical without any optional package", True)
        check("scheduler='none' returns None",
              build_scheduler(torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0),
                              10, name="none") is None)
        expect_raises("unknown scheduler raises", lambda: build_scheduler(opt, 10, name="bogus"), ValueError)

        # ---------------------------------------------------------------- #
        print("\n[7] checkpoint selection and gate")
        gate_manifest = os.path.join(tmpdir, "gate.txt")
        # Brightness encodes base_labels, but 25% of the manifest labels are flipped,
        # so accuracy is informative yet cannot reach 1.0. That keeps both gate cases
        # (partial filtering and total rejection) constructible.
        base_labels = [i % 2 for i in range(48)]
        gate_labels = list(base_labels)
        flip_rng = np.random.default_rng(23)
        for index in flip_rng.choice(len(gate_labels), size=len(gate_labels) // 4, replace=False):
            gate_labels[int(index)] = 1 - gate_labels[int(index)]
        gate_paths = make_images(os.path.join(tmpdir, "gate_images"), 48, seed=11, labels=base_labels)
        write_manifest(gate_manifest, gate_paths, gate_labels)
        checkpoints = scan_checkpoints(out_dir)

        accuracies = []
        for checkpoint in checkpoints:
            state_i, _ = load_checkpoint(checkpoint)
            reloaded.load_state_dict(state_i)
            accuracies.append(evaluate_manifests(reloaded, [gate_manifest], transform, device,
                                                 batch_size=8, num_workers=0)[0]["accuracy"])
        lo, hi = min(accuracies), max(accuracies)
        print(f"      measured accuracies: {[round(a, 3) for a in accuracies]}")

        best = select_best_checkpoint(reloaded, checkpoints, [gate_manifest], transform, device,
                                      batch_size=8, num_workers=0, gate_threshold=0.0)
        check("gate disabled considers every checkpoint",
              best["num_considered"] == len(checkpoints), f"{best['num_considered']}/{len(checkpoints)}")

        if hi < 1.0:
            strict = min(1.0, hi + (1.0 - hi) / 2)
            expect_raises("gate rejecting everything raises AllCheckpointsGatedError",
                          lambda: select_best_checkpoint(reloaded, checkpoints, [gate_manifest],
                                                         transform, device, batch_size=8,
                                                         num_workers=0, gate_threshold=strict),
                          AllCheckpointsGatedError)
        else:
            skip("gate rejecting everything", "a checkpoint already reaches accuracy 1.0")

        if lo < hi:
            middle = (lo + hi) / 2
            filtered = select_best_checkpoint(reloaded, checkpoints, [gate_manifest], transform, device,
                                              batch_size=8, num_workers=0, gate_threshold=middle)
            check("gate filters some but not all candidates",
                  0 < filtered["num_considered"] < len(checkpoints),
                  f"{filtered['num_considered']}/{len(checkpoints)} passed gate {middle:.3f}")
        else:
            skip("gate filtering", f"all checkpoints share accuracy {lo:.3f}")

        expect_raises("gate_threshold above 1.0 raises",
                      lambda: select_best_checkpoint(reloaded, checkpoints, [gate_manifest],
                                                     transform, device, gate_threshold=1.5),
                      ValueError)
        expect_raises("decision_threshold above 1.0 raises",
                      lambda: evaluate_manifests(reloaded, [gate_manifest], transform, device,
                                                 decision_threshold=1.5), ValueError)

        # ---------------------------------------------------------------- #
        print("\n[8] invalid configuration")
        expect_raises("unknown backbone raises", lambda: build_backbone("nope", pretrained=False), ValueError)
        expect_raises("unknown ResNet variant raises",
                      lambda: build_backbone("resnet101", pretrained=True), ValueError)
        mismatched = subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts", "train.py"),
             "--config", os.path.join(REPO, "configs", "genimage_resnet50.yaml"),
             "--set", f"data.train_manifest={train_manifest}",
             "--set", "model.input_size=32",
             "--set", "training.epochs=1"],
            capture_output=True, text=True, timeout=600,
        )
        check("CLI rejects model.input_size != sampling.patch_size",
              mismatched.returncode != 0 and "must equal" in (mismatched.stderr + mismatched.stdout),
              f"exit={mismatched.returncode}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    if SKIPPED:
        print(f"skipped: {', '.join(SKIPPED)}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("all smoke-test checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
