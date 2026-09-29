#!/usr/bin/env python
"""Generate a small synthetic dataset so the pipeline can be run without benchmarks.

    python scripts/make_synthetic_data.py --out data/synthetic --count 16

Writes ``images/img*.png`` plus ``train.txt`` and ``eval.txt`` manifests in the
format described in docs/DATA.md.

**Path convention.** Manifest entries are written *relative to the repository
root*, so the repository stays movable. Every documented command is therefore
meant to be run from the repository root. Saved image files use their resolved
absolute path internally; only what goes into the manifest is relativised. If you
pass ``--out`` pointing outside the repository, entries become paths such as
``../elsewhere/img000.png``, which still resolve from the repository root.

The images are smooth random fields, enough to exercise the code paths. They
carry no detection signal, so any metric obtained from them is meaningless as a
performance number.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/synthetic",
                        help="output directory (relative paths resolve from the repository root)")
    parser.add_argument("--count", type=int, default=16, help="total images")
    parser.add_argument("--size", type=int, default=320, help="image side length")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    entries = []
    for index in range(args.count):
        rng = np.random.default_rng(args.seed + index)
        coarse = rng.integers(0, 255, size=(max(args.size // 16, 2), max(args.size // 16, 2), 3),
                              dtype=np.uint8)
        image = Image.fromarray(coarse).resize((args.size, args.size), Image.BILINEAR)
        absolute = os.path.join(image_dir, f"img{index:03d}.png")
        image.save(absolute)
        # Manifest keeps a repository-relative path so the tree can be moved.
        entries.append((os.path.relpath(absolute, REPO_ROOT), index % 2))

    split = max(1, args.count * 3 // 4)
    for name, subset in (("train.txt", entries[:split]),
                         ("eval.txt", entries[split:] or entries[:1])):
        manifest = os.path.join(out_dir, name)
        with open(manifest, "w", encoding="utf-8") as handle:
            for path, label in subset:
                handle.write(f"{path} {label}\n")
        print(f"{os.path.relpath(manifest, REPO_ROOT)}: {len(subset)} images")

    print(f"images written to {os.path.relpath(image_dir, REPO_ROOT)}")
    manifest_rel = os.path.relpath(os.path.join(out_dir, "train.txt"), REPO_ROOT)
    print("\ntry (from the repository root):")
    print("  python scripts/train.py --config configs/genimage_resnet50.yaml \\")
    print(f"      --set data.train_manifest={manifest_rel} \\")
    print("      --set model.backbone=resnet18 --set model.pretrained=false \\")
    print("      --set training.epochs=1 --set training.batch_size=4 --set training.num_workers=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
