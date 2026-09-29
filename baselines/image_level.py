#!/usr/bin/env python
"""Image-level control: aggregate a patch's logits per image, then supervise once.

This is the "aggregate-then-supervise" arm of the supervision-granularity
comparison.  It shares the training loop, optimiser, schedule, AMP setting,
gradient clipping and sampler with the main PWS trainer, and differs in exactly
one respect: the loss is applied to the averaged logits of an image rather than
to each patch.  That is what makes it a controlled comparison.

Usage:
    python baselines/image_level.py --config configs/genimage_resnet50.yaml

This is a thin wrapper: the switch lives in the shared loop
(``look_closer.engine.train_one_epoch(supervision="image")``), so the two arms
cannot drift apart in unrelated settings.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None,
                        help="default: <config output_dir>_image_level")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from look_closer import load_config  # noqa: E402

    config = load_config(args.config)
    base = config["output_dir"].rstrip("/")
    output_dir = args.output_dir or f"{base}_image_level"

    command = [
        sys.executable, os.path.join(REPO, "scripts", "train.py"),
        "--config", args.config,
        "--set", "training.supervision=image",
        "--set", f"output_dir={output_dir}",
    ]
    if args.device:
        command += ["--device", args.device]
    if args.limit:
        command += ["--limit", str(args.limit)]

    print("delegating to the shared trainer with supervision=image:")
    print("  " + " ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
