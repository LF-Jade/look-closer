#!/usr/bin/env python
"""Evaluate a checkpoint with image-level probability averaging.

Single checkpoint
-----------------
    python scripts/evaluate.py --config configs/eval_genimage.yaml \
        --checkpoint runs/genimage_resnet50/checkpoint151.pth

Legacy checkpoint scan (reproduces the reported selection rule)
--------------------------------------------------------------
    python scripts/evaluate.py --config configs/eval_genimage.yaml \
        --scan runs/genimage_resnet50

Held-out selection (recommended for new work)
---------------------------------------------
    python scripts/evaluate.py --config configs/eval_genimage.yaml \
        --scan runs/genimage_resnet50 --select-on data/genimage_val.txt

``--scan`` picks the checkpoint with the best mean accuracy over the selection
manifests. Without ``--select-on`` those are the reported target sets, exactly as
in the retained evaluation implementation, and the paper documents target-selected
runs — which is why it discloses their scores as optimistic. With ``--select-on``
a held-out manifest drives selection and the config manifests are only reported.

Both thresholds are 0-1 ratios. ``gate_threshold`` is a minimum accuracy on the
first selection manifest (0 disables it); the historical logs printed it as a
percentage, e.g. 99.5 -> 0.995. If every candidate fails the gate the command
exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look_closer import (  # noqa: E402
    AllCheckpointsGatedError,
    DetectionModel,
    evaluate_manifests,
    load_checkpoint,
    load_config,
    resolve_device,
    scan_checkpoints,
    select_best_checkpoint,
)
from train import build_transform  # noqa: E402  (scripts share the transform builder)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--scan", metavar="EXP_DIR",
                        help="scan checkpoint*.pth in EXP_DIR and select by mean accuracy")
    parser.add_argument("--select-on", metavar="MANIFEST",
                        help="held-out manifest used to drive selection (recommended)")
    parser.add_argument("--manifests", nargs="*", default=None,
                        help="override the manifests from the config")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate at most N images per manifest (no files are written)")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set model.backbone=resnet18")
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        lowered = raw.lower()
        if lowered in ("true", "false"):
            value = lowered == "true"
        elif lowered in ("null", "none"):
            value = None
        else:
            value = raw
            for cast in (int, float):
                try:
                    value = cast(raw)
                    break
                except ValueError:
                    pass
        overrides[key] = value

    config = load_config(args.config, overrides)
    device = resolve_device(args.device)
    eval_cfg = config["eval"]
    sampling_cfg = config["sampling"]
    model_cfg = config["model"]

    manifests = args.manifests or eval_cfg["manifests"]

    input_size = model_cfg.get("input_size", sampling_cfg["patch_size"])
    if input_size != sampling_cfg["patch_size"]:
        raise ValueError(
            f"model.input_size ({input_size}) must equal sampling.patch_size "
            f"({sampling_cfg['patch_size']})."
        )

    _sampler, transform = build_transform(sampling_cfg, seed=config["seed"])
    model = DetectionModel(
        backbone_name=model_cfg["backbone"],
        pretrained=False,  # weights come from the checkpoint
        dropout=model_cfg.get("dropout", 0.5),
        input_size=input_size,
    ).to(device)

    common = dict(
        sampler=transform,
        device=device,
        batch_size=eval_cfg["batch_size"],
        num_workers=eval_cfg["num_workers"],
        decision_threshold=eval_cfg.get("decision_threshold", 0.5),
        on_missing=eval_cfg.get("on_missing", "error"),
        limit=args.limit,
    )

    if args.scan:
        checkpoints = scan_checkpoints(args.scan)
        if not checkpoints:
            print(f"no checkpoint*.pth (or legacy epoch*.pth) found in {args.scan}")
            return 1
        select_manifests = [args.select_on] if args.select_on else manifests
        mode = "held-out" if args.select_on else "legacy (reported target sets)"
        print(f"scanning {len(checkpoints)} checkpoint(s); selection = {mode}; "
              f"gate_threshold={eval_cfg.get('gate_threshold', 0.0)}")
        try:
            best = select_best_checkpoint(
                model, checkpoints, select_manifests,
                gate_threshold=eval_cfg.get("gate_threshold", 0.0),
                commit_to=args.scan, **common,
            )
        except AllCheckpointsGatedError as exc:
            print(f"error: {exc}")
            return 2
        print(f"selected {os.path.basename(best['checkpoint'])}  "
              f"mean accuracy {best['mean_accuracy']:.4f}  "
              f"({best['num_considered']}/{len(checkpoints)} passed the gate)")
        print(json.dumps(best["per_manifest"], indent=2))
        if args.json_out:
            payload = {
                "mode": "scan",
                "selection": "held-out" if args.select_on else "legacy-target-sets",
                "selected": best["checkpoint"],
                "mean_accuracy": best["mean_accuracy"],
                "num_candidates": len(checkpoints),
                "num_considered": best["num_considered"],
                "per_manifest": best["per_manifest"],
            }
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            print(f"wrote {args.json_out}")

        if args.select_on:
            state, _meta = load_checkpoint(best["checkpoint"])
            model.load_state_dict(state)
            results = evaluate_manifests(model, manifests, **common)
            for result in results:
                print(f"report {result['manifest']}  n={result['num_images']}  "
                      f"acc={result['accuracy']:.4f}  ap={result['ap']:.4f}")
        return 0

    if not args.checkpoint:
        args.checkpoint = config.get("checkpoint")
    if not args.checkpoint:
        parser.error("pass --checkpoint PATH, set `checkpoint:` in the config, or use --scan EXP_DIR")

    state, meta = load_checkpoint(args.checkpoint)
    model.load_state_dict(state)
    if meta:
        print(f"checkpoint metadata: epoch={meta.get('epoch')} step={meta.get('global_step')}")

    results = evaluate_manifests(model, manifests, **common)
    summary = []
    for result in results:
        print(f"{result['manifest']}  n={result['num_images']}  "
              f"acc={result['accuracy']:.4f}  ap={result['ap']:.4f}")
        summary.append({k: v for k, v in result.items() if k not in ("scores", "labels")})
    if len(summary) > 1:
        mean_acc = sum(s["accuracy"] for s in summary) / len(summary)
        print(f"mean accuracy over {len(summary)} manifest(s): {mean_acc:.4f}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
