#!/usr/bin/env python
"""Train the multi-patch PWS pipeline.

Examples
--------
    # 1. no benchmark needed: generate a tiny synthetic set, then train on it
    python scripts/make_synthetic_data.py --out data/synthetic --count 16
    python scripts/train.py --config configs/genimage_resnet50.yaml \
        --set data.train_manifest=data/synthetic/train.txt \
        --set model.backbone=resnet18 --set model.pretrained=false \
        --set training.epochs=1 --set training.batch_size=4 --set training.num_workers=0

    # 2. the reported recipe
    python scripts/train.py --config configs/genimage_resnet50.yaml

    # 3. the image-level control arm (same loop, loss applied after logit averaging)
    python scripts/train.py --config configs/genimage_resnet50.yaml \
        --set training.supervision=image --set output_dir=runs/genimage_resnet50_image

The scheduler named in the config is built from ``torch.optim.lr_scheduler``, so the
training protocol never depends on an optional package being installed. Asking for
``cosine`` either gets a cosine schedule or raises.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look_closer import (  # noqa: E402
    DeepfakeDataset,
    DetectionModel,
    FocalLoss,
    PatchSampler,
    build_optimizer,
    build_scheduler,
    load_config,
    resolve_device,
    save_resolved_config,
    set_seed,
    train_one_epoch,
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transform(sampling: dict, seed: int = 42):
    """Sampler -> float -> ImageNet normalisation. No per-patch resizing.

    ``sampling.base_seed`` defaults to the top-level ``seed`` so that changing
    ``seed`` changes the candidate subset too; set it explicitly to decouple the
    two.
    """
    sampler = PatchSampler(
        patch_size=sampling["patch_size"],
        stride=sampling["stride"],
        max_patches=sampling["max_patches"],
        pre_crop_resize=sampling.get("pre_crop_resize", None),
        resize_mode=sampling.get("resize_mode", "pad"),
        base_seed=sampling.get("base_seed", seed),
        resample_each_epoch=sampling.get("resample_each_epoch", False),
    )
    return sampler, transforms.Compose(
        [
            transforms.Lambda(sampler),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def _coerce(raw: str):
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N images (smoke tests)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set training.epochs=1")
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        overrides[key] = _coerce(raw)

    config = load_config(args.config, overrides)
    training_cfg = config["training"]
    set_seed(config["seed"])
    device = resolve_device(args.device)
    os.makedirs(config["output_dir"], exist_ok=True)

    sampling_cfg = config["sampling"]
    model_cfg = config["model"]
    input_size = model_cfg.get("input_size", sampling_cfg["patch_size"])
    if input_size != sampling_cfg["patch_size"]:
        raise ValueError(
            f"model.input_size ({input_size}) must equal sampling.patch_size "
            f"({sampling_cfg['patch_size']}): patches are fed to the backbone "
            "at their native size."
        )

    sampler, transform = build_transform(sampling_cfg, seed=config["seed"])
    train_set = DeepfakeDataset(config["data"]["train_manifest"], transform=transform,
                                limit=args.limit)
    train_loader = DataLoader(
        train_set,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg["num_workers"],
        collate_fn=DeepfakeDataset.collate_fn,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    model = DetectionModel(
        backbone_name=model_cfg["backbone"],
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.5),
        input_size=input_size,
    ).to(device)

    criterion = FocalLoss().to(device)
    optimizer = build_optimizer(model, training_cfg["lr"],
                               training_cfg.get("weight_decay", 0.01))

    total_steps = len(train_loader) * training_cfg["epochs"]
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        name=training_cfg.get("scheduler", "cosine"),
        warmup_fraction=training_cfg.get("warmup_fraction", 0.1),
    )

    use_amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    supervision = training_cfg.get("supervision", "patch")

    resolved_path = save_resolved_config(config, config["output_dir"])
    print(f"device={device}  backbone={model_cfg['backbone']}  supervision={supervision}  "
          f"images={len(train_set)}  image-batch={training_cfg['batch_size']}  "
          f"steps/epoch={len(train_loader)}  scheduler={training_cfg.get('scheduler', 'cosine')}  "
          f"amp={use_amp}")
    print(f"resolved config: {resolved_path}")

    log_path = os.path.join(config["output_dir"], "train.log")
    global_step, counter = 0, 1
    for epoch in range(1, training_cfg["epochs"] + 1):
        sampler.set_epoch(epoch)
        global_step, counter, mean_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, scheduler=scheduler,
            global_step=global_step,
            epoch=epoch,
            save_dir=config["output_dir"],
            save_interval=training_cfg.get("save_interval", 0),
            checkpoint_counter=counter,
            log_path=log_path,
            supervision=supervision,
            config=config,
        )
        print(f"epoch {epoch}/{training_cfg['epochs']}  "
              f"mean {supervision}-level loss {mean_loss:.4f}  steps {global_step}  "
              f"checkpoints {counter - 1}")

    from look_closer import save_checkpoint  # local import keeps the header tidy

    save_checkpoint(os.path.join(config["output_dir"], "final.pth"), model,
                    epoch=training_cfg["epochs"], global_step=global_step, config=config)
    print(f"saved {os.path.join(config['output_dir'], 'final.pth')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
