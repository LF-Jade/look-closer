"""Look Closer -- patch-wise supervision (PWS) for AI-generated image detection.

Reference implementation of the multi-patch pipeline studied in
"Look Closer: Patch-wise Supervision for AI-Generated Image Detection".

The method is deliberately small: a shared backbone classifies explicit RGB
crops, every crop receives its own classification loss, and crop probabilities
are averaged only at inference time to produce one image score.
"""

from .data import DeepfakeDataset, read_manifest
from .engine import (
    AllCheckpointsGatedError,
    average_precision,
    evaluate_manifests,
    load_checkpoint,
    predict_image_scores,
    save_checkpoint,
    scan_checkpoints,
    select_best_checkpoint,
    train_one_epoch,
)
from .losses import FocalLoss, image_level_logits, image_level_loss
from .models import DEFAULT_INPUT_SIZE, DetectionModel, build_backbone
from .sampler import PatchSampler
from .utils import (
    build_optimizer,
    build_scheduler,
    cosine_lr_multiplier,
    load_config,
    resolve_device,
    save_resolved_config,
    set_seed,
)

__version__ = "0.2.0"

__all__ = [
    "PatchSampler",
    "DeepfakeDataset",
    "read_manifest",
    "DetectionModel",
    "build_backbone",
    "DEFAULT_INPUT_SIZE",
    "FocalLoss",
    "image_level_logits",
    "image_level_loss",
    "train_one_epoch",
    "predict_image_scores",
    "evaluate_manifests",
    "average_precision",
    "save_checkpoint",
    "load_checkpoint",
    "scan_checkpoints",
    "select_best_checkpoint",
    "AllCheckpointsGatedError",
    "load_config",
    "save_resolved_config",
    "set_seed",
    "resolve_device",
    "build_optimizer",
    "build_scheduler",
    "cosine_lr_multiplier",
    "__version__",
]
