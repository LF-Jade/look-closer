"""Shared helpers: seeding, configuration, device, optimiser, scheduler.

The learning-rate schedule is implemented with ``torch.optim.lr_scheduler``
only, so the training protocol does not depend on whether an optional package is
installed.  A configuration that asks for ``cosine`` either gets it, or fails
loudly.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any

import numpy as np
import torch
import yaml

__all__ = [
    "load_config",
    "set_seed",
    "resolve_device",
    "build_optimizer",
    "build_scheduler",
    "cosine_lr_multiplier",
    "ensure_dir",
    "save_resolved_config",
]


def load_config(path: str, overrides: dict[str, Any] | None = None) -> dict:
    """Load a YAML config and apply dotted-key ``overrides``."""
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for key, value in (overrides or {}).items():
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return config


def save_resolved_config(config: dict, output_dir: str) -> str:
    """Persist the config actually used, so a run is self-describing."""
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "resolved_config.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float = 0.01):
    """AdamW, as in the saved multi-patch training configuration."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def cosine_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup then cosine decay, in ``[0, 1]``.

    ``step`` is 0-based and is the number of completed scheduler steps.
    """
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    name: str = "cosine",
    warmup_fraction: float = 0.1,
):
    """Build the learning-rate schedule named in the configuration.

    ``name="none"`` returns ``None`` and the optimiser keeps a constant rate.
    ``name="cosine"`` is the release schedule (linear warmup then cosine decay)
    and needs no third-party package. It is offered as a sensible default, not as
    a claim about the schedule used for every historical result.
    """
    name = (name or "none").lower()
    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"Unknown scheduler {name!r}; expected 'cosine' or 'none'")

    warmup_steps = int(total_steps * warmup_fraction)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_multiplier(step, total_steps, warmup_steps),
    )


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
