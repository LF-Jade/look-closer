"""Training, inference and evaluation for patch-wise supervision (PWS).

The two decisions that define the method live here:

* **training** applies the classification loss to every patch separately
  (``supervision="patch"``); ``supervision="image"`` averages an image's patch
  logits first and applies the loss once per image.  Both share one loop, one
  optimiser, one schedule and one AMP setting, so the comparison isolates the
  supervision granularity.
* **inference** groups synthetic-class softmax probabilities by source image and
  averages them into one image score, then applies a decision threshold.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
from contextlib import nullcontext
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import DeepfakeDataset
from .losses import image_level_loss

__all__ = [
    "average_precision",
    "train_one_epoch",
    "predict_image_scores",
    "evaluate_manifests",
    "save_checkpoint",
    "load_checkpoint",
    "scan_checkpoints",
    "select_best_checkpoint",
    "build_eval_loader",
    "AllCheckpointsGatedError",
]


class AllCheckpointsGatedError(RuntimeError):
    """Raised when every candidate checkpoint fails the gate."""


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision for the positive class, with tie-aware thresholds.

    Samples that share a score are treated as one threshold, so the result does
    not depend on the order of equal-scoring inputs.  Returns ``nan`` when there
    are no positives or no samples.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.size == 0 or labels.sum() == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    scores_sorted, labels_sorted = scores[order], labels[order]

    # One threshold per distinct score: the last index of each run of equal scores.
    distinct = np.where(np.diff(scores_sorted))[0]
    threshold_indices = np.r_[distinct, scores_sorted.size - 1]

    tp = np.cumsum(labels_sorted == 1)[threshold_indices]
    fp = np.cumsum(labels_sorted == 0)[threshold_indices]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(labels_sorted.sum()), 1)

    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


# --------------------------------------------------------------------------- #
# Checkpoint I/O
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model: torch.nn.Module, epoch: int, global_step: int,
                    config: dict | None = None) -> None:
    """Save weights together with the counters that produced them."""
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(
        {"model": state, "epoch": int(epoch), "global_step": int(global_step),
         "config": config or {}},
        path,
    )


def load_checkpoint(path: str) -> tuple[dict, dict]:
    """Load ``(state_dict, metadata)``.

    Handles both this release's format and the historical bare ``state_dict``
    files (``epoch*.pth``), whose metadata is empty because the filenames only
    encoded a save counter.
    """
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        meta = {k: v for k, v in payload.items() if k != "model"}
        return payload["model"], meta
    return payload, {}


_CHECKPOINT_RE = re.compile(r"^(?:checkpoint|epoch)(\d+)\.pth$")


def scan_checkpoints(exp_dir: str) -> list[str]:
    """All ``checkpoint*.pth`` files (and legacy ``epoch*.pth``), ordered by index."""
    paths = [
        path
        for path in glob.glob(os.path.join(exp_dir, "*.pth"))
        if _CHECKPOINT_RE.match(os.path.basename(path))
    ]

    def index(path: str) -> int:
        match = _CHECKPOINT_RE.match(os.path.basename(path))
        return int(match.group(1)) if match else -1

    return sorted(paths, key=index)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    scheduler=None,
    global_step: int = 0,
    epoch: int = 1,
    save_dir: str | None = None,
    save_interval: int = 0,
    checkpoint_counter: int = 1,
    clip_grad: float = 0.5,
    log_path: str | None = None,
    supervision: str = "patch",
    config: dict | None = None,
) -> tuple[int, int, float]:
    """One epoch of training.

    ``supervision="patch"`` applies the loss to every patch (PWS).
    ``supervision="image"`` averages each image's patch logits first and applies
    the loss once per image (the image-level control).  Everything else --
    optimiser, schedule, AMP, gradient clipping, checkpointing -- is shared.

    Checkpoints are written every ``save_interval`` **steps**, named
    ``checkpoint<N>.pth`` where ``N`` is a save counter.  The real epoch and step
    are stored inside the file.

    Returns ``(global_step, checkpoint_counter, mean_loss)``.
    """
    if supervision not in ("patch", "image"):
        raise ValueError(f"supervision must be 'patch' or 'image', got {supervision!r}")

    model.train()
    autocast = torch.autocast(device_type=device.type) if scaler is not None else nullcontext()
    running_loss, num_batches = 0.0, 0

    for patches, labels, num_patches in loader:
        global_step += 1
        patches = patches.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        # The classification head uses BatchNorm1d, which needs more than one
        # active element in training mode. A batch can collapse to a single patch
        # when the loader's last image batch holds one small image (a 64x64 image
        # yields exactly one patch at the release default). Fail with a clear
        # message instead of a BatchNorm traceback.
        if patches.shape[0] < 2:
            raise RuntimeError(
                f"effective patch batch is {patches.shape[0]} at step {global_step}; "
                "training needs at least 2 active patches per batch because the head "
                "uses BatchNorm1d. Drop the last batch, raise the image batch size, "
                "or ensure every image contributes more than one patch."
            )

        with autocast:
            outputs = model(patches)
            if supervision == "patch":
                loss = criterion(outputs, labels)
            else:
                loss = image_level_loss(criterion, outputs, labels, num_patches)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        running_loss += float(loss.detach())
        num_batches += 1

        if save_dir and save_interval > 0 and global_step % save_interval == 0:
            counter = checkpoint_counter
            checkpoint_counter += 1
            save_checkpoint(
                os.path.join(save_dir, f"checkpoint{counter}.pth"),
                model, epoch=epoch, global_step=global_step, config=config,
            )
            if log_path:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        f"epoch {epoch} | step {global_step} | checkpoint {counter} | "
                        f"{supervision}-level loss {running_loss / max(num_batches, 1):.4f}\n"
                    )

    return global_step, checkpoint_counter, running_loss / max(num_batches, 1)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def build_eval_loader(manifest: str, sampler, batch_size: int = 16, num_workers: int = 4,
                      on_missing: str = "error", limit: int | None = None) -> DataLoader:
    dataset = DeepfakeDataset(manifest, transform=sampler, on_missing=on_missing, limit=limit)
    dataset.manifest_stats = {"declared": dataset.num_declared, "missing": dataset.num_missing}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=DeepfakeDataset.collate_fn,
    )


@torch.no_grad()
def predict_image_scores(model: torch.nn.Module, loader: DataLoader,
                         device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image_scores, image_labels)``.

    Patch probabilities of one image are averaged into a single image score; the
    decision threshold is applied by the caller.
    """
    model.eval()
    scores: list[float] = []
    labels_out: list[int] = []

    for patches, labels, num_patches in loader:
        patches = patches.to(device, non_blocking=True)
        logits = model(patches)
        probs = F.softmax(logits, dim=1)[:, 1].float().cpu()
        for chunk in torch.split(probs, num_patches.tolist()):
            scores.append(float(chunk.mean()))
        image_labels = labels[torch.cumsum(num_patches, dim=0) - 1]
        labels_out.extend(int(value) for value in image_labels)

    return np.asarray(scores), np.asarray(labels_out)


def evaluate_manifests(
    model: torch.nn.Module,
    manifests: Sequence[str],
    sampler,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    decision_threshold: float = 0.5,
    on_missing: str = "error",
    limit: int | None = None,
) -> list[dict]:
    """Evaluate one or more manifests; returns per-manifest metrics.

    ``on_missing`` is forwarded to the dataset: ``"error"`` (default) refuses a
    manifest with absent images, ``"skip"`` drops them. Either way the result
    records ``num_declared`` and ``num_missing`` so the evaluated set can be
    compared with the declared one.
    """
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold is a 0-1 ratio, e.g. 0.5 for 50%")

    results = []
    for manifest in manifests:
        loader = build_eval_loader(manifest, sampler, batch_size, num_workers, on_missing, limit)
        scores, labels = predict_image_scores(model, loader, device)
        preds = (scores >= decision_threshold).astype(int)
        stats = getattr(loader.dataset, "manifest_stats", {"declared": None, "missing": None})
        results.append(
            {
                "manifest": manifest,
                "num_declared": stats["declared"],
                "num_missing": stats["missing"],
                "num_images": int(len(labels)),
                "accuracy": float((preds == labels).mean()) if len(labels) else float("nan"),
                "ap": average_precision(scores, labels),
                "scores": scores,
                "labels": labels,
            }
        )
    return results


def select_best_checkpoint(
    model: torch.nn.Module,
    checkpoints: Iterable[str],
    select_manifests: Sequence[str],
    sampler,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    decision_threshold: float = 0.5,
    gate_threshold: float = 0.0,
    commit_to: str | None = None,
    on_missing: str = "error",
    limit: int | None = None,
) -> dict:
    """Pick the checkpoint with the best mean accuracy over ``select_manifests``.

    .. warning::
       The retained evaluation implementation selected on the **reported target
       sets themselves**, and the paper documents target-selected runs; that makes
       the resulting scores optimistic. Complete selection histories for other runs
       were not recovered. Pass a held-out manifest via ``--select-on`` for any new
       experiment.

    Parameters
    ----------
    decision_threshold:
        0-1 ratio used to turn an image score into a label.
    gate_threshold:
        0-1 ratio; if greater than zero, a checkpoint is considered only when its
        accuracy on the **first** selection manifest reaches this value.  The
        historical logs printed the gate as a percentage (e.g. ``99.5``), which
        corresponds to ``0.995`` here.  ``0.0`` disables gating.

        Note a deliberate divergence from the retained evaluator: that code gated
        only its multi-manifest branch and ignored the gate in single-manifest
        mode, whereas this release applies the gate whenever it is non-zero.  The
        mean is taken over **all** selection manifests, including the first, in
        both.
    commit_to:
        Directory that receives a copy named ``best_epoch.pth``.

    Raises
    ------
    AllCheckpointsGatedError
        If no checkpoint satisfies the gate.
    """
    if not 0.0 <= gate_threshold <= 1.0:
        raise ValueError("gate_threshold is a 0-1 ratio, e.g. 0.995 for 99.5%")

    checkpoints = list(checkpoints)
    if not checkpoints:
        raise ValueError("no checkpoints to select from")

    best: dict | None = None
    considered = 0
    for checkpoint in checkpoints:
        state, _meta = load_checkpoint(checkpoint)
        model.load_state_dict(state)
        results = evaluate_manifests(model, select_manifests, sampler, device,
                                     batch_size, num_workers, decision_threshold,
                                     on_missing, limit)
        if gate_threshold > 0.0 and results[0]["accuracy"] < gate_threshold:
            continue
        considered += 1
        mean_acc = float(np.mean([r["accuracy"] for r in results]))
        if best is None or mean_acc > best["mean_accuracy"]:
            best = {
                "checkpoint": checkpoint,
                "mean_accuracy": mean_acc,
                "per_manifest": {r["manifest"]: r["accuracy"] for r in results},
            }

    if best is None:
        raise AllCheckpointsGatedError(
            f"all {len(checkpoints)} checkpoint(s) failed the gate "
            f"(gate_threshold={gate_threshold} on {select_manifests[0]})"
        )

    best["num_considered"] = considered
    if commit_to:
        shutil.copy(best["checkpoint"], os.path.join(commit_to, "best_epoch.pth"))
    return best
