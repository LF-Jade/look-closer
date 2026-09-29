"""Losses used by the patch-wise supervision (PWS) pipeline.

Two objectives are provided so that the supervision-granularity comparison in
the paper can be reproduced:

``FocalLoss``
    Applied to every patch prediction (PWS).  The mean is taken over all patches
    in the batch.  ``alpha`` is a common multiplier for all samples, not a
    class-balancing weight.

``image_level_loss``
    The image-level control: per-image logits are formed by averaging the patch
    logits of a source image, and the classification loss is then applied once
    per image.  This mirrors the historical ``train_for_image.py`` baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FocalLoss", "image_level_logits", "image_level_loss"]


class FocalLoss(nn.Module):
    """Focal loss with a common multiplier ``alpha`` and focusing exponent ``gamma``."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(outputs, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


def image_level_logits(
    outputs: torch.Tensor, num_patches: torch.Tensor
) -> torch.Tensor:
    """Average patch logits within each source image.

    Parameters
    ----------
    outputs:
        ``[total_patches, num_classes]`` logits for the concatenated batch.
    num_patches:
        ``[batch]`` number of patches contributed by each source image.

    Returns
    -------
    ``[batch, num_classes]`` image-level logits.
    """
    chunks = torch.split(outputs, num_patches.tolist(), dim=0)
    return torch.cat([c.mean(dim=0, keepdim=True) for c in chunks], dim=0)


def image_level_loss(
    criterion: nn.Module,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    num_patches: torch.Tensor,
) -> torch.Tensor:
    """Apply ``criterion`` once per image, after averaging that image's patch logits."""
    logits = image_level_logits(outputs, num_patches)
    image_labels = labels[torch.cumsum(num_patches, dim=0) - 1]
    return criterion(logits, image_labels)
