"""Dataset and collation for patch-wise supervision (PWS).

Manifest format
---------------
One image per line::

    <image path><whitespace><label>

``<label>`` is ``0`` (real) or ``1`` (synthetic).  The line is split on the last
run of whitespace, so the path may contain spaces and the separator may be more
than one character wide.  A path cannot end with whitespace.

A malformed line raises immediately.  A missing image also raises by default:
silently dropping entries changes the evaluated sample set.  Pass
``on_missing="skip"`` to drop them deliberately; the number dropped and the
number declared are then reported and carried into evaluation results.

Each ``__getitem__`` returns **all patches of one image** plus an image label
repeated per patch.  ``collate_fn`` concatenates the patches of the whole batch
and returns the per-image patch counts, which is what the image-level
aggregation in :mod:`look_closer.engine` needs.
"""

from __future__ import annotations

import os
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["DeepfakeDataset", "read_manifest"]

_LABELS = {"0", "1"}
_MISSING_POLICIES = ("error", "skip")


def read_manifest(path: str, on_missing: str = "error") -> list[tuple[str, int]]:
    """Read a manifest into ``(image_path, label)`` pairs.

    Parameters
    ----------
    on_missing:
        ``"error"`` (default) raises if an image is absent.  ``"skip"`` drops it
        and prints how many were dropped.
    """
    if on_missing not in _MISSING_POLICIES:
        raise ValueError(f"on_missing must be one of {_MISSING_POLICIES}, got {on_missing!r}")

    entries: list[tuple[str, int]] = []
    missing: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected '<image path> <0|1>', got {line!r}"
                )
            image_path, label_text = parts
            if label_text not in _LABELS:
                raise ValueError(
                    f"{path}:{lineno}: label must be 0 or 1, got {label_text!r}"
                )
            if not image_path:
                raise ValueError(f"{path}:{lineno}: empty image path")
            if not os.path.exists(image_path):
                if on_missing == "error":
                    raise FileNotFoundError(
                        f"{path}:{lineno}: image not found: {image_path!r}. "
                        "Paths are resolved relative to the working directory; "
                        "pass on_missing='skip' to drop missing entries explicitly."
                    )
                missing.append(image_path)
                continue
            entries.append((image_path, int(label_text)))

    if missing:
        preview = ", ".join(missing[:3]) + ("..." if len(missing) > 3 else "")
        print(f"[manifest] {path}: skipped {len(missing)} missing image(s): {preview}")
    return entries


class DeepfakeDataset(Dataset):
    """Yields every sampled patch of one image, with the image label repeated.

    ``num_declared`` is the number of usable manifest lines and ``num_missing``
    the number dropped, so a caller can report how much of the declared set was
    actually evaluated.
    """

    def __init__(
        self,
        txt_path: str,
        transform: Callable | None = None,
        on_missing: str = "error",
        limit: int | None = None,
    ) -> None:
        self.transform = transform
        self.manifest_path = txt_path
        entries = read_manifest(txt_path, on_missing=on_missing)

        with open(txt_path, "r", encoding="utf-8") as handle:
            declared = sum(1 for line in handle if line.strip())
        self.num_declared = declared
        self.num_missing = max(declared - len(entries), 0)

        if limit is not None:
            entries = entries[:limit]
        if not entries:
            raise ValueError(f"{txt_path}: no usable entries")
        self.data_list = entries

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        path, label = self.data_list[index]
        image = Image.open(path).convert("RGB")
        if self.transform is None:
            return image, label
        patches = self.transform(image)
        if patches.shape[0] == 0:
            raise RuntimeError(f"{path}: sampler produced no patches")
        return patches, torch.full((patches.shape[0],), label, dtype=torch.long)

    @staticmethod
    def collate_fn(batch: Sequence):
        """Concatenate patches across images and keep the per-image counts."""
        patches = torch.cat([item[0] for item in batch], dim=0)
        labels = torch.cat([item[1] for item in batch], dim=0)
        num_patches = torch.tensor([item[0].shape[0] for item in batch], dtype=torch.long)
        return patches, labels, num_patches
