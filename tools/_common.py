"""Shared helpers for the scripts in ``tools/``."""

from __future__ import annotations

__all__ = ["read_manifest_lines"]


def read_manifest_lines(path: str) -> list[str]:
    """Read a manifest, dropping blank lines but preserving line order."""
    with open(path, "r", encoding="utf-8") as handle:
        return [line for line in handle if line.strip()]
