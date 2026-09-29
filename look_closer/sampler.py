"""Patch sampling for patch-wise supervision (PWS).

This is a cleaned port of the sampler retained with the multi-patch experiments
in "Look Closer: Patch-wise Supervision for AI-Generated Image Detection".

Three settings must be kept apart, because they are not the same thing:

1. **What the manuscript records.** The default multi-patch setting crops at the
   source resolution and then applies the same backbone-specific transforms as
   the paired whole-image baseline. The manuscript's resolution-aligned control
   instead resizes the source into a fixed 224 or 299 pixel canvas *before*
   cropping and then enlarges the extracted patches to the backbone input size.
2. **What the retained sampler actually does.** Its active pre-crop path
   unconditionally resizes the whole source image to 299x299. Its own comment
   labels this "ablation, add a resize before cropping", and no retained
   configuration switch disables it. It does *not* enlarge patches after
   cropping, so it realises only the first half of the manuscript's aligned
   control. The retained model path consumes 64x64 patches directly.
3. **What this release defaults to.** ``pre_crop_resize=None`` -- crop at the
   source resolution (setting 1's default), with 299 available explicitly as the
   retained resize-before-crop ablation path (setting 2). This release does *not*
   support enlarging patches after cropping, so ``None`` here means
   "native-resolution cropping into 64x64 patches", not a full reproduction of
   the manuscript's default pipeline.

Other behavioural notes:

* Patches are cropped at ``patch_size`` and are passed to the backbone at that
  size.  No per-patch upscaling happens here.
* The candidate grid is deterministic; when there are more candidates than
  ``max_patches`` a seeded ``random.Random`` sample is taken.  The retained
  training loop never called ``set_epoch``, so the same subset was drawn for a
  given image on every epoch.  ``resample_each_epoch`` defaults to ``False`` to
  preserve that behaviour; set it to ``True`` only if you intend a different
  protocol.
"""

from __future__ import annotations

import random

import torch
from PIL import Image
from torchvision import transforms

__all__ = ["PatchSampler"]


class PatchSampler:
    """Split an image into up to ``max_patches`` square RGB patches.

    Parameters
    ----------
    patch_size:
        Side length of each crop.
    stride:
        Step between candidate crop origins.
    max_patches:
        Maximum number of patches returned per image.  If the candidate grid is
        larger, a seeded subset is drawn.
    pre_crop_resize:
        Side length the whole source image is resized to before cropping.
        ``None`` (the release default) crops at the source resolution.  ``299``
        selects the resize-before-crop ablation path retained in the saved
        sampler.  See the module docstring for how these differ from the
        manuscript.
    resize_mode:
        ``"pad"`` (default) grows the canvas to at least one patch in each
        direction by repeating the source (wrap-around tiling); it is a no-op for
        any source at least ``patch_size`` across.  ``"resize"`` instead squeezes
        the whole source into a single ``patch_size`` square.
    base_seed:
        Seed for the candidate subset draw.
    resample_each_epoch:
        If True, ``set_epoch`` changes the subset; if False (default, matching
        the saved training loop) the subset is fixed per image.
    """

    def __init__(
        self,
        patch_size: int = 64,
        stride: int = 32,
        max_patches: int = 64,
        pre_crop_resize: int | None = None,
        resize_mode: str = "pad",
        base_seed: int = 42,
        resample_each_epoch: bool = False,
    ) -> None:
        if resize_mode not in ("pad", "resize"):
            raise ValueError(f"Invalid resize_mode: {resize_mode!r}")
        for label, value in (("patch_size", patch_size), ("stride", stride),
                             ("max_patches", max_patches)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer, got {value!r}")
        if pre_crop_resize is not None and (
            not isinstance(pre_crop_resize, int) or pre_crop_resize < 1
        ):
            raise ValueError(
                f"pre_crop_resize must be a positive integer or None, got {pre_crop_resize!r}"
            )
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches = max_patches
        self.pre_crop_resize = pre_crop_resize
        self.resize_mode = resize_mode
        self.base_seed = base_seed
        self.resample_each_epoch = resample_each_epoch
        self.epoch_offset = 0
        self._to_tensor = transforms.ToTensor()

    # ------------------------------------------------------------------ #
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch offset used when drawing the candidate subset."""
        if self.resample_each_epoch:
            self.epoch_offset = epoch

    # ------------------------------------------------------------------ #
    def _adjust_image_size(self, img: Image.Image) -> Image.Image:
        if self.pre_crop_resize is not None:
            img = img.resize((self.pre_crop_resize, self.pre_crop_resize), Image.BILINEAR)

        W, H = img.size
        if self.resize_mode == "resize":
            return img.resize((self.patch_size, self.patch_size), Image.BILINEAR)

        if W >= self.patch_size and H >= self.patch_size:
            return img

        # "pad": grow the canvas to at least one patch in each direction by
        # repeating (wrapping) the source. Stepping by the source size guarantees
        # the canvas is covered for any input size, including one far smaller
        # than a patch. An earlier revision mirrored only the first
        # ``patch_size - W`` columns, which left black fill whenever the source
        # was narrower than that strip.
        new_W = max(W, self.patch_size)
        new_H = max(H, self.patch_size)
        canvas = Image.new(img.mode, (new_W, new_H))
        for y in range(0, new_H, H):
            for x in range(0, new_W, W):
                canvas.paste(img, (x, y))
        return canvas

    # ------------------------------------------------------------------ #
    def _get_coordinates(self, W: int, H: int) -> list[tuple[int, int, int, int]]:
        p, s = self.patch_size, self.stride

        x_steps = list(range(0, W - p + 1, s))
        if (W - p) % s != 0:
            x_steps.append(W - p)
        y_steps = list(range(0, H - p + 1, s))
        if (H - p) % s != 0:
            y_steps.append(H - p)

        return [
            (x, y, x + p, y + p)
            for x in x_steps
            for y in y_steps
            if x + p <= W and y + p <= H
        ]

    # ------------------------------------------------------------------ #
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Return a ``[N, 3, patch_size, patch_size]`` float tensor."""
        img = self._adjust_image_size(img)
        coords = self._get_coordinates(*img.size)

        if len(coords) > self.max_patches:
            rng = random.Random(self.base_seed + self.epoch_offset)
            coords = rng.sample(coords, self.max_patches)

        if not coords:
            return torch.zeros(0, 3, self.patch_size, self.patch_size)

        return torch.stack([self._to_tensor(img.crop(box)) for box in coords])

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"PatchSampler(patch_size={self.patch_size}, stride={self.stride}, "
            f"max_patches={self.max_patches}, pre_crop_resize={self.pre_crop_resize}, "
            f"resize_mode={self.resize_mode!r})"
        )
