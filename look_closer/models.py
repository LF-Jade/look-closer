"""Model definition: a shared backbone plus a shared classification head.

Every patch of an image passes through the *same* backbone and the *same* head
and produces its own two logits.  There is no within-image feature fusion at any
point; patches are combined only after the fact, by averaging synthetic-class
probabilities at inference (see :mod:`look_closer.engine`).

Backbone notes
--------------
* Backbones receive the patch tensor at ``input_size``; nothing is upscaled
  internally.  ``input_size`` defaults to 64 and is meant to be kept equal to the
  sampler's ``patch_size``.
* For Swin the input size must be given at construction time, because the timm
  model bakes it in; it is therefore passed through rather than hard-coded.
* ``torchvision`` ImageNet weights are *initialisation only*: the pretrained
  transform is never applied, and an identifier such as ``..._224`` does not
  imply a 224x224 input.
* Xception historically came from the unmaintained ``pretrainedmodels==0.7.4``
  package.  It is reimplemented here on top of ``timm``; the resulting
  ``state_dict`` keys therefore differ from the historical checkpoints, which are
  not distributed with this release.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "ClassificationHead",
    "DetectionModel",
    "build_backbone",
    "SUPPORTED_BACKBONES",
    "DEFAULT_INPUT_SIZE",
]

DEFAULT_INPUT_SIZE = 64

SUPPORTED_BACKBONES = (
    "resnet18",
    "resnet34",
    "resnet50",
    "swin_tiny_patch4_window7_224",
    "xception",
)


class ClassificationHead(nn.Module):
    """Linear -> BatchNorm1d -> GELU -> Dropout -> Linear.

    This is the head that is active in the saved snapshot of the research code.
    An earlier variant without ``BatchNorm1d`` also exists there, commented out;
    it is deliberately not reproduced, because the two produce different
    ``state_dict`` keys and a strict load raises instead of quietly succeeding.
    """

    def __init__(self, in_features: int, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def build_backbone(name: str, pretrained: bool = True,
                   input_size: int = DEFAULT_INPUT_SIZE) -> tuple[nn.Module, int]:
    """Return ``(backbone, feature_dim)`` for a supported backbone name."""
    if name.startswith("resnet"):
        from torchvision import models

        weights = None
        if pretrained:
            table = {
                "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
                "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
                "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
            }
            if name not in table:
                raise ValueError(f"Unsupported ResNet variant: {name}")
            weights = table[name]
        net = getattr(models, name)(weights=weights)
        features = nn.Sequential(*list(net.children())[:-1])
        return _ResNetBackbone(features), _probe_dim(features, input_size)

    if name.startswith("swin"):
        import timm

        net = timm.create_model(
            name, pretrained=pretrained, num_classes=0,
            img_size=input_size, patch_size=4,
        )
        return _TimmBackbone(net), _probe_dim(net, input_size)

    if name == "xception":
        import timm

        try:
            net = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=0)
        except Exception as exc:  # pragma: no cover - depends on timm version
            raise RuntimeError(
                "Could not create 'legacy_xception' from timm. The historical runs "
                "used pretrainedmodels==0.7.4, which is unmaintained; install a timm "
                f"version that provides legacy_xception, or pass pretrained=False. ({exc})"
            ) from exc
        return _TimmBackbone(net), _probe_dim(net, input_size)

    raise ValueError(f"Unsupported backbone: {name!r}. Supported: {SUPPORTED_BACKBONES}")


class _ResNetBackbone(nn.Module):
    def __init__(self, features: nn.Module) -> None:
        super().__init__()
        self.features = features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class _TimmBackbone(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _probe_dim(module: nn.Module, input_size: int) -> int:
    """Feature width, obtained by a forward pass on a dummy patch.

    The probe runs in eval mode under ``no_grad`` and restores the previous mode,
    so it does not disturb BatchNorm statistics.
    """
    was_training = module.training
    module.eval()
    with torch.no_grad():
        out = module(torch.zeros(1, 3, input_size, input_size))
    module.train(was_training)
    return int(out.flatten(1).shape[1])


class DetectionModel(nn.Module):
    """Shared backbone + shared head, applied independently to every patch.

    Constructing the model runs one dummy forward pass to discover the feature
    width, so instantiation is not a purely declarative operation.
    """

    def __init__(self, backbone_name: str = "resnet18", num_classes: int = 2,
                 pretrained: bool = True, dropout: float = 0.5,
                 input_size: int = DEFAULT_INPUT_SIZE) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.input_size = input_size
        self.backbone, in_features = build_backbone(
            backbone_name, pretrained=pretrained, input_size=input_size
        )
        self.head = ClassificationHead(in_features, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
