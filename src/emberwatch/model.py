"""SSDlite320 + MobileNetV3-Large (torchvision, BSD-3), COCO-pretrained, fire/smoke head."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

from .data import CLASSES, IMG_SIZE

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
torch.set_num_threads(4)
CKPT = Path("checkpoints/ssdlite.pt")


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(pretrained: bool = True, score_thresh: float = 0.05) -> nn.Module:
    """Load COCO weights, then swap the classification convs for (background + classes)."""
    w = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 if pretrained else None
    model = ssdlite320_mobilenet_v3_large(weights=w, weights_backbone=None)
    n = len(CLASSES) + 1
    ch = model.head.classification_head
    for i, seq in enumerate(ch.module_list):
        old = seq[-1]
        anchors = old.out_channels // ch.num_columns
        ch.module_list[i][-1] = nn.Conv2d(
            old.in_channels, anchors * n, old.kernel_size, old.stride, old.padding
        )
    ch.num_columns = n
    model.score_thresh = score_thresh
    model.detections_per_img = 50
    return model


class Detector:
    """Inference wrapper returning normalised (cls, score, box) triples per image."""

    def __init__(
        self, ckpt: Path = CKPT, device: torch.device | None = None, score_thresh: float = 0.05
    ):
        self.device = device or pick_device()
        self.model = build_model(pretrained=False, score_thresh=score_thresh)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, frames: list[np.ndarray]) -> list[list[tuple[str, float, list[float]]]]:
        """frames: RGB uint8 HxWx3 (any size; resized to 320x320 like training)."""
        import cv2

        x = torch.stack(
            [
                torch.from_numpy(cv2.resize(f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
                .permute(2, 0, 1)
                .float()
                / 255.0
                for f in frames
            ]
        ).to(self.device)
        out = self.model(list(x))
        res = []
        for o in out:
            b = (o["boxes"].cpu().numpy() / IMG_SIZE).clip(0, 1)
            s, lab = o["scores"].cpu().numpy(), o["labels"].cpu().numpy()
            res.append(
                [
                    (CLASSES[int(c) - 1], float(sc), bx.tolist())
                    for bx, sc, c in zip(b, s, lab, strict=True)
                ]
            )
        return res
