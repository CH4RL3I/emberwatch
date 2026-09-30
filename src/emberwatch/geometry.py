"""Box geometry in plain numpy: IoU and greedy NMS. Boxes are [x1, y1, x2, y2]."""

from __future__ import annotations

import numpy as np


def box_area(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    order = list(np.argsort(-np.asarray(scores, dtype=float), kind="stable"))
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        if order:
            ious = box_iou(boxes[i : i + 1], boxes[order])[0]
            order = [j for j, v in zip(order, ious, strict=True) if v <= iou_thr]
    return keep
