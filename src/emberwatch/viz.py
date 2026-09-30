"""Frame overlays shared by `watch` and `render`."""

from __future__ import annotations

import cv2
import numpy as np

from .alerts import Level

COLORS = {"fire": (0, 90, 255), "smoke": (200, 200, 200)}  # BGR
LEVEL_COLORS = {
    Level.CLEAR: (90, 90, 90),
    Level.WATCH: (0, 200, 255),
    Level.WARNING: (0, 140, 255),
    Level.ALARM: (40, 40, 230),
}


def draw(
    frame_bgr: np.ndarray, dets, level: Level, thr: dict[str, float], text: str = ""
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for cls, score, (x1, y1, x2, y2) in dets:
        if score < thr[cls]:
            continue
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(out, p1, p2, COLORS[cls], 2)
        cv2.putText(
            out,
            f"{cls} {score:.2f}",
            (p1[0], max(p1[1] - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLORS[cls],
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(out, (0, 0), (w, 22), LEVEL_COLORS[level], -1)
    cv2.putText(
        out,
        f"{level.name}  {text}",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out
