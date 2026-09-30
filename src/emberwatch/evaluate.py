"""Held-out evaluation: mAP@0.5 per class, operating-point precision/recall, image-level false
alarms on negative images (overall and on heuristically mined hard-negative subsets)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .data import CLASSES, load_image, split_items
from .metrics import evaluate_class, operating_point, pr_curve
from .model import Detector

RUNS = Path("runs")


def predict_split(det: Detector, split: str, limit: int | None = None, bs: int = 32):
    items = split_items(split)[:limit]
    preds = []
    for i in range(0, len(items), bs):
        chunk = items[i : i + bs]
        preds += det.predict([load_image(n) for n, _ in chunk])
    return items, preds


def hard_negative_groups(names: list[str], frac: float = 0.15) -> dict[str, list[str]]:
    """Mine hard negatives from the no-annotation images with colour statistics.

    D-Fire ships no scene tags, so: 'fire_coloured' = negatives with the largest share of
    saturated orange/red pixels; 'hazy' = negatives with the lowest contrast and low
    saturation (fog, haze, overcast, steam-like). Top `frac` of each, by rank.
    """
    fire, haze = {}, {}
    for n in names:
        hsv = cv2.cvtColor(load_image(n), cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255  # h in degrees
        fire[n] = float((((h < 40) | (h > 340)) & (s > 0.5) & (v > 0.5)).mean())
        gray = v
        haze[n] = float(gray.std() + s.mean())  # low = flat and unsaturated
    k = max(int(len(names) * frac), 1)
    return {
        "fire_coloured": sorted(names, key=lambda n: -fire[n])[:k],
        "hazy": sorted(names, key=lambda n: haze[n])[:k],
    }


def pick_threshold(preds, gts, cls) -> float:
    """Score threshold maximising box-level F1 for one class (run on the validation split)."""
    best, best_f = 0.5, -1.0
    for t in np.arange(0.1, 0.95, 0.05):
        op = operating_point(preds, gts, cls, float(t))
        f = 2 * op["precision"] * op["recall"] / max(op["precision"] + op["recall"], 1e-9)
        if f > best_f:
            best, best_f = float(t), f
    return round(best, 2)


def image_fa_rate(preds, names_subset, names_all, thr: dict[str, float]) -> dict:
    idx = {n: i for i, n in enumerate(names_all)}
    sel = [idx[n] for n in names_subset]
    hit = sum(any(s >= thr[c] for c, s, _ in preds[i]) for i in sel)
    return {"n_images": len(sel), "false_alarm_images": hit, "rate": hit / max(len(sel), 1)}


def run_eval(limit: int | None = None) -> dict:
    RUNS.mkdir(exist_ok=True)
    det = Detector()
    v_items, v_preds = predict_split(det, "val", limit)
    t_items, t_preds = predict_split(det, "test", limit)
    v_gts = [[(c, b) for c, b in bx] for _, bx in v_items]
    t_gts = [[(c, b) for c, b in bx] for _, bx in t_items]
    thr = {c: pick_threshold(v_preds, v_gts, c) for c in CLASSES}
    out = {"thresholds": thr, "n_test": len(t_items), "classes": {}}
    curves = {}
    for c in CLASSES:
        r = evaluate_class(t_preds, t_gts, c)
        op = operating_point(t_preds, t_gts, c, thr[c])
        out["classes"][c] = {"ap50": r["ap"], "n_gt": r["n_gt"], **op}
        curves[c] = {"prec": r["prec"].tolist(), "rec": r["rec"].tolist()}
        # threshold position on the curve, for plotting
        out["classes"][c]["op_point"] = [op["recall"], op["precision"]]
    out["map50"] = float(np.mean([out["classes"][c]["ap50"] for c in CLASSES]))
    names = [n for n, _ in t_items]
    neg = [n for n, b in t_items if not b]
    pos = [n for n, b in t_items if b]
    groups = {"all_negatives": neg, **hard_negative_groups(neg)}
    out["image_level"] = {
        "positives_detected": image_fa_rate(t_preds, pos, names, thr)["rate"],
        **{f"fa_{k}": image_fa_rate(t_preds, v, names, thr) for k, v in groups.items()},
    }
    (RUNS / "eval.json").write_text(json.dumps({**out, "curves": curves}))
    (RUNS / "hard_negatives.json").write_text(json.dumps(groups))
    print(json.dumps(out, indent=1))
    return out


__all__ = ["pr_curve", "run_eval"]
