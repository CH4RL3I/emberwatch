"""Detection metrics: per-class average precision (all-point interpolation, VOC2010+ style)
and precision/recall at an operating threshold.

Matching rule: detections are processed in descending score order; each is matched to the
unmatched ground-truth box of the same class with highest IoU >= iou_thr, otherwise it is a
false positive.
"""

from __future__ import annotations

import numpy as np

from .geometry import box_iou


def match_detections(dets, gts, iou_thr=0.5):
    """dets: [(score, box)], gts: [box] for ONE class on ONE image.
    Returns (scores, is_tp) in descending score order and the number of GT boxes."""
    dets = sorted(dets, key=lambda d: -d[0])
    used = set()
    scores, tps = [], []
    g = np.asarray(gts, dtype=float).reshape(-1, 4)
    for s, b in dets:
        tp = False
        if len(g):
            ious = box_iou(np.asarray(b)[None], g)[0]
            for j in np.argsort(-ious):
                if ious[j] < iou_thr:
                    break
                if int(j) not in used:
                    used.add(int(j))
                    tp = True
                    break
        scores.append(s)
        tps.append(tp)
    return scores, tps, len(g)


def pr_curve(scores, tps, n_gt):
    """Precision/recall arrays sorted by descending score."""
    if not len(scores):
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tps, dtype=float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    return prec, rec, np.asarray(scores)[order]


def average_precision(prec: np.ndarray, rec: np.ndarray) -> float:
    """Area under the monotone precision envelope (all-point interpolation)."""
    if not len(prec):
        return 0.0
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_class(preds, gts, cls, iou_thr=0.5):
    """preds: per image list of (cls, score, box); gts: per image list of (cls, box).
    Returns dict with ap, curve arrays, and n_gt."""
    all_s, all_tp, n_gt = [], [], 0
    for p, g in zip(preds, gts, strict=True):
        s, tp, n = match_detections(
            [(sc, b) for c, sc, b in p if c == cls], [b for c, b in g if c == cls], iou_thr
        )
        all_s += s
        all_tp += tp
        n_gt += n
    prec, rec, thr = pr_curve(all_s, all_tp, n_gt)
    return {"ap": average_precision(prec, rec), "prec": prec, "rec": rec, "thr": thr, "n_gt": n_gt}


def operating_point(preds, gts, cls, score_thr, iou_thr=0.5):
    """Box-level precision/recall for one class at a score threshold."""
    tp = fp = n_gt = 0
    for p, g in zip(preds, gts, strict=True):
        kept = [(sc, b) for c, sc, b in p if c == cls and sc >= score_thr]
        _, tps, n = match_detections(kept, [b for c, b in g if c == cls], iou_thr)
        tp += sum(tps)
        fp += len(tps) - sum(tps)
        n_gt += n
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(n_gt, 1),
        "tp": tp,
        "fp": fp,
        "n_gt": n_gt,
    }
