"""Synthetic video clips from still images, and the frame-level vs temporal comparison.

D-Fire is an image dataset (its surveillance videos are not scriptable to download), so clips
are synthesised. The method is documented in the README; in short:

* negative clip: one negative image viewed by a static camera with per-frame jitter (sub-pixel
  translation, slow zoom drift, brightness flicker, sensor noise);
* positive clip: the same jitter model on a positive image whose annotated fire/smoke regions
  are first inpainted away (OpenCV Telea); the original pixels are then revealed inside
  ellipses over each ground-truth box that grow from 15% to 100% of the box over `ramp`
  frames. This gives a real onset time and growing smoke area with real pixels.

Clips run at SIM_FPS frames per second of simulated time. Detections are cached once per frame
and the alert engine is replayed offline for every configuration.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .alerts import AlertConfig, AlertEngine, Det, Level
from .data import IMG_SIZE, load_image, split_items

SIM_FPS = 2.0
N_FRAMES = 60
ONSET = 10
RAMP = 20
RUNS = Path("runs")


def _jitter(img: np.ndarray, t: int, rng: np.random.Generator, p: dict) -> np.ndarray:
    zoom = 1.0 + 0.04 * math.sin(t / 15 + p["ph"])
    M = np.float32(
        [
            [zoom, 0, p["dx"] * math.sin(t / 7 + p["ph"]) + (1 - zoom) * IMG_SIZE / 2],
            [0, zoom, p["dy"] * math.cos(t / 9 + p["ph"]) + (1 - zoom) * IMG_SIZE / 2],
        ]
    )
    f = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT).astype(
        np.float32
    )
    f = f * rng.uniform(0.92, 1.08) + rng.normal(0, 4, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def _reveal(clean: np.ndarray, orig: np.ndarray, boxes, s: float) -> np.ndarray:
    if s <= 0:
        return clean
    mask = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 * IMG_SIZE, (y1 + y2) / 2 * IMG_SIZE
        ax, ay = max((x2 - x1) * IMG_SIZE / 2 * s, 1), max((y2 - y1) * IMG_SIZE / 2 * s, 1)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(ax * 1.15), int(ay * 1.15)), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 4)[..., None]
    return (orig * mask + clean * (1 - mask)).astype(np.uint8)


def make_clip(name: str, boxes: list, seed: int) -> tuple[list[np.ndarray], int | None]:
    """Returns (frames, onset_frame or None for negative clips)."""
    rng = np.random.default_rng(seed)
    img = load_image(name)
    p = {"ph": rng.uniform(0, 6.28), "dx": rng.uniform(1, 4), "dy": rng.uniform(1, 4)}
    if not boxes:
        return [_jitter(img, t, rng, p) for t in range(N_FRAMES)], None
    mask = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    bx = [b for _, b in boxes]
    for x1, y1, x2, y2 in bx:
        mask[
            int(y1 * IMG_SIZE) : int(y2 * IMG_SIZE) + 1, int(x1 * IMG_SIZE) : int(x2 * IMG_SIZE) + 1
        ] = 255
    clean = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    frames = []
    for t in range(N_FRAMES):
        s = 0.0 if t < ONSET else 0.15 + 0.85 * min((t - ONSET) / RAMP, 1.0)
        frames.append(_jitter(_reveal(clean, img, bx, s), t, rng, p))
    return frames, ONSET


def build_clip_detections(det, n_pos: int = 150, n_neg: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    items = split_items("test")
    pos = [
        (n, b) for n, b in items if b and max((x[3] - x[1]) * (x[2] - x[0]) for _, x in b) > 0.02
    ]
    neg = [(n, b) for n, b in items if not b]
    hard = json.loads((RUNS / "hard_negatives.json").read_text())

    def pick(pool, k):
        return [pool[i] for i in rng.choice(len(pool), min(k, len(pool)), replace=False)]

    clips = [("pos", n, b) for n, b in pick(pos, n_pos)]
    clips += [("neg", n, []) for n, _ in pick(neg, n_neg)]
    clips += [("neg_fire_coloured", n, []) for n in hard["fire_coloured"][:40]]
    clips += [("neg_hazy", n, []) for n in hard["hazy"][:40]]
    out = []
    for i, (kind, n, b) in enumerate(clips):
        frames, onset = make_clip(n, b, seed * 100003 + i)
        dets = []
        for j in range(0, len(frames), 30):
            dets += det.predict(frames[j : j + 30])
        out.append({"kind": kind, "image": n, "onset": onset, "dets": dets})
        if i % 20 == 0:
            print(f"clip {i}/{len(clips)}", flush=True)
    RUNS.mkdir(exist_ok=True)
    (RUNS / "clip_dets.json").write_text(json.dumps(out))
    return out


def replay(dets_per_frame, cfg: AlertConfig):
    """Replay the alert engine. Returns (emitted alerts, frames-per-level, upward crossings)."""
    eng = AlertEngine(cfg)
    alerts, levels, up = [], [], []
    prev = Level.CLEAR
    for i, frame in enumerate(dets_per_frame):
        d = [Det(c, s, tuple(b)) for c, s, b in frame]
        a = eng.update(d, i / SIM_FPS)
        if a:
            alerts.append(a)
        if eng.level > prev:
            up.append((i, eng.level))
        prev = eng.level
        levels.append(eng.level)
    return alerts, levels, up


def summarise(clips, cfg: AlertConfig, level: Level = Level.WARNING) -> dict:
    """Recall, false-alarm episodes per hour and time-to-alert at a given minimum level."""
    pos = [c for c in clips if c["kind"] == "pos"]
    negs = {
        k: [c for c in clips if c["kind"] == k] for k in ("neg", "neg_fire_coloured", "neg_hazy")
    }
    tta, det_n = [], 0
    for c in pos:
        _, _, up = replay(c["dets"], cfg)
        hits = [i for i, lv in up if lv >= level and i >= c["onset"]]
        # a level held since before onset (spurious) is not credited as early detection
        if hits:
            det_n += 1
            tta.append((hits[0] - c["onset"]) / SIM_FPS)
    out = {
        "recall": det_n / max(len(pos), 1),
        "n_pos": len(pos),
        "tta_median_s": float(np.median(tta)) if tta else None,
        "tta_p90_s": float(np.percentile(tta, 90)) if tta else None,
    }
    tot_ep, tot_h = 0, 0.0
    for k, cl in negs.items():
        ep = sum(sum(1 for _, lv in replay(c["dets"], cfg)[2] if lv >= level) for c in cl)
        # count only crossings that start an episode (upward crossings into >= level)
        hours = len(cl) * N_FRAMES / SIM_FPS / 3600
        out[f"fa_per_hour_{k}"] = ep / hours if hours else None
        tot_ep, tot_h = tot_ep + ep, tot_h + hours
    out["fa_per_hour_all"] = tot_ep / tot_h if tot_h else None
    out["neg_hours"] = tot_h
    return out


def frame_level(clips, thr: dict[str, float]) -> dict:
    """Baseline without temporal logic: every frame with any detection over threshold alarms."""
    neg = [c for c in clips if c["kind"] != "pos"]
    pos = [c for c in clips if c["kind"] == "pos"]
    fa = sum(any(s >= thr[c_] for c_, s, _ in fr) for c in neg for fr in c["dets"])
    frames = len(neg) * N_FRAMES
    tta, det_n = [], 0
    for c in pos:
        hit = [
            i for i in range(c["onset"], N_FRAMES) if any(s >= thr[c_] for c_, s, _ in c["dets"][i])
        ]
        if hit:
            det_n += 1
            tta.append((hit[0] - c["onset"]) / SIM_FPS)
    return {
        "false_alarm_frame_rate": fa / frames,
        "fa_frames_per_hour": fa / (frames / SIM_FPS / 3600),
        "clip_false_alarm_rate": float(
            np.mean([any(any(s >= thr[c_] for c_, s, _ in fr) for fr in c["dets"]) for c in neg])
        ),
        "recall": det_n / max(len(pos), 1),
        "tta_median_s": float(np.median(tta)) if tta else None,
    }


def _detector():
    from .model import Detector

    return Detector()


def run_temporal(rebuild: bool = False, n_pos: int = 150, n_neg: int = 120) -> dict:
    ev = json.loads((RUNS / "eval.json").read_text())
    thr = ev["thresholds"]
    path = RUNS / "clip_dets.json"
    clips = (
        json.loads(path.read_text())
        if path.exists() and not rebuild
        else build_clip_detections(_detector(), n_pos, n_neg)
    )
    base = AlertConfig(fire_thr=thr["fire"], smoke_thr=thr["smoke"])
    res = {"thresholds": thr, "frame_level": frame_level(clips, thr), "sweep": [], "levels": {}}
    for M in (1, 2, 3, 4, 6, 8, 10, 12, 16):
        cfg = replace(base, window=M, min_hits=max(1, math.ceil(M * 0.67)), alarm_hold=M)
        for lv in (Level.WATCH, Level.WARNING, Level.ALARM):
            res["sweep"].append(
                {
                    "window": M,
                    "min_hits": cfg.min_hits,
                    "level": lv.name,
                    **summarise(clips, cfg, lv),
                }
            )
    default = replace(base, window=6, min_hits=4, alarm_hold=6)
    for lv in (Level.WATCH, Level.WARNING, Level.ALARM):
        res["levels"][lv.name] = summarise(clips, default, lv)
    # ablation: growth check disabled (smoke alone can then only reach WATCH)
    res["no_growth"] = summarise(clips, replace(default, growth_ratio=1e9), Level.WARNING)
    res["default"] = {"window": 6, "min_hits": 4}
    res["sim"] = {
        "fps": SIM_FPS,
        "frames": N_FRAMES,
        "onset": ONSET,
        "ramp": RAMP,
        "n_clips": {
            k: sum(c["kind"] == k for c in clips)
            for k in ("pos", "neg", "neg_fire_coloured", "neg_hazy")
        },
    }
    (RUNS / "temporal.json").write_text(json.dumps(res, indent=1))
    return res


__all__ = ["replace", "run_temporal"]
