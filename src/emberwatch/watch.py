"""Real-time loop: source -> detector -> alert engine -> sink, with an FPS/latency readout."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .alerts import AlertConfig, AlertEngine, Det, Level, make_sink
from .model import Detector
from .viz import draw


def watch(
    source: str,
    sink_spec: str = "stdout",
    show: bool = False,
    out: str | None = None,
    max_frames: int | None = None,
    camera_id: str = "cam-0",
    ckpt: str | None = None,
    thresholds: Path = Path("runs/eval.json"),
) -> dict:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {source}")
    live = isinstance(src, int)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    thr = (
        json.loads(thresholds.read_text())["thresholds"]
        if thresholds.exists()
        else {"fire": 0.5, "smoke": 0.5}
    )
    det = Detector(Path(ckpt)) if ckpt else Detector()
    det.predict([np.zeros((320, 320, 3), np.uint8)])  # warm-up (MPS kernel compile)
    sink = make_sink(sink_spec)
    engine, writer = None, None
    lat, loop_t, n, t_start = [], [], 0, time.perf_counter()
    while True:
        t0 = time.perf_counter()
        ok, bgr = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t1 = time.perf_counter()
        dets = det.predict([rgb])[0]
        lat.append((time.perf_counter() - t1) * 1000)
        ts = time.perf_counter() - t_start if live else n / src_fps
        if engine is None:  # tune persistence to the achieved processing rate
            engine = AlertEngine(
                AlertConfig.for_fps(
                    src_fps if not live else 10.0, fire_thr=thr["fire"], smoke_thr=thr["smoke"]
                )
            )
        alert = engine.update([Det(c, s, tuple(b)) for c, s, b in dets], ts)
        if alert:
            sink.send(alert.to_payload(camera_id))
        n += 1
        loop_t.append(time.perf_counter() - t0)
        fps = 1.0 / np.mean(loop_t[-30:])
        if show or out:
            vis = draw(bgr, dets, engine.level, thr, f"{fps:4.1f} fps  {lat[-1]:4.0f} ms")
            if out and writer is None:
                writer = cv2.VideoWriter(
                    out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (vis.shape[1], vis.shape[0])
                )
            if writer:
                writer.write(vis)
            if show:
                cv2.imshow("emberwatch", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    cap.release()
    if writer:
        writer.release()
    stats = {
        "frames": n,
        "fps_end_to_end": n / max(sum(loop_t), 1e-9),
        "latency_ms_mean": float(np.mean(lat)) if lat else None,
        "latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
        "device": str(det.device),
        "final_level": (engine.level if engine else Level.CLEAR).name,
    }
    print(json.dumps(stats), flush=True)
    return stats
