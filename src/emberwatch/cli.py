"""emberwatch command line: fetch, train, eval, watch, render."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="emberwatch", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download D-Fire and cache it as 320x320 JPEGs in data/")
    t = sub.add_parser("train", help="fine-tune the detector")
    t.add_argument("--minutes", type=float, default=35.0)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--resume", action="store_true")
    e = sub.add_parser("eval", help="mAP, operating point, false alarms, temporal simulation")
    e.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    e.add_argument("--temporal-only", action="store_true", help="reuse cached detections")
    e.add_argument("--skip-temporal", action="store_true")
    e.add_argument("--rebuild-clips", action="store_true")
    w = sub.add_parser("watch", help="real-time detection + alerting on a video file or webcam")
    w.add_argument("--source", required=True, help="video path or webcam index")
    w.add_argument("--sink", default="stdout", help="stdout | file:<path> | webhook")
    w.add_argument("--show", action="store_true")
    w.add_argument("--out", help="write annotated mp4")
    w.add_argument("--max-frames", type=int)
    w.add_argument("--camera-id", default="cam-0")
    w.add_argument("--stats-out", help="write throughput stats JSON")
    r = sub.add_parser("render", help="figures, hero GIF and docs/results.md")
    r.add_argument(
        "--bench-video",
        action="store_true",
        help="also build a demo video and measure watch throughput",
    )
    a = p.parse_args(argv)

    if a.cmd == "fetch":
        from .data import fetch

        fetch()
    elif a.cmd == "train":
        from .train import train

        train(a.minutes, a.batch_size, resume=a.resume)
    elif a.cmd == "eval":
        from .evaluate import run_eval

        if not a.temporal_only:
            run_eval(a.limit)
        if not a.skip_temporal:
            from .temporal import run_temporal

            run_temporal(rebuild=a.rebuild_clips)
    elif a.cmd == "watch":
        import json
        from pathlib import Path

        from .watch import watch

        stats = watch(a.source, a.sink, a.show, a.out, a.max_frames, a.camera_id)
        if a.stats_out:
            Path(a.stats_out).write_text(json.dumps(stats))
    elif a.cmd == "render":
        from .render import render

        render(a.bench_video)
