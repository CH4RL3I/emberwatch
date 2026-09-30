"""Figures, hero GIF and docs/results.md from runs/*.json. No dataset pixels are written to
docs/: the GIF replays real detector boxes and alert levels on a neutral canvas."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from .alerts import AlertConfig, Level  # noqa: E402
from .temporal import N_FRAMES, SIM_FPS, replay  # noqa: E402
from .viz import COLORS, LEVEL_COLORS  # noqa: E402

RUNS, DOCS = Path("runs"), Path("docs")
C = {
    "fire": "#d95f02",
    "smoke": "#4a4a4a",
    "WATCH": "#b8a000",
    "WARNING": "#d95f02",
    "ALARM": "#b2182b",
}


def _load(n):
    return json.loads((RUNS / n).read_text())


def pr_figure(ev):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, c in zip(axs, ("fire", "smoke"), strict=True):
        cur = ev["curves"][c]
        ax.plot(cur["rec"], cur["prec"], color=C[c], lw=2)
        r, p = ev["classes"][c]["op_point"]
        ax.scatter(
            [r], [p], color="k", zorder=3, label=f"operating point (thr {ev['thresholds'][c]})"
        )
        ax.set(
            title=f"{c}: AP@0.5 = {ev['classes'][c]['ap50']:.3f}",
            xlabel="recall",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
    axs[0].set_ylabel("precision")
    fig.suptitle("Held-out test split, IoU 0.5", fontsize=10)
    fig.tight_layout()
    fig.savefig(DOCS / "pr_curves.png", dpi=150)
    plt.close(fig)


def tradeoff_figure(tm):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ax = axs[0]
    for lv, col in (("WARNING", C["WARNING"]),):
        rows = [r for r in tm["sweep"] if r["level"] == lv]
        x = [max(r["fa_per_hour_all"], 0.5) for r in rows]
        ax.plot(
            x,
            [r["recall"] for r in rows],
            "-o",
            color=col,
            ms=4,
            label=f"{lv.lower()}: vary window M",
        )
        if lv == "WARNING":
            for r, xi in zip(rows, x, strict=True):
                ax.annotate(
                    f"M={r['window']}",
                    (xi, r["recall"]),
                    textcoords="offset points",
                    xytext=(3, -10),
                    fontsize=7,
                )
    rows = [r for r in tm["thr_sweep"] if r["level"] == "WARNING"]
    x = [max(r["fa_per_hour_all"], 0.5) for r in rows]
    ax.plot(
        x,
        [r["recall"] for r in rows],
        "-s",
        color="#1b6ca8",
        ms=4,
        label="warning: vary score threshold (M=6)",
    )
    for r, xi in zip(rows, x, strict=True):
        ax.annotate(
            f"{r['thr']:.1f}",
            (xi, r["recall"]),
            textcoords="offset points",
            xytext=(3, 4),
            fontsize=7,
            color="#1b6ca8",
        )
    ax.set(
        xscale="log",
        xlabel="false-alarm episodes per hour of negative video (log axis)",
        ylabel="clip recall (alert after onset)",
        title="Recall vs false alarms",
    )
    ax.legend(fontsize=7, loc="upper left")
    rows = [r for r in tm["sweep"] if r["level"] == "WARNING"]
    axs[1].plot(
        [r["window"] for r in rows],
        [r["tta_median_s"] or np.nan for r in rows],
        "-o",
        color=C["WARNING"],
        ms=4,
        label="median",
    )
    axs[1].plot(
        [r["window"] for r in rows],
        [r["tta_p90_s"] or np.nan for r in rows],
        "--o",
        color="#888",
        ms=4,
        label="p90",
    )
    axs[1].set(
        xlabel="persistence window M (frames at 2 fps; N = ceil(0.67 M))",
        ylabel="time-to-alert after onset (s)",
        title="Cost of a longer window: delay",
    )
    axs[1].legend(fontsize=8)
    for a_ in axs:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "tradeoff.png", dpi=150)
    plt.close(fig)


def pick_clips(clips, cfg):
    """Illustrative pair: a positive clip whose detections start at onset and reach WARNING, and
    a negative clip with the most raw false-alarm frames that never leaves CLEAR."""
    thr = {"fire": cfg.fire_thr, "smoke": cfg.smoke_thr}

    def hits(c):
        return [any(sc >= thr[k] for k, sc, _ in fr) for fr in c["dets"]]

    best_pos, best_neg = None, None
    for c in clips:
        _, levels, up = replay(c["dets"], cfg)
        h = hits(c)
        if c["kind"] == "pos":
            first = [i for i, lv in up if lv >= Level.WARNING and i >= c["onset"]]
            if first and not any(h[: c["onset"]]):
                t = first[0] - c["onset"]
                if best_pos is None or abs(t - 4) < abs(best_pos[1] - 4):
                    best_pos = (c, t)
        elif max(levels) == Level.CLEAR:
            if best_neg is None or sum(h) > best_neg[1]:
                best_neg = (c, sum(h))
    return best_pos[0], best_neg[0]


def _panel(title, dets_all, levels, onset, i, cfg, w=320):
    """Canvas with the current frame's boxes plus a timeline: raw detection ticks and level."""
    img = np.full((w, w, 3), 34, np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, w - 1), (70, 70, 70), 1)
    thr = {"fire": cfg.fire_thr, "smoke": cfg.smoke_thr}
    for dcls, sc, (x1, y1, x2, y2) in dets_all[i]:
        if sc >= thr[dcls]:
            cv2.rectangle(
                img, (int(x1 * w), int(y1 * w)), (int(x2 * w), int(y2 * w)), COLORS[dcls], 2
            )
            cv2.putText(
                img,
                f"{dcls} {sc:.2f}",
                (int(x1 * w) + 3, min(int(y2 * w) - 5, w - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLORS[dcls],
                1,
                cv2.LINE_AA,
            )
    cv2.rectangle(img, (0, 0), (w, 26), (20, 20, 20), -1)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    tl = np.full((70, w, 3), 20, np.uint8)
    n = len(dets_all)
    step = w / n
    cv2.putText(
        tl,
        "raw detection per frame",
        (4, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        tl,
        "alert level after filter",
        (4, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    for t in range(i + 1):
        hit = any(sc >= thr[c] for c, sc, _ in dets_all[t])
        x0, x1_ = int(t * step), max(int((t + 1) * step) - 1, int(t * step) + 1)
        if hit:
            cv2.rectangle(tl, (x0, 16), (x1_, 32), (0, 120, 255), -1)
        cv2.rectangle(tl, (x0, 48), (x1_, 66), LEVEL_COLORS[levels[t]], -1)
    if onset is not None:
        cv2.line(tl, (int(onset * step), 14), (int(onset * step), 68), (255, 255, 255), 1)
    bar = np.full((30, w, 3), 20, np.uint8)
    cv2.rectangle(bar, (0, 0), (w, 30), LEVEL_COLORS[levels[i]], -1)
    cv2.putText(
        bar,
        f"alert level: {levels[i].name}",
        (6, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([img, tl, bar])


def hero_gif(clips, cfg):
    pos, neg = pick_clips(clips, cfg)
    _, lp, _ = replay(pos["dets"], cfg)
    _, ln, _ = replay(neg["dets"], cfg)
    frames = []
    for i in range(N_FRAMES):
        a = _panel(
            "positive clip (fire/smoke starts at line)", pos["dets"], lp, pos["onset"], i, cfg
        )
        b = _panel("negative clip (no fire or smoke)", neg["dets"], ln, None, i, cfg)
        sep = np.full((a.shape[0], 8, 3), 12, np.uint8)
        foot = np.full((30, a.shape[1] * 2 + 8, 3), 12, np.uint8)
        cv2.putText(
            foot,
            f"t = {i / SIM_FPS:4.1f} s   boxes: detector output, no dataset pixels shown",
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        frames.append(
            Image.fromarray(
                cv2.cvtColor(np.vstack([np.hstack([a, sep, b]), foot]), cv2.COLOR_BGR2RGB)
            )
        )
    frames[0].save(
        DOCS / "hero.gif",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
        optimize=True,
    )
    return pos["image"], neg["image"]


def fmt(v, p=3):
    return "n/a" if v is None else f"{v:.{p}f}"


def results_md(ev, tm, bench, train_log, n_counts):
    cl = ev["classes"]
    L = tm["levels"]
    fl = tm["frame_level"]
    il = ev["image_level"]
    s = [
        "# Results",
        "",
        "All numbers come from `runs/*.json` written by `emberwatch eval` / `render`; nothing here is typed by hand.",
        "",
        "## Detector (held-out D-Fire test split)",
        "",
        f"Test images: {ev['n_test']}. Model: SSDlite320 MobileNetV3-Large, COCO-pretrained, fine-tuned "
        f"{train_log['steps']} steps (batch {train_log['batch_size']}, {train_log['minutes']:.0f} min on {train_log['device']}). "
        "Thresholds were chosen per class on a 5% validation carve-out of the train split (max box-level F1), "
        "never on test.",
        "",
        "| class | AP@0.5 | threshold | precision | recall | GT boxes |",
        "|---|---|---|---|---|---|",
    ]
    for c in ("fire", "smoke"):
        s.append(
            f"| {c} | {cl[c]['ap50']:.3f} | {ev['thresholds'][c]} | {cl[c]['precision']:.3f} | "
            f"{cl[c]['recall']:.3f} | {cl[c]['n_gt']} |"
        )
    s += [
        f"| mean | {ev['map50']:.3f} | | | | |",
        "",
        "## False alarms on negative test images (image level)",
        "",
        "An image is a false alarm if any detection exceeds its class threshold. D-Fire has no scene tags, so hard "
        "negatives are mined from the negatives by colour statistics (see README, limitations).",
        "",
        "| subset | images | false-alarm images | rate |",
        "|---|---|---|---|",
    ]
    for k, name in (
        ("fa_all_negatives", "all negatives"),
        ("fa_fire_coloured", "fire-coloured (top 15% orange/red)"),
        ("fa_hazy", "hazy / low-contrast (bottom 15%)"),
    ):
        r = il[k]
        s.append(f"| {name} | {r['n_images']} | {r['false_alarm_images']} | {r['rate']:.3f} |")
    s += [
        "",
        f"Image-level recall on images that contain fire or smoke: {il['positives_detected']:.3f}.",
        "",
        "## Temporal logic on synthetic clips",
        "",
        f"Clips: {n_counts}. {tm['sim']['frames']} frames at {tm['sim']['fps']:g} fps of simulated time "
        f"(onset at frame {tm['sim']['onset']}, growth ramp {tm['sim']['ramp']} frames). "
        f"Negative footage total: {L['WARNING']['neg_hours']:.2f} h. "
        f"Default persistence: {tm['default']['min_hits']} of {tm['default']['window']} frames.",
        "",
        "| system | recall | false-alarm episodes / h | median time-to-alert (s) |",
        "|---|---|---|---|",
        f"| frame level (any detection alarms) | {fl['recall']:.3f} | {fl['fa_frames_per_hour']:.0f} frames/h "
        f"({fl['false_alarm_frame_rate']:.3f} of negative frames) | {fmt(fl['tta_median_s'], 1)} |",
    ]
    for lv in ("WATCH", "WARNING", "ALARM"):
        r = L[lv]
        s.append(
            f"| temporal, level >= {lv.lower()} | {r['recall']:.3f} | {r['fa_per_hour_all']:.1f} | "
            f"{fmt(r['tta_median_s'], 1)} |"
        )
    ng = tm["no_growth"]
    s += [
        f"| temporal, warning, growth check off | {ng['recall']:.3f} | {ng['fa_per_hour_all']:.1f} | "
        f"{fmt(ng['tta_median_s'], 1)} |",
        "",
        f"Fraction of negative clips with at least one frame-level false alarm: {fl['clip_false_alarm_rate']:.3f}.",
        "",
        "False-alarm episodes per hour at the warning level, by negative subset:",
        "",
        "| subset | episodes / h |",
        "|---|---|",
    ]
    for k in ("neg", "neg_fire_coloured", "neg_hazy"):
        s.append(f"| {k} | {fmt(L['WARNING'][f'fa_per_hour_{k}'], 1)} |")
    s += [
        "",
        "### Persistence window sweep (warning level)",
        "",
        "| M | N | recall | false-alarm episodes / h | median time-to-alert (s) | p90 (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in tm["sweep"]:
        if r["level"] == "WARNING":
            s.append(
                f"| {r['window']} | {r['min_hits']} | {r['recall']:.3f} | {r['fa_per_hour_all']:.1f} | "
                f"{fmt(r['tta_median_s'], 1)} | {fmt(r['tta_p90_s'], 1)} |"
            )
    s += [
        "",
        "### Score-threshold sweep (warning level, M=6, N=4)",
        "",
        "| threshold (both classes) | recall | false-alarm episodes / h | median time-to-alert (s) |",
        "|---|---|---|---|",
    ]
    for r in tm["thr_sweep"]:
        if r["level"] == "WARNING":
            s.append(
                f"| {r['thr']:.1f} | {r['recall']:.3f} | {r['fa_per_hour_all']:.1f} | "
                f"{fmt(r['tta_median_s'], 1)} |"
            )
    s += ["", "![trade-off](tradeoff.png)", "", "![PR curves](pr_curves.png)", ""]
    if bench:
        s += [
            "## Real-time throughput",
            "",
            f"`emberwatch watch` on a 960x540 25 fps file, device `{bench['device']}`, batch 1, {bench['frames']} frames, "
            f"4 torch threads (shared 16 GB M1 Pro):",
            "",
            f"- end-to-end (decode, resize, inference, alert logic): {bench['fps_end_to_end']:.1f} fps",
            f"- detector latency: mean {bench['latency_ms_mean']:.1f} ms, p95 {bench['latency_ms_p95']:.1f} ms",
            "",
        ]
    (DOCS / "results.md").write_text("\n".join(s))


def render(run_bench: bool = False) -> None:
    DOCS.mkdir(exist_ok=True)
    ev, tm = _load("eval.json"), _load("temporal.json")
    pr_figure(ev)
    tradeoff_figure(tm)
    clips = _load("clip_dets.json")
    thr = ev["thresholds"]
    cfg = replace(AlertConfig.for_fps(2.0), fire_thr=thr["fire"], smoke_thr=thr["smoke"])
    print("hero clips:", hero_gif(clips, cfg))
    if run_bench:
        from .watch import watch

        stats = watch(str(bench_video()), sink_spec="file:runs/bench_alerts.jsonl", max_frames=300)
        (RUNS / "bench.json").write_text(json.dumps(stats))
    bench = _load("bench.json") if (RUNS / "bench.json").exists() else None
    results_md(
        ev,
        tm,
        bench,
        _load("train_log.json"),
        ", ".join(f"{v} {k}" for k, v in tm["sim"]["n_clips"].items()),
    )
    print("gif MB:", (DOCS / "hero.gif").stat().st_size / 1e6)


def bench_video(path: Path = Path("data/bench.mp4"), seconds: int = 12) -> Path:
    """960x540 25 fps test video built from one positive clip (not committed, data/ is ignored)."""
    from .data import split_items
    from .temporal import make_clip

    ev = _load("eval.json")
    name, boxes = next((n, b) for n, b in split_items("test") if len(b) >= 2 and ev)
    frames, _ = make_clip(name, boxes, seed=1)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (960, 540))
    for i in range(seconds * 25):
        f = cv2.resize(frames[(i * len(frames)) // (seconds * 25)], (960, 540))
        w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    w.release()
    return path
