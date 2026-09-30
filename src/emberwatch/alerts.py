"""Temporal alert logic: per-frame detections -> persistence filter -> growth check ->
alert levels with escalation, de-escalation debounce and cooldown -> JSON payload -> sink.

The state machine is deterministic and independent of the detector and of wall-clock time:
the caller passes a timestamp (seconds) with every frame.
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Protocol

SCHEMA_VERSION = "1.0"


class Level(IntEnum):
    CLEAR = 0
    WATCH = 1
    WARNING = 2
    ALARM = 3


@dataclass(frozen=True)
class Det:
    cls: str  # "fire" | "smoke"
    score: float
    box: tuple[float, float, float, float]  # normalised [x1, y1, x2, y2]

    @property
    def area(self) -> float:
        return max(self.box[2] - self.box[0], 0.0) * max(self.box[3] - self.box[1], 0.0)


@dataclass
class AlertConfig:
    fire_thr: float = 0.5  # min score for a fire detection to count as evidence
    smoke_thr: float = 0.5
    window: int = 6  # M: frames in the sliding persistence window
    min_hits: int = 4  # N: frames within the window that must contain the class
    growth_ratio: float = 1.25  # smoke area in newer half / older half of window
    growth_min_area: float = 0.005  # ignore growth of negligible smoke (fraction of frame)
    alarm_hold: int = 6  # consecutive WARNING frames before it becomes ALARM
    clear_frames: int = 10  # consecutive lower-level frames before de-escalating
    cooldown_s: float = 30.0  # min seconds between repeat alerts at the same or lower level

    @classmethod
    def for_fps(cls, fps: float, window_s: float = 3.0, **kw) -> AlertConfig:
        """Scale the frame-count parameters to a processing rate (persistence over `window_s`)."""
        m = max(1, round(window_s * fps))
        return cls(
            window=m,
            min_hits=max(1, math.ceil(m * 0.67)),
            alarm_hold=m,
            clear_frames=max(1, round(5 * fps)),
            **kw,
        )


@dataclass
class Alert:
    level: Level
    timestamp: float
    frame_index: int
    reason: str
    evidence: dict = field(default_factory=dict)

    def to_payload(self, camera_id: str = "cam-0") -> dict:
        return alert_payload(self, camera_id)


def alert_payload(alert: Alert, camera_id: str = "cam-0") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "emberwatch.alert",
        "camera_id": camera_id,
        "level": alert.level.name.lower(),
        "level_value": int(alert.level),
        "timestamp_s": round(alert.timestamp, 3),
        "frame_index": alert.frame_index,
        "reason": alert.reason,
        "evidence": alert.evidence,
        "disclaimer": "Not a certified fire detection device.",
    }


PAYLOAD_REQUIRED = {
    "schema_version": str,
    "type": str,
    "camera_id": str,
    "level": str,
    "level_value": int,
    "timestamp_s": (int, float),
    "frame_index": int,
    "reason": str,
    "evidence": dict,
}


def validate_payload(p: dict) -> None:
    """Raise ValueError if the payload does not match the documented schema."""
    for k, t in PAYLOAD_REQUIRED.items():
        if k not in p:
            raise ValueError(f"missing key: {k}")
        if not isinstance(p[k], t) or isinstance(p[k], bool):
            raise ValueError(f"bad type for {k}: {type(p[k]).__name__}")
    if p["type"] != "emberwatch.alert":
        raise ValueError("bad type value")
    if p["level"] not in ("watch", "warning", "alarm"):
        raise ValueError(f"bad level: {p['level']}")
    if Level[p["level"].upper()] != p["level_value"]:
        raise ValueError("level and level_value disagree")
    for k in (
        "fire_hits",
        "smoke_hits",
        "window",
        "smoke_growth",
        "max_fire_score",
        "max_smoke_score",
    ):
        if k not in p["evidence"]:
            raise ValueError(f"missing evidence key: {k}")


class AlertEngine:
    def __init__(self, cfg: AlertConfig | None = None):
        self.cfg = cfg or AlertConfig()
        self.reset()

    def reset(self) -> None:
        c = self.cfg
        self._fire: deque[bool] = deque(maxlen=c.window)
        self._smoke: deque[bool] = deque(maxlen=c.window)
        self._area: deque[float] = deque(maxlen=c.window)
        self.level = Level.CLEAR
        self._warn_run = 0
        self._low_run = 0
        self._last_emit: dict[Level, float] = {}
        self._frame = -1

    def _growing(self) -> bool:
        a = list(self._area)
        if len(a) < self.cfg.window:
            return False
        h = len(a) // 2
        old, new = sum(a[:h]) / h, sum(a[h:]) / (len(a) - h)
        return new >= self.cfg.growth_min_area and new >= old * self.cfg.growth_ratio

    def raw_level(self) -> tuple[Level, str]:
        c = self.cfg
        fire_p = sum(self._fire) >= c.min_hits
        smoke_p = sum(self._smoke) >= c.min_hits
        growing = smoke_p and self._growing()
        if fire_p and smoke_p:
            return Level.ALARM, "persistent fire and smoke"
        if fire_p:
            return Level.WARNING, "persistent fire"
        if growing:
            return Level.WARNING, "persistent smoke with growing area"
        if smoke_p:
            return Level.WATCH, "persistent smoke"
        return Level.CLEAR, ""

    def update(self, dets: list[Det], timestamp: float) -> Alert | None:
        """Feed one frame. Returns an Alert if one should be emitted for this frame."""
        c = self.cfg
        self._frame += 1
        fire = [d for d in dets if d.cls == "fire" and d.score >= c.fire_thr]
        smoke = [d for d in dets if d.cls == "smoke" and d.score >= c.smoke_thr]
        self._fire.append(bool(fire))
        self._smoke.append(bool(smoke))
        self._area.append(sum(d.area for d in smoke))

        raw, reason = self.raw_level()
        if raw == Level.WARNING:
            self._warn_run += 1
            if self._warn_run >= c.alarm_hold:
                raw, reason = Level.ALARM, reason + f" sustained {self._warn_run} frames"
        else:
            self._warn_run = 0

        emit = None
        if raw > self.level:  # escalate immediately
            self.level, self._low_run = raw, 0
            emit = self._maybe_emit(raw, reason, timestamp, escalation=True)
        elif raw < self.level:  # de-escalate only after a debounce period
            self._low_run += 1
            if self._low_run >= c.clear_frames:
                self.level, self._low_run = raw, 0
        else:
            self._low_run = 0
            if raw > Level.CLEAR:
                emit = self._maybe_emit(raw, reason, timestamp, escalation=False)
        if emit:
            emit.frame_index = self._frame
            emit.evidence = {
                "fire_hits": sum(self._fire),
                "smoke_hits": sum(self._smoke),
                "window": c.window,
                "smoke_growth": self._growing(),
                "max_fire_score": round(max([d.score for d in fire], default=0.0), 3),
                "max_smoke_score": round(max([d.score for d in smoke], default=0.0), 3),
            }
        return emit

    def _maybe_emit(self, level: Level, reason: str, t: float, escalation: bool) -> Alert | None:
        """Escalations to a level never emitted in the cooldown window always fire; repeats of a
        level (or lower) are suppressed until the cooldown has elapsed."""
        last = self._last_emit.get(level)
        if last is not None and t - last < self.cfg.cooldown_s:
            return None
        self._last_emit[level] = t
        return Alert(level, t, self._frame, reason)


class Sink(Protocol):
    def send(self, payload: dict) -> None: ...


class StdoutSink:
    def send(self, payload: dict) -> None:
        print(json.dumps(payload), file=sys.stdout, flush=True)


class FileSink:
    def __init__(self, path: str):
        self.path = path

    def send(self, payload: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(payload) + "\n")


class WebhookSink:
    def __init__(self, url: str, timeout: float = 3.0):
        self.url, self.timeout = url, timeout

    def send(self, payload: dict) -> None:
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=self.timeout).close()
        except OSError as e:  # never let a dead endpoint stop the detection loop
            print(f"webhook failed: {e}", file=sys.stderr)


def make_sink(spec: str) -> Sink:
    """'stdout' | 'file:<path>' | 'webhook' (URL read from EMBERWATCH_WEBHOOK_URL)."""
    if spec == "stdout":
        return StdoutSink()
    if spec.startswith("file:"):
        return FileSink(spec[5:])
    if spec == "webhook":
        url = os.environ.get("EMBERWATCH_WEBHOOK_URL")
        if not url:
            raise SystemExit("set EMBERWATCH_WEBHOOK_URL for the webhook sink")
        return WebhookSink(url)
    raise SystemExit(f"unknown sink: {spec}")


__all__ = ["Alert", "AlertConfig", "AlertEngine", "Det", "Level", "asdict"]
