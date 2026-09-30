import json

import pytest

from emberwatch.alerts import (
    Alert,
    AlertConfig,
    AlertEngine,
    Det,
    FileSink,
    Level,
    alert_payload,
    validate_payload,
)
from emberwatch.temporal import replay


def _alert():
    e = AlertEngine(AlertConfig(window=3, min_hits=2, alarm_hold=10))
    for i in range(3):
        a = e.update([Det("fire", 0.8, (0, 0, 0.2, 0.2))], float(i))
        if a:
            return a
    raise AssertionError("no alert")


def test_payload_matches_schema_and_is_json():
    p = alert_payload(_alert(), "cam-7")
    validate_payload(p)
    assert json.loads(json.dumps(p)) == p
    assert p["camera_id"] == "cam-7" and p["level"] == "warning" and p["level_value"] == 2
    assert "Not a certified" in p["disclaimer"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("level"),
        lambda p: p.update(level="panic"),
        lambda p: p.update(level_value=3),
        lambda p: p.update(frame_index="3"),
        lambda p: p["evidence"].pop("smoke_growth"),
    ],
)
def test_invalid_payloads_rejected(mutate):
    p = alert_payload(_alert())
    mutate(p)
    with pytest.raises(ValueError):
        validate_payload(p)


def test_file_sink_appends_jsonl(tmp_path):
    sink = FileSink(str(tmp_path / "a.jsonl"))
    p = alert_payload(_alert())
    sink.send(p)
    sink.send(p)
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0]) == p


class FakeDetector:
    """Stands in for the neural net: returns scripted detections per frame."""

    def __init__(self, script):
        self.script, self.i = script, 0

    def predict(self, frames):
        out = []
        for _ in frames:
            out.append(self.script[self.i])
            self.i += 1
        return out


def test_pipeline_with_mocked_detector(tmp_path):
    hit = [("fire", 0.9, [0.1, 0.1, 0.4, 0.4])]
    det = FakeDetector([[]] * 3 + [hit] * 8)
    frames_dets = [det.predict([None])[0] for _ in range(11)]
    alerts, levels, up = replay(frames_dets, AlertConfig(window=4, min_hits=3, alarm_hold=100))
    assert [a.level for a in alerts] == [Level.WARNING]
    assert up == [(5, Level.WARNING)]
    assert max(levels) == Level.WARNING
    assert isinstance(alerts[0], Alert)
