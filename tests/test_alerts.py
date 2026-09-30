from emberwatch.alerts import AlertConfig, AlertEngine, Det, Level

FIRE = [Det("fire", 0.9, (0.1, 0.1, 0.3, 0.3))]
NONE: list[Det] = []


def smoke(area_side: float) -> list[Det]:
    return [Det("smoke", 0.9, (0.0, 0.0, area_side, area_side))]


def cfg(**kw):
    base = dict(window=4, min_hits=3, alarm_hold=4, clear_frames=3, cooldown_s=10.0)
    base.update(kw)
    return AlertConfig(**base)


def run(engine, frames, dt=1.0, t0=0.0):
    out = []
    for i, f in enumerate(frames):
        a = engine.update(f, t0 + i * dt)
        if a:
            out.append((i, a.level))
    return out


def test_single_frame_blip_never_alerts():
    e = AlertEngine(cfg())
    assert run(e, [FIRE] + [NONE] * 10) == []


def test_persistence_needs_n_of_m():
    e = AlertEngine(cfg())
    # hit, miss, hit: only 2 of 4 -> no alert; third hit inside the window triggers
    assert run(e, [FIRE, NONE, FIRE]) == []
    assert run(e, [FIRE], t0=3) == [(0, Level.WARNING)]


def test_persistent_fire_reaches_warning_then_alarm_after_hold():
    e = AlertEngine(cfg())
    ev = run(e, [FIRE] * 8)
    assert ev[0] == (2, Level.WARNING)
    assert (ev[-1][1]) == Level.ALARM
    assert [lv for _, lv in ev] == [Level.WARNING, Level.ALARM]  # no repeats inside cooldown


def test_fire_and_smoke_together_is_immediate_alarm():
    e = AlertEngine(cfg())
    both = FIRE + smoke(0.3)
    assert run(e, [both] * 3) == [(2, Level.ALARM)]


def test_smoke_alone_is_watch_unless_growing():
    e = AlertEngine(cfg())
    assert run(e, [smoke(0.3)] * 6) == [(2, Level.WATCH)]
    g = AlertEngine(cfg())
    ev = run(g, [smoke(0.2), smoke(0.2), smoke(0.4), smoke(0.45), smoke(0.5), smoke(0.55)])
    assert Level.WARNING in [lv for _, lv in ev]


def test_cooldown_suppresses_repeat_then_allows():
    e = AlertEngine(cfg(alarm_hold=100, clear_frames=2))
    first = run(e, [FIRE] * 3)
    assert first == [(2, Level.WARNING)]
    # stay in WARNING: repeats suppressed until cooldown (10 s) elapses
    repeat = run(e, [FIRE] * 12, t0=3)
    assert len(repeat) == 1 and repeat[0][0] >= 9  # first repeat at t >= 12


def test_debounce_keeps_level_through_short_gap_then_clears():
    e = AlertEngine(cfg(alarm_hold=100))
    run(e, [FIRE] * 4)
    assert e.level == Level.WARNING
    run(e, [NONE], t0=4)
    assert e.level == Level.WARNING  # window still holds 3 hits; and debounce
    run(e, [NONE] * 12, t0=5)
    assert e.level == Level.CLEAR


def test_escalation_bypasses_cooldown():
    e = AlertEngine(cfg(alarm_hold=100, cooldown_s=1000.0))
    ev = run(e, [FIRE] * 3)
    assert ev == [(2, Level.WARNING)]
    both = FIRE + smoke(0.3)
    ev2 = run(e, [both] * 4, t0=3)
    assert ev2 and ev2[0][1] == Level.ALARM  # different (higher) level is not suppressed


def test_low_score_detections_ignored():
    e = AlertEngine(cfg(fire_thr=0.6))
    weak = [Det("fire", 0.4, (0, 0, 0.5, 0.5))]
    assert run(e, [weak] * 10) == []
