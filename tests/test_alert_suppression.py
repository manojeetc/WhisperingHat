"""Regression tests for generic ToF obstacle deduplication.

These exercise the decision logic directly. Real sensor noise, mounting and
thermal behavior still have to be measured on the Pi with the benchmark/log
commands in docs/PI_VOICE_ALERTS_VISION.md.
"""
import time as _time

import numpy as np
import pytest

from conftest import reset_state

UNKNOWN_KEY = "tof:unknown:in_path"
NOTICE = "unknown obstacle in path"
CAUTION = "unknown obstacle close in path"
STOP = "unknown obstacle very close in path"


def _set_time(monkeypatch, value):
    monkeypatch.setattr(_time, "time", lambda: value)


def _feed(monkeypatch, new, now, risk, reason, distance):
    _set_time(monkeypatch, now)
    new.update_object_memory(UNKNOWN_KEY, risk, reason, distance_m=distance)
    return new.should_alert(UNKNOWN_KEY, risk, reason, distance_m=distance)


def test_first_unknown_obstacle_speaks_once_then_dedupes(new, monkeypatch):
    reset_state(new)
    assert _feed(monkeypatch, new, 1000.0, 1, NOTICE, 1.9) is False   # stabilizing
    assert _feed(monkeypatch, new, 1000.4, 1, NOTICE, 1.9) is True    # first warning
    assert _feed(monkeypatch, new, 1000.8, 1, NOTICE, 1.9) is False   # same hazard, no repeat
    assert _feed(monkeypatch, new, 1001.2, 1, NOTICE, 1.8) is False   # small jitter


def test_unknown_band_reason_change_does_not_speak(new, monkeypatch):
    reset_state(new)
    assert _feed(monkeypatch, new, 1000.0, 1, NOTICE, 1.9) is False
    assert _feed(monkeypatch, new, 1000.4, 1, NOTICE, 1.9) is True

    # The same key crossing a band name without a risk change is not a new
    # obstacle. This was the source of repeated "unknown obstacle" chatter.
    assert _feed(monkeypatch, new, 1006.0, 1, "unknown obstacle in path (jitter)", 1.8) is False
    assert new.last_alert_decision["suppressed_by"] == "deduplicated"


def test_materially_closer_same_band_bypasses_suppression(new, monkeypatch):
    reset_state(new)
    assert _feed(monkeypatch, new, 1000.0, 1, NOTICE, 2.2) is False
    assert _feed(monkeypatch, new, 1000.4, 1, NOTICE, 2.2) is True
    # Still notice, but 0.8 m closer: a real approaching hazard must speak.
    assert _feed(monkeypatch, new, 1000.6, 1, NOTICE, 1.4) is True
    assert new.last_alert_decision["materially_closer"] is True


def test_risk_escalation_speaks_after_short_gap(new, monkeypatch):
    reset_state(new)
    assert _feed(monkeypatch, new, 1000.0, 1, NOTICE, 2.0) is False
    assert _feed(monkeypatch, new, 1000.4, 1, NOTICE, 2.0) is True
    # Not enough time for the full 5 s unknown-obstacle gap, but this is a
    # sustained escalation and must not be hidden.
    assert _feed(monkeypatch, new, 1002.3, 2, CAUTION, 1.0) is False   # escalation stabilizes
    assert _feed(monkeypatch, new, 1002.6, 2, CAUTION, 1.0) is True
    assert new.last_alert_decision["escalated"] is True


def test_band_flapping_inside_escalation_gap_is_suppressed(new, monkeypatch):
    reset_state(new)
    assert _feed(monkeypatch, new, 1000.0, 1, NOTICE, 2.0) is False
    assert _feed(monkeypatch, new, 1000.4, 1, NOTICE, 2.0) is True
    # 3 -> 2 -> 3-style chatter inside the 1.5 s escalation window is ignored.
    assert _feed(monkeypatch, new, 1001.5, 2, CAUTION, 1.9) is False   # starts stabilizing
    assert _feed(monkeypatch, new, 1001.8, 2, CAUTION, 1.9) is False   # 1.4 s < 1.5 s
    assert new.last_alert_decision["suppressed_by"] == "global_gap"


def test_new_hazard_after_path_was_clear_speaks(new, monkeypatch):
    reset_state(new)
    assert _feed(monkeypatch, new, 1000.0, 1, NOTICE, 1.8) is False
    assert _feed(monkeypatch, new, 1000.4, 1, NOTICE, 1.8) is True

    _set_time(monkeypatch, 4300.0)
    new.cleanup_stale_memory(4300.0)
    assert UNKNOWN_KEY not in new.track_memory

    assert _feed(monkeypatch, new, 4300.0, 2, CAUTION, 1.0) is False   # new memory stabilizes
    assert _feed(monkeypatch, new, 4300.4, 2, CAUTION, 1.0) is True    # new hazard speaks


def test_empty_path_has_no_unknown_obstacle(new, monkeypatch):
    grid = np.full((8, 8), np.nan, dtype=np.float32)
    reset_state(new, grid=grid)
    _set_time(monkeypatch, 1000.0)
    assert new.tof_unknown_path_distance() is None


def test_single_outlier_cell_does_not_dominate_median(new, monkeypatch):
    grid = np.full((8, 8), np.nan, dtype=np.float32)
    grid[2:8, 2:6] = 2000.0
    grid[3, 2:5] = 50.0        # a few spurious near cells, as in the Pi log
    grid.setflags(write=False)
    reset_state(new, grid=grid)
    _set_time(monkeypatch, 1000.0)

    result = new.tof_unknown_path_distance()
    assert result is not None
    assert result["distance_m"] == pytest.approx(2.0, abs=0.01)
    assert result["spread_m"] > 1.0


def test_too_few_valid_cells_is_ignored(new, monkeypatch):
    grid = np.full((8, 8), np.nan, dtype=np.float32)
    grid[3, 3] = 50.0
    grid.setflags(write=False)
    reset_state(new, grid=grid)
    _set_time(monkeypatch, 1000.0)
    assert new.tof_unknown_path_distance() is None


def test_unknown_distance_smoothing_and_reset(new, monkeypatch):
    reset_state(new)
    assert new.smooth_unknown_path_distance(2.0, now=1000.0) == 2.0
    smoothed = new.smooth_unknown_path_distance(0.5, now=1000.1)
    assert smoothed == pytest.approx(1.475)
    # A stale gap starts over instead of carrying an old obstacle forever.
    assert new.smooth_unknown_path_distance(0.8, now=1002.0) == 0.8


def test_unknown_risk_hysteresis_prevents_band_chatter(new):
    # Escalating into caution uses the normal threshold.
    assert new.unknown_obstacle_risk(1.20, previous_risk=1)[0] == 2
    # De-escalating out of caution needs to clear the hysteresis margin.
    assert new.unknown_obstacle_risk(1.30, previous_risk=2)[0] == 2
    assert new.unknown_obstacle_risk(1.40, previous_risk=2)[0] == 1


def test_generic_alert_keeps_known_object_behavior(new, monkeypatch):
    """A non-unknown key still speaks on a meaningful reason change."""
    reset_state(new)
    key = "zone:person:in path"
    _set_time(monkeypatch, 1000.0)
    new.update_object_memory(key, 1, "person in path")
    assert new.should_alert(key, 1, "person in path") is False
    _set_time(monkeypatch, 1000.4)
    assert new.should_alert(key, 1, "person in path") is True
    _set_time(monkeypatch, 0.0)  # deterministic: bypass the global gap
    new.last_global_alert_time = 0.0
    _set_time(monkeypatch, 1000.8)
    assert new.should_alert(key, 1, "person right in path") is True


def test_only_changed_stop_is_urgent(new):
    # A new, escalated, or materially closer STOP interrupts a reading.
    assert new.safety_alert_is_urgent(3, {"new": True}) is True
    assert new.safety_alert_is_urgent(3, {"escalated": True}) is True
    assert new.safety_alert_is_urgent(3, {"materially_closer": True}) is True
    # An unchanged repeat of the same stationary STOP queues normally.
    assert new.safety_alert_is_urgent(
        3, {"new": False, "escalated": False,
            "materially_closer": False, "repeat": True}
    ) is False
    assert new.safety_alert_is_urgent(3, {}) is False
    assert new.safety_alert_is_urgent(2, {"new": True}) is False
    assert new.safety_alert_is_urgent(1, {"escalated": True}) is False


def test_queued_repeat_is_dropped_when_hazard_is_stale_or_gone(new):
    reset_state(new)
    mem = new.update_object_memory(UNKNOWN_KEY, 3, STOP, distance_m=0.3)
    now = mem["last_seen"]

    assert new.hazard_still_present(
        UNKNOWN_KEY, 3, now=now + 0.5
    ) is True
    assert new.hazard_still_present(
        UNKNOWN_KEY, 3, now=now + 5.0
    ) is False
    assert new.hazard_still_present(
        UNKNOWN_KEY, 2, now=now + 0.5
    ) is True
    # A queued STOP must not speak after the current risk has dropped to caution.
    new.update_object_memory(UNKNOWN_KEY, 2, CAUTION, distance_m=1.0)
    assert new.hazard_still_present(
        UNKNOWN_KEY, 3, now=now + 0.5
    ) is False


def test_force_interrupt_stops_protected_playback(new, monkeypatch):
    class FakeProc:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = FakeProc()
    monkeypatch.setattr(new, "current_audio_process", process)
    monkeypatch.setattr(new, "current_audio_protected", True)

    new.interrupt_current_speech(force=True)
    assert process.terminated is True


def test_stationary_stop_repeat_queues_but_closer_stop_interrupts(new, monkeypatch):
    reset_state(new)
    interrupts = []
    monkeypatch.setattr(new, "start_speech_worker", lambda: None)
    monkeypatch.setattr(
        new,
        "interrupt_current_speech",
        lambda force=False: interrupts.append(force),
    )

    _set_time(monkeypatch, 1000.0)
    new.update_object_memory(UNKNOWN_KEY, 3, STOP, distance_m=0.8)
    assert new.should_alert(UNKNOWN_KEY, 3, STOP, distance_m=0.8) is False
    _set_time(monkeypatch, 1000.2)
    assert new.should_alert(UNKNOWN_KEY, 3, STOP, distance_m=0.8) is True
    assert new.safety_alert_is_urgent(3, new.last_alert_decision) is True

    # Unchanged stationary repeat: no interrupt, normal queue entry.
    _set_time(monkeypatch, 1031.0)
    new.update_object_memory(UNKNOWN_KEY, 3, STOP, distance_m=0.8)
    assert new.should_alert(UNKNOWN_KEY, 3, STOP, distance_m=0.8) is True
    assert new.last_alert_decision["repeat"] is True
    assert new.safety_alert_is_urgent(3, new.last_alert_decision) is False
    assert new.play_alert_sound(
        3, STOP, urgent=False, alert_key=UNKNOWN_KEY, alert_repeat=True
    )
    assert interrupts == []

    # Same hazard, now materially closer: urgent interrupt.
    _set_time(monkeypatch, 1031.5)
    new.update_object_memory(UNKNOWN_KEY, 3, STOP, distance_m=0.2)
    assert new.should_alert(UNKNOWN_KEY, 3, STOP, distance_m=0.2) is True
    assert new.last_alert_decision["materially_closer"] is True
    assert new.safety_alert_is_urgent(3, new.last_alert_decision) is True
    assert new.play_alert_sound(3, STOP, urgent=True)
    assert interrupts == [True]


def test_unchanged_stop_queues_without_interrupting(new, monkeypatch):
    reset_state(new)
    calls = []
    monkeypatch.setattr(new, "start_speech_worker", lambda: None)
    monkeypatch.setattr(
        new,
        "interrupt_current_speech",
        lambda force=False: calls.append(force),
    )

    assert new.play_alert_sound(
        3,
        "STOP. obstacle very close.",
        urgent=False,
        alert_key=UNKNOWN_KEY,
        alert_repeat=True,
    )
    assert calls == []

    item = new.speech_queue.get_nowait()
    new.speech_queue.task_done()
    assert item[0] == -3
    assert item[2] == "STOP. obstacle very close."
    assert item[7] == 3
    assert item[9] is True


def test_speech_worker_rechecks_repeat_presence(new, monkeypatch):
    reset_state(new)
    spoken = []
    monkeypatch.setattr(
        new, "_speak_message_now",
        lambda message, origin_perf, origin_label: spoken.append(message),
    )

    # No track_memory entry: the queued repeat must be dropped.
    new.speech_queue.put_nowait((
        -3, 1, STOP, True, None, "hazard",
        UNKNOWN_KEY, 3, 0.3, True,
    ))
    new.speech_queue.put_nowait((999, 2, None, False, None, None))
    new._speech_worker()
    assert spoken == []

    # Fresh memory at the same risk: the repeat may speak.
    new.update_object_memory(UNKNOWN_KEY, 3, STOP, distance_m=0.3)
    new.speech_queue.put_nowait((
        -3, 3, STOP, True, None, "hazard",
        UNKNOWN_KEY, 3, 0.3, True,
    ))
    new.speech_queue.put_nowait((999, 4, None, False, None, None))
    new._speech_worker()
    assert spoken == [STOP]


def test_urgent_play_calls_force_interrupt(new, monkeypatch):
    reset_state(new)
    calls = {}
    monkeypatch.setattr(new, "start_speech_worker", lambda: None)
    monkeypatch.setattr(
        new,
        "interrupt_current_speech",
        lambda force=False: calls.update(force=force),
    )

    assert new.play_alert_sound(3, "STOP. obstacle very close.", urgent=True)
    assert calls["force"] is True

    while not new.speech_queue.empty():
        new.speech_queue.get_nowait()
        new.speech_queue.task_done()
