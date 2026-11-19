"""Reproduce the stationary-chair, clipped-page and repeated-command failures."""
import threading
from types import SimpleNamespace

import pytest

from conftest import reset_state


def test_visual_box_growth_cannot_claim_approach_but_close_still_warns(new, monkeypatch):
    reset_state(new)
    monkeypatch.setattr(new, "tof_distance_for_box", lambda **kw: None)
    monkeypatch.setattr(new, "estimate_motion", lambda tid: (0, 0, 10., 0))
    values = iter([(2., "calibrated_height"), (1.8, "calibrated_height"), (.3, "calibrated_height")])
    monkeypatch.setattr(new, "raw_fake_distance_meters", lambda **kw: next(values))
    kwargs = dict(class_name="chair", cx=320, cy=360, x1=220, y1=200, x2=420, y2=480,
                  frame_width=640, frame_height=480, track_id=1, distance_key="track:1",
                  update_distance=True, in_corridor=True, corridor_position="in path")
    for now in (1000., 1001.):
        monkeypatch.setattr(new.time, "time", lambda: now)
        risk, reason, debug = new.score_risk(**kwargs)
        assert "closer" not in reason and "approach" not in reason
        assert debug["ttc_seconds"] == float("inf")
    new.distance_memory.clear()
    risk, reason, debug = new.score_risk(**kwargs)
    assert risk == 3 and "very close" in reason


def test_sustained_depth_trend_resets_on_source_track_gap_and_duplicate(new, monkeypatch):
    reset_state(new)
    def sample(stamp, distance, key="track:1", source="tof", track=1):
        monkeypatch.setattr(new.time, "time", lambda: stamp)
        new.update_distance_memory(key, distance, source=source)
        return new.sustained_closing_speed(key, distance, source, track,
            dict(fresh=True, last_update=stamp), True)
    assert [sample(1000 + i * .25, 2 - .15 * i) for i in range(3)] == [0, 0, 0]
    assert sample(1000.75, 1.55) == pytest.approx(.6)
    # One repeated sensor frame cannot fabricate another derivative.
    assert sample(1000.75, .01) == pytest.approx(.6)
    assert sample(1001., 1.4, key="track:2", track=2) == 0
    assert sample(1001., 1.4, source="calibrated_height") == 0
    assert sample(1001.25, 1.2) == 0
    assert sample(1003., .5) == 0
    assert sample(1003.25, .4, track=-1) == 0


def test_unchanged_unknown_notice_dedupes_but_urgent_escalation_survives(new, monkeypatch):
    reset_state(new)
    key = "tof:unknown:in_path"
    for now in (1000., 1001., 1100.):
        monkeypatch.setattr(new.time, "time", lambda: now)
        new.update_object_memory(key, 1, "unknown obstacle in path", distance_m=1.8)
        spoke = new.should_alert(key, 1, "unknown obstacle in path", 1.8)
    assert not spoke
    for now in (1101., 1102.):
        monkeypatch.setattr(new.time, "time", lambda: now)
        new.update_object_memory(key, 3, "unknown obstacle very close in path", distance_m=.2)
        spoke = new.should_alert(key, 3, "unknown obstacle very close in path", .2)
    assert spoke


def test_logged_garbled_clipped_page_is_rejected(new):
    metrics = dict(word_count=87, mean_conf=69.4, good_word_ratio=.701,
                   alpha_ratio=.968, vowel_word_ratio=.92, common_word_ratio=.414,
                   text_box=(97, 399, 854, 768))
    assert new._ocr_quality_ok(metrics)  # Reproduces the previous acceptance.
    assert not new._ocr_page_reliable(metrics, (768, 1024, 3))
    metrics.update(text_box=(97, 100, 854, 700))
    assert not new._ocr_page_reliable(metrics, (768, 1024, 3))
    metrics.update(mean_conf=95., good_word_ratio=.98)
    assert new._ocr_page_reliable(metrics, (768, 1024, 3))


def test_chair_band_and_direction_jitter_does_not_repeat_caution(new, monkeypatch):
    reset_state(new)
    key = "zone:chair:in path"
    def feed(stamp, risk, reason, distance):
        monkeypatch.setattr(new.time, "time", lambda: stamp)
        new.update_object_memory(key, risk, reason, distance)
        return new.should_alert(key, risk, reason, distance)
    assert not feed(1000., 1, "chair in path", 1.4)
    assert feed(1001., 1, "chair in path", 1.4)
    assert not feed(1002., 2, "chair close in path", 1.2)
    assert feed(1003., 2, "chair close in path", 1.2)  # First real escalation remains.
    assert not feed(1010., 1, "chair in path", 1.4)
    assert not feed(1011., 2, "chair close right in path", 1.2)
    assert not feed(1100., 2, "chair close in path", 1.2)
    assert feed(1101., 2, "chair close in path", .6)  # Materially closer still speaks.
    assert not feed(1102., 3, "chair very close in path", .2)
    assert feed(1103., 3, "chair very close in path", .2)


def test_repeated_read_and_describe_reach_workers_after_failure_with_detection_off(new, monkeypatch):
    reset_state(new)
    new.ocr_stop_event.clear()
    new.ocr_trigger_event.clear()
    new.scene_describe_event.clear()
    monkeypatch.setattr(new, "last_ocr_trigger_time", 0.)
    monkeypatch.setattr(new, "last_scene_describe_time", 0.)
    monkeypatch.setattr(new, "caption_worker_thread", None)
    spoken, reads = [], []
    monkeypatch.setattr(new, "_play_direct_response", lambda text, *a, **kw: spoken.append(text) or True)
    def read(source):
        reads.append(True)
        raise ValueError("unreadable page")
    monkeypatch.setattr(new, "_perform_ocr_read", read)
    monkeypatch.setattr(new, "caption_model", SimpleNamespace(request=lambda *a: dict(text="A chair in a room.")))
    monkeypatch.setattr(new, "VOICE_CONFIRM_COMMANDS", False)
    new.set_detection_enabled(False, announce=False)
    worker = threading.Thread(target=new._ocr_reader_worker)
    worker.start()
    try:
        for _ in range(3):
            assert new._apply_voice_command("read this")
            # Synchronize on worker completion, not a guessed sleep duration.
            deadline = __import__('time').monotonic() + 2
            while new.ocr_request_busy.is_set():
                assert __import__('time').monotonic() < deadline
                threading.Event().wait(.001)
            assert new._apply_voice_command("describe")
            assert new.scene_describe_event.is_set()
            new.scene_describe_event.clear()
            new._respond_to_scene(None, "Fallback.", 1)
            new.caption_worker_thread.join(2)
            assert not new.scene_request_busy.is_set()
        assert len(reads) == 3
        assert spoken.count("Reading.") == spoken.count("Looking.") == 3
        assert spoken.count(new.OCR_FAILURE_MESSAGE) == 3
        assert spoken.count("A chair in a room.") == 3
    finally:
        new.ocr_stop_event.set()
        new.ocr_trigger_event.set()
        worker.join(2)
        new.ocr_stop_event.clear()
        new.ocr_trigger_event.clear()


def test_busy_request_is_spoken_and_does_not_overwrite_origin(new, monkeypatch):
    reset_state(new)
    new.ocr_trigger_event.clear()
    spoken = []
    monkeypatch.setattr(new, "_play_direct_response", lambda text, **kw: spoken.append(text) or True)
    assert new._trigger_ocr_reading()
    origin = new.ocr_last_trigger_monotonic
    assert new._trigger_ocr_reading()
    assert new.ocr_last_trigger_monotonic == origin
    assert spoken == ["Reading.", "I'm still reading. Please wait."]
    new.ocr_trigger_event.clear()
    new.ocr_request_busy.clear()


def test_new_names_do_not_expand_hazard_policy(new):
    assert {"book", "pillow"}.issubset(new.OPEN_VOCAB_CLASSES)
    assert not {"book", "pillow"} & new.STATIC_OBSTACLE_CLASSES
