import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import NAMES, make_results, reset_state


SCENE_PHRASES = (
    "what's there", "what do you see", "describe", "what's around me", "what's ahead",
)
READ_PHRASES = (
    "read this", "read that", "read", "read text", "what does it say", "what's that say",
)


@pytest.mark.parametrize("position,expected", [
    ("in path", "ahead"), ("ahead", "ahead"),
    ("left side", "to your left"), ("left", "to your left"),
    ("right side", "to your right"), ("right", "to your right"),
])
def test_description_uses_spoken_position(new, position, expected):
    items = [("person", position, "very_close", 0.4, 2)]
    assert new._compose_scene_description(items) == f"Person {expected}, very close."


@pytest.mark.parametrize("level,expected", [
    ("very_close", "very close"), ("close", "close"),
    ("medium", "a few meters away"), ("far", "far away"),
])
def test_description_uses_existing_distance_levels(new, level, expected):
    items = [("chair", "left side", level, 2.0, 0)]
    assert new._compose_scene_description(items) == f"Chair to your left, {expected}."


def test_description_prioritizes_risk_then_distance_and_caps_at_three(new):
    items = [
        ("bench", "left side", "far", 6.0, 0),
        ("chair", "left side", "close", 1.0, 1),
        ("car", "right side", "medium", 3.0, 3),
        ("person", "in path", "very_close", 0.4, 2),
        ("dog", "in path", "medium", 2.0, 1),
    ]
    original = list(items)
    assert new._compose_scene_description(items) == (
        "Car to your right, a few meters away. Person ahead, very close. "
        "Chair to your left, close."
    )
    assert items == original


def test_empty_description(new):
    assert new._compose_scene_description([]) == "I don't see anything ahead."


def _reset_voice_state(new, monkeypatch):
    monkeypatch.setattr(new, "ENABLE_SCENE_DESCRIBE", True)
    monkeypatch.setattr(new, "ENABLE_OCR_READING", True)
    monkeypatch.setattr(new, "scene_describe_event", threading.Event())
    monkeypatch.setattr(new, "ocr_trigger_event", threading.Event())
    monkeypatch.setattr(new, "last_scene_describe_time", 0.0)
    monkeypatch.setattr(new, "last_ocr_trigger_time", 0.0)
    monkeypatch.setattr(new, "last_voice_command_time", 0.0)


@pytest.mark.parametrize("phrase", SCENE_PHRASES + ("  WHAT'S   THERE ",))
def test_scene_phrase_sets_only_scene_event(new, monkeypatch, frozen_time, phrase):
    _reset_voice_state(new, monkeypatch)
    assert new._apply_voice_command(phrase) is True
    assert new.scene_describe_event.is_set()
    assert not new.ocr_trigger_event.is_set()


@pytest.mark.parametrize("phrase", READ_PHRASES)
def test_read_phrase_sets_only_ocr_event(new, monkeypatch, frozen_time, phrase):
    _reset_voice_state(new, monkeypatch)
    assert new._apply_voice_command(phrase) is True
    assert new.ocr_trigger_event.is_set()
    assert not new.scene_describe_event.is_set()


def test_scene_trigger_debounces_and_accepts_later_request(new, monkeypatch):
    _reset_voice_state(new, monkeypatch)
    monkeypatch.setattr(new.time, "time", lambda: 1000.0)
    assert new._trigger_scene_description() is True
    assert new.scene_describe_event.is_set()
    new.scene_describe_event.clear()
    monkeypatch.setattr(new.time, "time", lambda: 1000.5)
    assert new._trigger_scene_description() is True
    assert not new.scene_describe_event.is_set()
    monkeypatch.setattr(new.time, "time", lambda: 1003.0)
    assert new._trigger_scene_description() is True
    assert new.scene_describe_event.is_set()


def test_scene_disabled_does_not_fall_back_to_ocr(new, monkeypatch):
    _reset_voice_state(new, monkeypatch)
    monkeypatch.setattr(new, "ENABLE_SCENE_DESCRIBE", False)
    assert new._trigger_scene_description() is False
    assert new._apply_voice_command("what's there") is False
    assert not new.scene_describe_event.is_set()
    assert not new.ocr_trigger_event.is_set()


def test_detection_commands_remain_independent(new, monkeypatch, frozen_time):
    _reset_voice_state(new, monkeypatch)
    enabled = threading.Event()
    enabled.set()
    monkeypatch.setattr(new, "detection_enabled_event", enabled)
    monkeypatch.setattr(new, "VOICE_CONFIRM_COMMANDS", False)
    assert new._apply_voice_command("what's there") is True
    assert new._apply_voice_command("detection off") is True
    assert not new.detection_is_enabled()
    monkeypatch.setattr(new.time, "time", lambda: 1003.0)
    assert new._apply_voice_command("detection on") is True
    assert new.detection_is_enabled()


@pytest.mark.parametrize("ocr_enabled", [True, False])
@pytest.mark.parametrize("scene_enabled", [True, False])
def test_grammar_enables_phrase_groups_independently(new, monkeypatch,
                                                    ocr_enabled, scene_enabled):
    monkeypatch.setattr(new, "ENABLE_OCR_READING", ocr_enabled)
    monkeypatch.setattr(new, "ENABLE_SCENE_DESCRIBE", scene_enabled)
    grammar = json.loads(new._voice_command_grammar())
    assert {"detection on", "detection off", "[unk]"}.issubset(grammar)
    for phrase in SCENE_PHRASES:
        assert (phrase in grammar) is scene_enabled
    for phrase in READ_PHRASES:
        assert (phrase in grammar) is ocr_enabled
    assert len(grammar) == len(set(grammar))


@pytest.mark.parametrize(
    "detection_enabled,describe_requested,objects,pause_during_scoring,request_frame,expected", [
        (False, True, True, False, 4, [(2, "Detection is off.")]),
        (True, True, True, False, 4, [(2, "Chair ahead, far away.")]),
        (True, True, False, False, 4, [(2, "I don't see anything ahead.")]),
        (True, False, True, False, 4, []),
        (False, False, True, False, 4, []),
        (True, True, True, True, 4, [(2, "Detection is off.")]),
        (True, True, True, False, 1, [(2, "Chair ahead, far away.")]),
        (True, True, False, False, 1, [(2, "I don't see anything ahead.")]),
        (False, True, True, False, 1, [(2, "Detection is off.")]),
    ],
)
def test_main_describes_once_without_extra_inference(new, monkeypatch, frozen_time,
                                                    detection_enabled, describe_requested,
                                                    objects, pause_during_scoring,
                                                    request_frame, expected):
    reset_state(new)
    _reset_voice_state(new, monkeypatch)
    enabled = threading.Event()
    if detection_enabled:
        enabled.set()
    monkeypatch.setattr(new, "detection_enabled_event", enabled)
    for setting in ("ENABLE_TOF", "ENABLE_FACE_RECOGNITION", "ENABLE_EMOTION_RECOGNITION",
                    "SHOW_WINDOW", "SPEAK_STARTUP_MESSAGE", "LOG_ALL_DETECTIONS"):
        monkeypatch.setattr(new, setting, False)
    for operation in ("ensure_log_file", "close_log_file", "start_tof_worker",
                      "stop_tof_worker", "start_speech_worker", "stop_speech_worker",
                      "start_voice_control", "stop_voice_control", "stop_ocr_reader"):
        monkeypatch.setattr(new, operation, lambda: None)
    monkeypatch.setattr(new, "start_ocr_reader", lambda source: None)
    monkeypatch.setattr(new, "load_known_faces", lambda: ([], []))
    monkeypatch.setattr(new, "EmotionRecognizer", lambda path: None)
    model = SimpleNamespace(names=NAMES)
    monkeypatch.setattr(new, "load_detection_model", lambda: model)
    spoken = []
    speech_frames = []

    def speak(risk, message):
        spoken.append((risk, message))
        speech_frames.append(camera.count)
        return True

    monkeypatch.setattr(new, "play_alert_sound", speak)

    class Camera:
        count = 0
        released = False

        def is_opened(self):
            return True

        def read(self):
            self.count += 1
            if self.count == request_frame and describe_requested:
                new._apply_voice_command("what's there")
            if self.count <= 6:
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            return False, None

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(new, "open_source", lambda: camera)
    inference_frames = []

    def infer(actual_model, frame):
        assert actual_model is model
        inference_frames.append(camera.count)
        if not objects:
            return make_results(None, None, None, None)
        return make_results([[300, 380, 340, 440]], [0.9], [56], [1])

    monkeypatch.setattr(new, "run_yolo", infer)
    score_risk = new.score_risk

    def score_with_pause(**kwargs):
        result = score_risk(**kwargs)
        if pause_during_scoring and camera.count == 4:
            new.set_detection_enabled(False, announce=False)
        return result

    monkeypatch.setattr(new, "score_risk", score_with_pause)
    try:
        new.main()
        assert spoken == expected
        reply_frame = max(3, request_frame) if detection_enabled else request_frame
        assert speech_frames == ([reply_frame] if describe_requested else [])
        scheduled_frames = [3] if pause_during_scoring else [3, 6]
        assert inference_frames == (scheduled_frames if detection_enabled else [])
        assert not new.scene_describe_event.is_set()
        assert not new.ocr_trigger_event.is_set()
        assert camera.released
    finally:
        reset_state(new)
