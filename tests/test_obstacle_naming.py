"""Naming can change the words of an approved hazard, never its decision."""
import copy
import threading

import numpy as np
import pytest

from conftest import make_results, reset_state


BOX = [250, 250, 390, 470]
UNKNOWN_KEY = "tof:unknown:in_path"


class NamingModel:
    # Sparse/non-COCO ID verifies names come from the loaded model, not a table.
    names = {123: "bed"}

    def __init__(self, box=BOX, confidence=.9):
        self.box = box
        self.confidence = confidence
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return make_results([self.box], [self.confidence], [123], None)


@pytest.mark.parametrize("label", ["bed", "toaster", "model-specific object"])
def test_all_loaded_names_can_match_without_known_name_membership(new, frozen_time, label):
    reset_state(new, np.full((8, 8), 600, dtype=np.float32))
    assert label not in new.KNOWN_OBJECT_CLASSES
    model = NamingModel()
    model.names = {123: label}
    snapshot = new.get_tof_snapshot()
    memory_before = copy.deepcopy(new.distance_memory)
    name = new.name_tof_obstacle(model, np.zeros((480, 640, 3), np.uint8), .6,
                                 snapshot, new.tof_overlap_rectangle(640, 480))
    assert name == label
    assert model.calls == [dict(verbose=False, conf=.4, imgsz=new.DETECTION_IMG_SIZE,
                                device=new.DEVICE, classes=None, save=False)]
    assert new.distance_memory == memory_before


@pytest.mark.parametrize("box,confidence,grid_mm", [
    ([0, 250, 80, 470], .99, 600),  # unrelated off-path object
    (BOX, .39, 600),               # unchanged confidence threshold
    (BOX, .99, 1800),              # depth does not explain the .6m hazard
])
def test_nonmatching_detections_do_not_supply_names(new, frozen_time, box, confidence, grid_mm):
    reset_state(new, np.full((8, 8), grid_mm, dtype=np.float32))
    assert new.name_tof_obstacle(
        NamingModel(box, confidence), np.zeros((480, 640, 3), np.uint8), .6,
        new.get_tof_snapshot(), new.tof_overlap_rectangle(640, 480),
    ) is None


def test_visual_association_fallback_uses_existing_rule(new, frozen_time, monkeypatch):
    reset_state(new, np.full((8, 8), 600, dtype=np.float32))
    # This in-corridor low box is outside the overlap, with clamping disabled.
    monkeypatch.setattr(new, "TOF_BOX_CLAMP_LOW_OBJECTS", False)
    monkeypatch.setattr(new, "raw_fake_distance_meters", lambda *args: (.7, "area_fallback"))
    assert new.name_tof_obstacle(
        NamingModel([250, 400, 390, 478]), np.zeros((480, 640, 3), np.uint8), .6,
        new.get_tof_snapshot(), new.tof_overlap_rectangle(640, 480),
    ) == "bed"


@pytest.mark.parametrize("case,expected", [
    ("matched", "Caution. bed close in path."),
    ("overlays", "Caution. bed close in path."),
    ("off_path", "Caution. unknown obstacle close in path."),
    ("empty", "Caution. unknown obstacle close in path."),
    ("prediction_error", "Caution. unknown obstacle close in path."),
    ("clone_error", "Caution. unknown obstacle close in path."),
    ("no_hazard", None),
])
def test_main_names_only_after_unchanged_tof_alert_approval(new, monkeypatch, capsys, case, expected):
    clock = [1000.0]
    monkeypatch.setattr(new.time, "time", lambda: clock[0])
    reset_state(new, np.full((8, 8), 3500 if case == "no_hazard" else 600, dtype=np.float32))
    enabled = threading.Event()
    enabled.set()
    monkeypatch.setattr(new, "detection_enabled_event", enabled)
    monkeypatch.setattr(new, "DETECTION_CLASSES", new.YOLO_CLASSES_TO_KEEP)
    monkeypatch.setattr(new, "USE_TRACKING", True)
    monkeypatch.setattr(new, "ALERT_TRACE", True)
    for setting in ("SHOW_WINDOW", "SPEAK_STARTUP_MESSAGE", "LOG_ALL_DETECTIONS",
                    "ENABLE_FACE_RECOGNITION", "ENABLE_EMOTION_RECOGNITION"):
        monkeypatch.setattr(new, setting, False)
    if case == "overlays":
        monkeypatch.setattr(new, "SHOW_WINDOW", True)
        monkeypatch.setattr(new, "draw_walking_corridor", lambda frame: frame.fill(255))
        monkeypatch.setattr(new, "draw_tof_overlay", lambda *args, **kwargs: None)
        monkeypatch.setattr(new, "draw_hud", lambda *args: None)
        monkeypatch.setattr(new.cv2, "imshow", lambda *args: None)
        monkeypatch.setattr(new.cv2, "waitKey", lambda *args: -1)
        monkeypatch.setattr(new.cv2, "destroyAllWindows", lambda: None)
    for operation in ("ensure_log_file", "close_log_file", "start_tof_worker", "stop_tof_worker",
                      "start_speech_worker", "stop_speech_worker", "start_voice_control",
                      "stop_voice_control", "stop_ocr_reader", "print_voice_summary", "log_row"):
        monkeypatch.setattr(new, operation, lambda *args, **kwargs: None)
    monkeypatch.setattr(new, "start_ocr_reader", lambda source: None)
    monkeypatch.setattr(new, "load_known_faces", lambda: ([], []))
    monkeypatch.setattr(new, "EmotionRecognizer", lambda path: None)
    # Synchronized fresh snapshot: only its grid varies across cases.
    snapshot = new.get_tof_snapshot()
    monkeypatch.setattr(new, "get_tof_snapshot", lambda: snapshot)
    tracking_calls, naming_calls, approvals, speech = [], [], [], []

    class Detector:
        names = {56: "chair", 59: "bed"}

        def __deepcopy__(self, memo):
            if case == "clone_error":
                raise RuntimeError("clone failed")
            return Detector()

        def track(self, frame, **kwargs):
            assert self is detector
            assert kwargs["classes"] == new.YOLO_CLASSES_TO_KEEP
            assert kwargs["conf"] == .4
            tracking_calls.append(camera.count)
            return []  # as before, the legacy whitelist excludes bed

        def predict(self, frame, **kwargs):
            assert self is not detector  # never reuse the live tracker/predictor
            assert approvals[-1][0] is True  # approved BEFORE naming inference
            assert kwargs["classes"] is None
            assert not frame.any()  # raw camera pixels, including with overlays on
            naming_calls.append(camera.count)
            if case == "prediction_error":
                raise RuntimeError("prediction failed")
            if case == "empty":
                return []
            box = [0, 250, 80, 470] if case == "off_path" else BOX
            return make_results([box], [.9], [59], None)

    detector = Detector()
    monkeypatch.setattr(new, "load_detection_model", lambda: detector)

    class Camera:
        count = 0
        released = False

        def is_opened(self):
            return True

        def read(self):
            self.count += 1
            clock[0] = 1000 + self.count * .15
            return (True, np.zeros((480, 640, 3), np.uint8)) if self.count <= 8 else (False, None)

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(new, "open_source", lambda: camera)
    original_should_alert = new.should_alert

    def approve(key, risk, reason, distance_m=None):
        assert key == UNKNOWN_KEY
        assert reason == "unknown obstacle close in path"
        assert distance_m == pytest.approx(.6)
        approved = original_should_alert(key, risk, reason, distance_m=distance_m)
        approvals.append((approved, key, risk, reason, distance_m))
        return approved

    def speak(risk, message, urgent, **kwargs):
        speech.append((risk, message, kwargs["alert_key"], kwargs["alert_distance_m"]))
        return True

    monkeypatch.setattr(new, "should_alert", approve)
    monkeypatch.setattr(new, "_play_safety_alert", speak)
    try:
        new.main()
        assert tracking_calls == [3, 6]  # unchanged cadence and legacy filter
        assert camera.released
        if expected is None:
            assert not speech and not naming_calls and not approvals
        else:
            assert speech == [(2, expected, UNKNOWN_KEY, pytest.approx(.6))]
            assert len(naming_calls) == (0 if case == "clone_error" else 1)
            assert sum(row[0] for row in approvals) == 1
            assert new.track_memory[UNKNOWN_KEY]["last_reason"] == "unknown obstacle close in path"
            assert new.unknown_obstacle_state["risk"] == 2
            output = capsys.readouterr().out
            assert "[alert trace]" in output and repr(expected) in output
    finally:
        reset_state(new)
