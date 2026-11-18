"""Shared fixtures for behavior-equivalence checks of the Pi optimizations.

The working tree already contained uncommitted optimizations before this task,
so git HEAD is NOT a clean baseline. Instead we prove equivalence directly:
the NEW wired fast paths are compared against the ORIGINAL unwired paths within
the same module on identical inputs, and extract_detections() is compared
against a faithful replication of the old per-box tensor extraction.

Only `ultralytics` is stubbed (not installed here, never instantiated at import);
every other optional dependency degrades to None via the script's own guards.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _stub_ultralytics():
    if "ultralytics" in sys.modules:
        return
    mod = types.ModuleType("ultralytics")

    class _YOLO:  # placeholder; never constructed during import
        def __init__(self, *a, **k):
            raise RuntimeError("YOLO should not be instantiated in tests")

    mod.YOLO = _YOLO
    sys.modules["ultralytics"] = mod


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def new():
    _stub_ultralytics()
    return _load_module("whisper_new", REPO / "whisper.py")


@pytest.fixture
def frozen_time(monkeypatch):
    """Freeze time.time so approach/age math is deterministic."""
    import time as _time

    monkeypatch.setattr(_time, "time", lambda: 1000.0)
    return 1000.0


def reset_state(mod, grid=None, now=1000.0):
    mod.track_history.clear()
    mod.distance_memory.clear()
    mod.track_memory.clear()
    mod.face_memory.clear()
    mod.emotion_history.clear()
    mod.last_emotion_print.clear()
    mod.emotion_speech_memory.clear()
    mod.last_global_alert_time = 0.0
    mod.last_cleanup_time = 0.0
    mod.last_alert_decision = {}
    mod.scene_last_trigger_monotonic = 0.0
    mod.ocr_last_trigger_monotonic = 0.0
    mod.reset_unknown_obstacle_state()
    mod.last_unknown_debug_time = 0.0
    mod._set_speech_active(False)
    for name in (
        "voice_audio_frames", "voice_audio_bytes", "voice_audio_timeouts",
        "voice_nonempty_results", "voice_commands_matched",
        "voice_commands_dispatched", "voice_direct_responses_started",
        "voice_ocr_triggers", "voice_scene_triggers", "voice_recovery_count",
        "speech_drop_count", "speech_cleared_count",
    ):
        if hasattr(mod, name):
            setattr(mod, name, 0)
    for name in ("voice_listener_heartbeat", "voice_last_audio_monotonic",
                 "voice_last_processed_monotonic", "voice_last_recovery_monotonic"):
        if hasattr(mod, name):
            setattr(mod, name, 0.0)
    if hasattr(mod, "voice_listener_stage"):
        mod.voice_listener_stage = "stopped"
    if hasattr(mod, "_set_voice_listener_source"):
        mod._set_voice_listener_source(None, None)
    if hasattr(mod, "clear_waiting_speech"):
        mod.clear_waiting_speech()
    with mod.tof_state_lock:
        mod.tof_state["grid_mm"] = grid
        mod.tof_state["status_grid"] = None
        mod.tof_state["connected"] = grid is not None
        mod.tof_state["initialized"] = grid is not None
        mod.tof_state["last_update"] = now if grid is not None else 0.0
        mod.tof_state["last_error"] = None
        mod.tof_state["frame_counter"] = 1 if grid is not None else 0


# --------------------------------------------------------------------------
# Minimal stand-ins for the ultralytics tensor/Boxes API used by the main loop.
# They reproduce the exact semantics the original per-box code relied on:
#   box.cls[0], box.conf[0], box.xyxy[0].cpu().numpy(), box.id[0] (or None)
# and the aggregate API the new extract_detections() uses:
#   boxes.xyxy.cpu().numpy(), boxes.conf/cls/id, len(boxes)
# --------------------------------------------------------------------------
class FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __getitem__(self, idx):
        val = self._arr[idx]
        return FakeTensor(val) if isinstance(val, np.ndarray) else val

    def __len__(self):
        return len(self._arr)


class FakeBox:
    def __init__(self, xyxy_row, conf, cls, tid):
        self.xyxy = FakeTensor(np.asarray(xyxy_row, dtype=np.float32).reshape(1, 4))
        self.conf = FakeTensor(np.array([conf], dtype=np.float32))
        self.cls = FakeTensor(np.array([cls], dtype=np.float32))
        self.id = (
            FakeTensor(np.array([tid], dtype=np.float32)) if tid is not None else None
        )


class FakeBoxes:
    def __init__(self, xyxy, confs, clss, ids):
        self._xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self._confs = np.asarray(confs, dtype=np.float32).reshape(-1)
        self._clss = np.asarray(clss, dtype=np.float32).reshape(-1)
        self._ids = None if ids is None else np.asarray(ids, dtype=np.float32).reshape(-1)

    @property
    def xyxy(self):
        return FakeTensor(self._xyxy)

    @property
    def conf(self):
        return FakeTensor(self._confs)

    @property
    def cls(self):
        return FakeTensor(self._clss)

    @property
    def id(self):
        return None if self._ids is None else FakeTensor(self._ids)

    def __len__(self):
        return len(self._xyxy)

    def __iter__(self):
        for i in range(len(self)):
            tid = None if self._ids is None else float(self._ids[i])
            yield FakeBox(self._xyxy[i], self._confs[i], self._clss[i], tid)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def make_results(xyxy, confs, clss, ids):
    if xyxy is None:
        return [FakeResult(None)]
    return [FakeResult(FakeBoxes(xyxy, confs, clss, ids))]


def original_parse(results, names):
    """Faithful replication of the ORIGINAL main-loop per-box extraction."""
    out = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = -1
                out.append((class_name, conf, x1, y1, x2, y2, track_id))
    return out


def original_linear_resample(samples, rate, target):
    """Faithful replication of the ORIGINAL linear-interp resample (no anti-alias).

    Used to prove the new path removes out-of-band energy that this one aliases
    back into the speech band when downsampling.
    """
    samples = np.asarray(samples, dtype=np.float32)
    out_len = max(1, int(round(samples.size * target / rate)))
    x_old = np.arange(samples.size, dtype=np.float32)
    x_new = np.linspace(0.0, float(samples.size - 1), out_len, dtype=np.float32)
    return np.interp(x_new, x_old, samples)


NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    9: "traffic light", 11: "stop sign", 13: "bench", 15: "cat", 16: "dog",
    17: "horse", 19: "cow", 24: "backpack", 26: "handbag", 28: "suitcase",
    56: "chair",
}