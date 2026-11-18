"""Capture-probe ordering and cleanup checks; no Pi hardware required."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import ocr_focus_probe as probe


CONTROLS = SimpleNamespace(
    AfStateEnum=SimpleNamespace(Idle=0, Scanning=1, Focused=2, Failed=3),
    AfModeEnum=SimpleNamespace(Auto=1, Continuous=2),
    AfTriggerEnum=SimpleNamespace(Start=0),
)


class Request:
    def __init__(self, state, lens=1.0, frame=None):
        self.metadata = {"AfState": state, "LensPosition": lens}
        self.frame = frame
        self.released = False

    def get_metadata(self):
        return self.metadata

    def make_array(self, name):
        assert name == "main"
        if isinstance(self.frame, Exception):
            raise self.frame
        return self.frame

    def release(self):
        assert not self.released
        self.released = True


def test_focus_wait_ignores_old_focused_frame_until_scan():
    requests = [Request(2), Request(1, 3.0), Request(2, 4.0)]
    stream = iter(requests)
    camera = SimpleNamespace(capture_request=lambda: next(stream))
    result, outcome = probe.focus_request(camera, "test", CONTROLS, require_scan=True)
    assert result is requests[-1] and outcome == "focused"
    assert [r.released for r in requests] == [True, True, False]
    result.release()


@pytest.mark.parametrize("state,outcome", [(None, "metadata-unavailable"), (3, "failed")])
def test_focus_wait_reports_missing_or_failed(state, outcome):
    request = Request(state)
    result, actual = probe.focus_request(
        SimpleNamespace(capture_request=lambda: request), "test", CONTROLS
    )
    assert actual == outcome and result is request
    result.release()


def test_focus_wait_times_out_and_releases_discarded_frames(monkeypatch):
    requests = [Request(0), Request(0)]
    stream = iter(requests)
    ticks = iter([0.0, 1.0, 7.0])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    result, outcome = probe.focus_request(
        SimpleNamespace(capture_request=lambda: next(stream)), "test", CONTROLS
    )
    assert outcome == "timeout" and result is requests[-1]
    assert requests[0].released
    result.release()


@pytest.mark.parametrize("broken_image", [False, True])
def test_settled_capture_observes_switch_then_triggers_and_restores(broken_image, capsys):
    frame = RuntimeError("image failed") if broken_image else np.zeros((8, 8, 3), np.uint8)
    requests = [Request(0, 0.0), Request(1, 2.0), Request(2, 5.0, frame)]
    stream = iter(requests)
    calls = []
    camera = SimpleNamespace(
        camera_configuration=lambda: "preview",
        switch_mode=lambda config: calls.append(("switch", config)),
        set_controls=lambda controls: calls.append(("controls", controls)),
        capture_request=lambda: next(stream),
    )
    source = SimpleNamespace(picam2=camera, io_lock=threading.Lock())
    if broken_image:
        with pytest.raises(RuntimeError, match="image failed"):
            probe.capture_attempt(source, "still", True, CONTROLS)
    else:
        assert probe.capture_attempt(source, "still", True, CONTROLS) is frame
    assert calls == [
        ("switch", "still"), ("controls", {"AfMode": 1, "AfTrigger": 0}),
        ("switch", "preview"), ("controls", {"AfMode": 2}),
    ]
    assert all(r.released for r in requests)
    log = capsys.readouterr().out
    assert '"stage": "settled-first-still-before-trigger", "AfMode": null, "AfState": 0' in log
    assert '"LensPosition": 5.0' in log and '"outcome": "focused"' in log


def test_immediate_capture_does_not_wait_or_change_controls():
    frame = np.zeros((8, 8, 3), np.uint8)
    camera = SimpleNamespace(switch_mode_and_capture_arrays=lambda config, names: ([frame], {}))
    source = SimpleNamespace(picam2=camera, io_lock=threading.Lock())
    assert probe.capture_attempt(source, "still", False, CONTROLS) is frame


def test_probe_saves_two_originals_and_exact_inputs(new, tmp_path, monkeypatch, capsys):
    frame = np.random.default_rng(42).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    before = frame.copy()
    inputs = []
    class Process:
        returncode = 0
        def communicate(self, input=None, timeout=None):
            inputs.append(input)
            return b"", b""
    monkeypatch.setattr(new.subprocess, "Popen", lambda *a, **k: Process())
    for label in ("immediate", "settled"):
        probe.save_and_read(new, tmp_path, label, frame, mirrored=True)
        original = tmp_path / f"{tmp_path.name}-{label}-original.png"
        final = tmp_path / f"{tmp_path.name}-{label}-tesseract-input.png"
        assert np.array_equal(new.cv2.imread(str(original)), before)
        assert final.read_bytes() == inputs[-1]
        assert np.array_equal(new.cv2.imread(str(final)), before[:, ::-1])
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert np.array_equal(frame, before)
    assert capsys.readouterr().out.count('"quality_ok": false') == 2


def test_gap_wait_reports_first_resumed_preview():
    source = SimpleNamespace(condition=threading.Condition(),
                             still_capture_started_monotonic=None, last_still_gap_s=2.5)
    assert probe.wait_for_preview(source, "settled") == 2.5


def test_gap_wait_never_reports_previous_attempt_when_preview_is_stuck(monkeypatch):
    source = SimpleNamespace(condition=threading.Condition(),
                             still_capture_started_monotonic=1.0, last_still_gap_s=0.2)
    ticks = iter([0.0, 7.0])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="preview did not resume"):
        probe.wait_for_preview(source, "settled")
