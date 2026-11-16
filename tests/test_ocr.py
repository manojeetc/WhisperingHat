"""Regression tests for voice-activated OCR reading (Tesseract via subprocess).

Tesseract is NOT installed on the desktop, so the binary is mocked everywhere;
frame capture and the speech queue are mocked too. These tests prove the
decision logic, phrase routing, clean-frame access, message selection, command
construction, cooldown and shutdown cleanup. Real OCR accuracy, Pi latency and
microphone recognition of the new phrases must be verified on the Pi.
"""
import json
import subprocess
import threading

import numpy as np
import pytest


def _capture_speech(new, monkeypatch):
    spoken = []
    monkeypatch.setattr(
        new, "play_alert_sound", lambda risk, msg: spoken.append((risk, msg)) or True
    )
    return spoken


def _frame():
    return np.zeros((8, 8, 3), dtype=np.uint8)


# --- text cleaning --------------------------------------------------------
def test_clean_ocr_text_collapses_whitespace(new):
    assert new._clean_ocr_text("  HELLO   world\n\nthis is\t a sign  \n") == \
        "HELLO world this is a sign"


def test_clean_ocr_text_empty(new):
    assert new._clean_ocr_text("") == ""
    assert new._clean_ocr_text("   \n\t ") == ""


def test_clean_ocr_text_caps_at_word_boundary(new):
    cap = new.OCR_MAX_SPEECH_CHARS
    out = new._clean_ocr_text("word " * 500)
    assert len(out) <= cap
    assert len(out) >= cap - 5            # cut close to the cap, not over-truncated
    assert out == out.strip()             # no trailing space
    assert all(tok == "word" for tok in out.split())   # whole words only


# --- voice phrase routing -------------------------------------------------
def test_read_phrases_trigger_ocr(new, monkeypatch):
    calls = []
    monkeypatch.setattr(new, "ENABLE_OCR_READING", True)
    monkeypatch.setattr(new, "_trigger_ocr_reading", lambda: calls.append(1) or True)
    for phrase in new.OCR_READ_PHRASES:
        calls.clear()
        assert new._apply_voice_command(phrase) is True
        assert calls == [1]


def test_read_phrase_normalizes_casing(new, monkeypatch):
    calls = []
    monkeypatch.setattr(new, "ENABLE_OCR_READING", True)
    monkeypatch.setattr(new, "_trigger_ocr_reading", lambda: calls.append(1) or True)
    assert new._apply_voice_command("  Read THIS ") is True
    assert calls == [1]


def test_detection_off_still_routes(new, monkeypatch):
    seen = {}
    monkeypatch.setattr(new, "set_detection_enabled",
                        lambda e, announce=True: seen.update(e=e))
    monkeypatch.setattr(new, "last_voice_command_time", 0.0)
    assert new._apply_voice_command("detection off") is True
    assert seen["e"] is False


def test_non_command_returns_false(new, monkeypatch):
    monkeypatch.setattr(new, "_trigger_ocr_reading", lambda: True)
    assert new._apply_voice_command("hello there") is False


# --- Vosk grammar ---------------------------------------------------------
def test_grammar_includes_read_phrases_when_enabled(new, monkeypatch):
    monkeypatch.setattr(new, "ENABLE_OCR_READING", True)
    g = json.loads(new._voice_command_grammar())
    assert "detection on" in g and "detection off" in g and "[unk]" in g
    for p in new.OCR_READ_PHRASES:
        assert p in g


def test_grammar_omits_read_phrases_when_disabled(new, monkeypatch):
    monkeypatch.setattr(new, "ENABLE_OCR_READING", False)
    g = json.loads(new._voice_command_grammar())
    assert "[unk]" in g and "detection on" in g
    for p in new.OCR_READ_PHRASES:
        assert p not in g


# --- clean frame access ---------------------------------------------------
def test_peek_latest_returns_copy_without_advancing(new):
    src = object.__new__(new.LatestFrameSource)
    src.condition = threading.Condition()
    frame = _frame()
    frame[0, 0, 0] = 42
    src.latest_frame = frame
    src.sequence = 7
    src.last_returned_sequence = 3

    out = src.peek_latest()
    assert out is not None and out is not frame
    assert np.array_equal(out, frame)
    out[0, 0, 0] = 99
    assert frame[0, 0, 0] == 42                      # original untouched
    assert src.sequence == 7 and src.last_returned_sequence == 3   # not advanced


def test_peek_latest_none_when_no_frame(new):
    src = object.__new__(new.LatestFrameSource)
    src.condition = threading.Condition()
    src.latest_frame = None
    assert src.peek_latest() is None


def test_grab_ocr_frame_prefers_peek(new):
    class FakeSource:
        def peek_latest(self):
            return "PEEKED"

        def read(self):
            return True, "READ"

    assert new._grab_ocr_frame(FakeSource()) == "PEEKED"


def test_grab_ocr_frame_falls_back_to_read(new):
    class FakeSource:
        def read(self):
            return True, "READ"

    assert new._grab_ocr_frame(FakeSource()) == "READ"


def test_grab_ocr_frame_none_source(new):
    assert new._grab_ocr_frame(None) is None


# --- read decision / message selection -----------------------------------
def test_perform_ocr_read_speaks_text(new, monkeypatch):
    spoken = _capture_speech(new, monkeypatch)
    monkeypatch.setattr(new, "_ocr_available", lambda: True)
    monkeypatch.setattr(new, "_grab_ocr_frame", lambda s: _frame())
    monkeypatch.setattr(new, "_run_tesseract", lambda f: "EXIT\n  only  ")
    new._perform_ocr_read(object())
    assert spoken == [(2, "EXIT only")]


def test_perform_ocr_read_no_text(new, monkeypatch):
    spoken = _capture_speech(new, monkeypatch)
    monkeypatch.setattr(new, "_ocr_available", lambda: True)
    monkeypatch.setattr(new, "_grab_ocr_frame", lambda s: _frame())
    monkeypatch.setattr(new, "_run_tesseract", lambda f: "   \n ")
    new._perform_ocr_read(object())
    assert spoken == [(2, "I don't see any text.")]


def test_perform_ocr_read_unavailable(new, monkeypatch):
    spoken = _capture_speech(new, monkeypatch)
    monkeypatch.setattr(new, "_ocr_available", lambda: False)
    new._perform_ocr_read(object())
    assert spoken == [(2, "Text reading isn't available.")]


def test_perform_ocr_read_failure(new, monkeypatch):
    spoken = _capture_speech(new, monkeypatch)
    monkeypatch.setattr(new, "_ocr_available", lambda: True)
    monkeypatch.setattr(new, "_grab_ocr_frame", lambda s: _frame())

    def boom(f):
        raise RuntimeError("tesseract died")

    monkeypatch.setattr(new, "_run_tesseract", boom)
    new._perform_ocr_read(object())
    assert spoken == [(2, "I couldn't read that.")]


def test_perform_ocr_read_no_frame(new, monkeypatch):
    spoken = _capture_speech(new, monkeypatch)
    monkeypatch.setattr(new, "_ocr_available", lambda: True)
    monkeypatch.setattr(new, "_grab_ocr_frame", lambda s: None)
    new._perform_ocr_read(object())
    assert spoken == [(2, "I can't see right now.")]


# --- tesseract subprocess -------------------------------------------------
def test_run_tesseract_command_and_output(new, monkeypatch):
    captured = {}

    class FakeProc:
        returncode = 0

        def communicate(self, input=None, timeout=None):
            captured["input"] = input
            captured["timeout"] = timeout
            return b"SIGN TEXT\n", b""

        def kill(self):
            captured["killed"] = True

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        captured["kw"] = kw
        return FakeProc()

    monkeypatch.setattr(new.subprocess, "Popen", fake_popen)
    out = new._run_tesseract(_frame())
    assert out == "SIGN TEXT\n"
    cmd = captured["cmd"]
    assert cmd[0] == new.OCR_BINARY
    assert "stdin" in cmd and "stdout" in cmd
    assert "-l" in cmd and new.OCR_LANGUAGE in cmd
    assert "--psm" in cmd and str(new.OCR_PSM) in cmd
    assert captured["timeout"] == new.OCR_TIMEOUT_SECONDS
    assert captured["input"]                       # PNG bytes were fed
    assert captured["kw"]["stdin"] == subprocess.PIPE


def test_run_tesseract_timeout_kills(new, monkeypatch):
    killed = {}

    class FakeProc:
        returncode = -9

        def communicate(self, input=None, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="tesseract", timeout=timeout)
            return b"", b""

        def kill(self):
            killed["yes"] = True

    monkeypatch.setattr(new.subprocess, "Popen", lambda *a, **k: FakeProc())
    with pytest.raises(subprocess.TimeoutExpired):
        new._run_tesseract(_frame())
    assert killed.get("yes")


def test_run_tesseract_nonzero_raises(new, monkeypatch):
    class FakeProc:
        returncode = 1

        def communicate(self, input=None, timeout=None):
            return b"", b"err"

        def kill(self):
            pass

    monkeypatch.setattr(new.subprocess, "Popen", lambda *a, **k: FakeProc())
    with pytest.raises(RuntimeError):
        new._run_tesseract(_frame())


# --- trigger / cooldown ---------------------------------------------------
def test_trigger_ocr_sets_event(new, monkeypatch):
    monkeypatch.setattr(new, "ENABLE_OCR_READING", True)
    monkeypatch.setattr(new, "last_ocr_trigger_time", 0.0)
    new.ocr_trigger_event.clear()
    assert new._trigger_ocr_reading() is True
    assert new.ocr_trigger_event.is_set()
    new.ocr_trigger_event.clear()


def test_trigger_ocr_disabled(new, monkeypatch):
    monkeypatch.setattr(new, "ENABLE_OCR_READING", False)
    new.ocr_trigger_event.clear()
    assert new._trigger_ocr_reading() is False
    assert not new.ocr_trigger_event.is_set()


def test_trigger_ocr_cooldown(new, monkeypatch):
    monkeypatch.setattr(new, "ENABLE_OCR_READING", True)
    monkeypatch.setattr(new, "last_ocr_trigger_time", 0.0)
    new.ocr_trigger_event.clear()
    new._trigger_ocr_reading()
    new.ocr_trigger_event.clear()
    assert new._trigger_ocr_reading() is True      # recognized but debounced
    assert not new.ocr_trigger_event.is_set()


# --- worker lifecycle / shutdown -----------------------------------------
def test_ocr_worker_lifecycle(new, monkeypatch):
    monkeypatch.setattr(new, "ENABLE_OCR_READING", True)
    monkeypatch.setattr(new, "_ocr_available", lambda: True)
    monkeypatch.setattr(new, "_perform_ocr_read", lambda source: None)
    new.start_ocr_reader(None)
    assert new.ocr_worker_thread is not None and new.ocr_worker_thread.is_alive()
    new.stop_ocr_reader()
    assert not new.ocr_worker_thread.is_alive()


def test_stop_ocr_reader_kills_running_process(new, monkeypatch):
    monkeypatch.setattr(new, "ocr_worker_thread", None)
    killed = {}

    class FakeProc:
        def kill(self):
            killed["yes"] = True

    monkeypatch.setattr(new, "current_ocr_process", FakeProc())
    new.stop_ocr_reader()
    assert killed.get("yes")
    assert new.ocr_stop_event.is_set()