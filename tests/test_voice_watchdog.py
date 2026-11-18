"""Regression tests for voice listener persistence and watchdog recovery.

Silence is normal; only a dead worker or a stale stage heartbeat should trigger
recovery. These tests use fake streams so they do not need PyAudio or a Pi.
"""
import threading
import time
from types import SimpleNamespace

from conftest import reset_state


class _AliveThread:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


class _FakeStream:
    def __init__(self):
        self.calls = []

    def stop_stream(self):
        self.calls.append("stop_stream")

    def close(self):
        self.calls.append("close")


class _FakeMic:
    def __init__(self):
        self.exited = False

    def __exit__(self, *args):
        self.exited = True


def _prepare_watchdog(new, monkeypatch, alive=True):
    monkeypatch.setattr(new, "VOICE_WATCHDOG_ENABLED", True)
    monkeypatch.setattr(new, "VOICE_WATCHDOG_STALE_SECONDS", 12.0)
    monkeypatch.setattr(new, "VOICE_RECOVERY_MIN_INTERVAL_SECONDS", 30.0)
    monkeypatch.setattr(new, "VOICE_STALE_PLAYBACK_SECONDS", 25.0)
    monkeypatch.setattr(new, "voice_control_stop_event", threading.Event())
    monkeypatch.setattr(new, "voice_control_thread", _AliveThread(alive))
    monkeypatch.setattr(new, "voice_listener_stage", "listening")
    monkeypatch.setattr(new, "voice_listener_heartbeat", 1000.0)
    monkeypatch.setattr(new, "voice_last_recovery_monotonic", 0.0)
    monkeypatch.setattr(new, "voice_recovery_count", 0)
    monkeypatch.setattr(new, "speech_active_event", threading.Event())


def test_detection_off_does_not_stop_voice_listener(new, monkeypatch):
    reset_state(new)
    stop_event = threading.Event()
    thread = _AliveThread(True)
    monkeypatch.setattr(new, "voice_control_stop_event", stop_event)
    monkeypatch.setattr(new, "voice_control_thread", thread)
    monkeypatch.setattr(new, "VOICE_CONFIRM_COMMANDS", False)

    new.set_detection_enabled(False, announce=False)
    assert not stop_event.is_set()
    assert thread.is_alive()

    new.set_detection_enabled(True, announce=False)
    assert not stop_event.is_set()
    assert thread.is_alive()


def test_watchdog_ignores_normal_silence(new, monkeypatch):
    reset_state(new)
    _prepare_watchdog(new, monkeypatch)
    stream = _FakeStream()
    mic = _FakeMic()
    new._set_voice_listener_source(SimpleNamespace(stream=stream), mic)

    # Five seconds without a Vosk result is just a quiet room; the 1 s listen
    # timeout normally refreshes the heartbeat.
    assert new._voice_watchdog_check(now=1005.0) is False
    assert stream.calls == []
    assert mic.exited is False


def test_watchdog_unblocks_stale_stream_and_rate_limits(new, monkeypatch):
    reset_state(new)
    _prepare_watchdog(new, monkeypatch)
    stream = _FakeStream()
    mic = _FakeMic()
    new._set_voice_listener_source(SimpleNamespace(stream=stream), mic)

    assert new._voice_watchdog_check(now=1020.0) is True
    assert "stop_stream" in stream.calls
    assert "close" in stream.calls
    assert mic.exited is True
    assert new.voice_recovery_count == 1

    # A second recovery inside the 30 s minimum interval is skipped, not
    # hammered every second.
    assert new._voice_watchdog_check(now=1021.0) is False
    assert new.voice_recovery_count == 1


def test_watchdog_restarts_dead_worker(new, monkeypatch):
    reset_state(new)
    _prepare_watchdog(new, monkeypatch, alive=False)
    restarted = []
    monkeypatch.setattr(
        new,
        "start_voice_control",
        lambda start_watchdog=True: restarted.append(start_watchdog),
    )

    assert new._voice_watchdog_check(now=1001.0) is True
    assert restarted == [False]
    assert new.voice_recovery_count == 1


def test_watchdog_clears_stale_playback_suppression(new, monkeypatch):
    reset_state(new)
    _prepare_watchdog(new, monkeypatch)
    new._set_speech_active(True, "stuck alert")
    monkeypatch.setattr(new, "speech_last_start_time", time.time() - 30.0)
    monkeypatch.setattr(new, "current_audio_process", None)

    assert new._voice_watchdog_check(now=1001.0) is True
    assert not new.speech_active_event.is_set()


def test_voice_counters_separate_matched_and_dispatched(new, monkeypatch):
    reset_state(new)
    monkeypatch.setattr(new, "ENABLE_OCR_READING", True)
    monkeypatch.setattr(new, "_trigger_ocr_reading", lambda: True)
    monkeypatch.setattr(new, "voice_commands_matched", 0)
    monkeypatch.setattr(new, "voice_commands_dispatched", 0)

    assert new._apply_voice_command("read this") is True
    assert new.voice_commands_matched == 1
    assert new.voice_commands_dispatched == 1
