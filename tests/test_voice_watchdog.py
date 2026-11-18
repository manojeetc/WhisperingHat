"""Regression tests for voice listener persistence and watchdog recovery.

Silence is normal; only a dead worker or a stale stage heartbeat should trigger
recovery. These tests use fake streams so they do not need PyAudio or a Pi.
"""
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

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
    reset_state(new)
    for name in ("voice_recovery_requested", "voice_recovery_pending_reported",
                 "voice_listener_cleanup_failed"):
        monkeypatch.setattr(new, name, threading.Event())
    monkeypatch.setattr(new, "voice_recovery_request_time", 0.0)
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


def test_watchdog_requests_recovery_without_teardown(new, monkeypatch, capsys):
    reset_state(new)
    _prepare_watchdog(new, monkeypatch)
    stream = _FakeStream()
    mic = _FakeMic()
    new._set_voice_listener_source(SimpleNamespace(stream=stream), mic)

    assert new._voice_watchdog_check(now=1020.0) is True
    assert stream.calls == []
    assert mic.exited is False
    assert new.voice_recovery_requested.is_set()
    assert new.voice_recovery_count == 1

    monkeypatch.setattr(new, "voice_recovery_request_time", time.monotonic() - 4)
    assert new._voice_watchdog_check(now=1055.0) is False
    assert "recovery still pending" in capsys.readouterr().out
    assert stream.calls == []
    assert new.voice_recovery_count == 1


@pytest.mark.parametrize("debug", [True, False])
def test_watchdog_rate_limit_logging_has_no_argument_collision(new, monkeypatch, debug):
    _prepare_watchdog(new, monkeypatch)
    monkeypatch.setattr(new, "VOICE_DEBUG_TIMING", debug)
    monkeypatch.setattr(new, "voice_last_recovery_monotonic", 1019.0)
    assert new._voice_watchdog_check(now=1020.0) is False
    assert not new.voice_recovery_requested.is_set()

    # A second recovery inside the 30 s minimum interval is skipped, not
    # hammered every second.
    assert new._voice_watchdog_check(now=1021.0) is False
    assert new.voice_recovery_count == 0


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


def test_dead_owner_with_unconfirmed_teardown_cannot_restart(new, monkeypatch):
    _prepare_watchdog(new, monkeypatch, alive=False)
    new._set_voice_listener_source(SimpleNamespace(stream=_FakeStream()), _FakeMic())
    restarted = []
    monkeypatch.setattr(new, "start_voice_control", lambda **kwargs: restarted.append(True))
    assert not new._voice_watchdog_check(now=1020.0)
    assert new.voice_listener_cleanup_failed.is_set()
    assert restarted == []


def test_open_fallback_stops_when_partial_stream_teardown_fails(new, monkeypatch):
    _prepare_watchdog(new, monkeypatch)
    opened, exited = [], []

    class Microphone:
        def __init__(self, **kwargs):
            opened.append(kwargs)
            self.stream = object()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            exited.append(True)
            raise OSError("native close failed")

    monkeypatch.setattr(new, "sr", SimpleNamespace(Microphone=Microphone))
    monkeypatch.setattr(new, "suppress_audio_stderr", nullcontext)
    monkeypatch.setattr(new, "_voice_microphone_name", lambda index: "fake mic")
    # Missing SAMPLE_RATE makes the partially opened source fail validation/logging.
    with pytest.raises(new._VoiceCleanupFailed, match="native close failed"):
        new._open_voice_source(0)
    assert len(opened) == len(exited) == 1


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


def _blocked_worker(new, monkeypatch, tmp_path, block_stage="listening", close_fails=False):
    """Drive the real worker with a native read/teardown held at explicit barriers."""
    _prepare_watchdog(new, monkeypatch)
    state = SimpleNamespace(
        reading=threading.Event(), release_read=threading.Event(),
        closing=threading.Event(), release_close=threading.Event(),
        processed=threading.Event(), microphones=[], calls=[],
    )

    class NativeStream:
        def __init__(self, generation):
            self.generation = generation
            self.owner = threading.get_ident()
            self.in_read = False
            self.closed = 0

        def read(self, size):
            assert threading.get_ident() == self.owner
            self.in_read = True
            try:
                if self.generation == 1:
                    state.reading.set()
                    assert state.release_read.wait(3), "test did not release native read"
                return b"\0\0"
            finally:
                self.in_read = False

        def close(self):
            state.calls.append(("close", self.generation, threading.get_ident(), self.in_read))
            assert threading.get_ident() == self.owner
            assert not self.in_read
            self.closed += 1
            assert self.closed == 1, "double close"
            if close_fails:
                raise OSError("fake teardown failure")

    class Microphone:
        SAMPLE_RATE = 16000
        SAMPLE_WIDTH = 2
        CHUNK = 1

        def __init__(self, generation):
            self.native = NativeStream(generation)
            self.stream = self.native
            self.exits = 0
            self.exited = False

        def __exit__(self, *args):
            self.exits += 1
            assert self.exits == 1, "duplicate microphone teardown"
            self.stream.close()
            if self.native.generation == 1:
                state.closing.set()
                assert state.release_close.wait(3), "test did not release teardown"
            self.stream = None
            self.exited = True

    def open_source(index):
        assert all(mic.exited for mic in state.microphones), "early reopen"
        mic = Microphone(len(state.microphones) + 1)
        state.microphones.append(mic)
        state.calls.append(("open", mic.native.generation, threading.get_ident(), False))
        return mic, mic, mic.SAMPLE_RATE

    class Recognizer:
        energy_threshold = 1.0

        def adjust_for_ambient_noise(self, source, duration):
            if block_stage == "calibrating":
                source.stream.read(source.CHUNK)

        def listen(self, source, **kwargs):
            data = source.stream.read(source.CHUNK)
            return SimpleNamespace(frame_data=data, sample_rate=16000, sample_width=2)

    def apply_command(text):
        assert text == "read this"
        state.processed.set()
        new.voice_control_stop_event.set()

    monkeypatch.setattr(new, "VOICE_VOSK_MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(new, "VOICE_PRINT_DEBUG", False)
    monkeypatch.setattr(new, "VOICE_DEBUG_TIMING", False)
    monkeypatch.setattr(new, "sr", SimpleNamespace(Recognizer=Recognizer, WaitTimeoutError=TimeoutError))
    monkeypatch.setattr(new, "VoskModel", lambda path: object())
    monkeypatch.setattr(new, "KaldiRecognizer", lambda *args: SimpleNamespace(
        AcceptWaveform=lambda data: True, PartialResult=lambda: '{}',
        FinalResult=lambda: '{"text":"read this"}',
    ))
    monkeypatch.setattr(new, "_select_voice_microphone_index", lambda: 0)
    monkeypatch.setattr(new, "_open_voice_source", open_source)
    monkeypatch.setattr(new, "suppress_audio_stderr", nullcontext)
    monkeypatch.setattr(new, "_apply_voice_command", apply_command)
    monkeypatch.setattr(new, "_convert_audio_to_16khz_mono_int16", lambda audio: audio.frame_data)
    monkeypatch.setattr(new, "voice_watchdog_thread", None)
    thread = threading.Thread(target=new._voice_control_worker, daemon=True)
    monkeypatch.setattr(new, "voice_control_thread", thread)
    state.thread = thread
    thread.start()
    assert state.reading.wait(2), "listener did not enter native read"
    return state


def _release_worker(new, state):
    new.voice_control_stop_event.set()
    state.release_read.set()
    state.release_close.set()
    state.thread.join(2)
    assert not state.thread.is_alive()


@pytest.mark.parametrize("stage", ["listening", "calibrating"])
def test_recovery_during_read_has_one_owner_and_replacement_processes_audio(
    new, monkeypatch, tmp_path, capsys, stage,
):
    state = _blocked_worker(new, monkeypatch, tmp_path, stage)
    try:
        # Old _force_unblock_voice_stream closed this stream while in_read=True.
        assert new._voice_watchdog_check(now=new.voice_listener_heartbeat + 13)
        assert state.calls == [("open", 1, state.thread.ident, False)]
        assert state.microphones[0].exits == 0
        assert not new._force_unblock_voice_stream()  # duplicate request is coalesced
        state.release_read.set()
        assert state.closing.wait(2)
        assert len(state.microphones) == 1  # read returned, teardown still held
        assert new.voice_recovery_requested.is_set()
        state.release_close.set()
        assert state.processed.wait(2), "replacement never decoded audio"
        state.thread.join(2)
        assert not state.thread.is_alive()
        assert new.voice_audio_frames == 1
        assert new.voice_nonempty_results == 1
        assert new.voice_last_processed_monotonic > 0
        assert len(state.microphones) == 2
        assert all(mic.exits == mic.native.closed == 1 for mic in state.microphones)
        assert all(call[2] == state.thread.ident and not call[3] for call in state.calls)
        assert [call[:2] for call in state.calls] == [("open",1),("close",1),("open",2),("close",2)]
        assert "replacement stream processed audio" in capsys.readouterr().out
    finally:
        _release_worker(new, state)


def test_shutdown_during_recovery_does_not_reopen(new, monkeypatch, tmp_path):
    state = _blocked_worker(new, monkeypatch, tmp_path)
    try:
        assert new._force_unblock_voice_stream()
        state.release_read.set()
        assert state.closing.wait(2)
        results = []
        stopper = threading.Thread(target=lambda: results.append(new.stop_voice_control()))
        stopper.start()
        assert new.voice_control_stop_event.wait(2)
        assert len(state.microphones) == 1
        state.release_close.set()
        stopper.join(2)
        assert results == [True]
        assert len(state.microphones) == 1
        assert not state.processed.is_set()
        assert state.microphones[0].exits == state.microphones[0].native.closed == 1
    finally:
        _release_worker(new, state)


def test_stuck_native_read_has_bounded_shutdown_and_no_false_recovery(
    new, monkeypatch, tmp_path, capsys,
):
    state = _blocked_worker(new, monkeypatch, tmp_path)
    try:
        assert new._force_unblock_voice_stream()
        monkeypatch.setattr(new, "VOICE_SHUTDOWN_TIMEOUT_SECONDS", .02)
        started = time.monotonic()
        assert new.stop_voice_control() is False
        assert time.monotonic() - started < .5
        assert state.thread.is_alive()
        assert new.voice_listener_stage == "shutdown_pending"
        assert "shutdown pending" in capsys.readouterr().out
        assert len(state.calls) == 1
        new.start_voice_control(start_watchdog=False)
        assert len(state.microphones) == 1
        assert new.voice_control_thread is state.thread
    finally:
        _release_worker(new, state)


def test_failed_owner_teardown_blocks_reopen(new, monkeypatch, tmp_path, capsys):
    state = _blocked_worker(new, monkeypatch, tmp_path, close_fails=True)
    try:
        assert new._force_unblock_voice_stream()
        state.release_read.set()
        state.thread.join(2)
        assert not state.thread.is_alive()
        assert new.voice_listener_cleanup_failed.is_set()
        assert not new._voice_watchdog_check(now=time.monotonic() + 100)
        new.start_voice_control()
        assert len(state.microphones) == 1
        assert state.microphones[0].exits == state.microphones[0].native.closed == 1
        assert "teardown failed; no reopen" in capsys.readouterr().out
    finally:
        _release_worker(new, state)
