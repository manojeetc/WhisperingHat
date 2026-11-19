"""Real subprocess pipes, no microphone: a permanently stuck owner is killable."""
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from conftest import reset_state
from voice_capture import CaptureProcess


def command(audio=False):
    body = ("\nwhile True:\n _send(p,b'A',struct.pack('!d',time.monotonic())+bytes(2048)); time.sleep(.02)"
            if audio else "\ntime.sleep(120)")
    return [sys.executable, "-u", "-c",
            "import sys,time,struct,json; from voice_capture import _send; p=sys.stdout.buffer; "
            "_send(p,b'R',json.dumps(dict(rate=16000,width=2,chunk=1024,name='fixture')).encode())" + body]


def test_stuck_read_and_shutdown_are_bounded_and_silence_is_progress():
    capture = CaptureProcess(0, 16000, command=command(), stall_timeout=.1)
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="no PCM"):
        capture.read(1024)
    capture.close()
    capture.close()  # Idempotent; no double teardown.
    assert time.monotonic() - start < 2.5
    assert capture.process.poll() is not None
    with_capture = CaptureProcess(0, 16000, command=command(True))
    try:
        for _ in range(12):
            assert with_capture.read(1024) == bytes(2048)
        assert with_capture.frames.qsize() <= 8
    finally:
        with_capture.close()


@pytest.mark.parametrize("shutdown,automatic", [(False, False), (True, False), (False, True)])
def test_real_worker_reaps_blocked_capture_before_recovery_or_shutdown(new, monkeypatch, tmp_path, shutdown, automatic):
    reset_state(new)
    new.voice_control_stop_event.clear()
    new.voice_recovery_requested.clear()
    new.voice_listener_cleanup_failed.clear()
    opened, reading, decoded = [], threading.Event(), threading.Event()

    def capture(index, rate, **kwargs):
        assert all(old.process.poll() is not None and old.closed for old in opened)
        result = CaptureProcess(index, rate, command=command(bool(opened)), stall_timeout=.15, **kwargs)
        opened.append(result)
        return result

    class Recognizer:
        dynamic_energy_threshold = True
        energy_threshold = 10
        def adjust_for_ambient_noise(self, source, duration):
            pass
        def listen(self, source, **kwargs):
            reading.set()
            data = source.stream.read(source.CHUNK)
            return SimpleNamespace(frame_data=data, sample_rate=16000, sample_width=2)

    monkeypatch.setattr(new, "CaptureProcess", capture)
    class AudioSource:
        def __init__(self):
            raise NotImplementedError("abstract source")
    class WaitTimeoutError(Exception):
        pass
    monkeypatch.setattr(new, "sr", SimpleNamespace(AudioSource=AudioSource, Recognizer=Recognizer, WaitTimeoutError=WaitTimeoutError))
    monkeypatch.setattr(new, "VOICE_VOSK_MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(new, "VoskModel", lambda path: object())
    def final_result():
        new._set_speech_active(True, "warning starting after the captured utterance")
        return '{"text":"describe"}'
    monkeypatch.setattr(new, "KaldiRecognizer", lambda *a: SimpleNamespace(AcceptWaveform=lambda x: True,
        PartialResult=lambda: '{}', FinalResult=final_result))
    monkeypatch.setattr(new, "_apply_voice_command", lambda text: (decoded.set(), new.voice_control_stop_event.set()))
    thread = threading.Thread(target=new._voice_control_worker)
    monkeypatch.setattr(new, "voice_control_thread", thread)
    monkeypatch.setattr(new, "voice_watchdog_thread", None)
    thread.start()
    try:
        assert reading.wait(3)
        if not automatic:
            assert new._force_unblock_voice_stream()
        if shutdown:
            assert new.stop_voice_control()
        else:
            assert decoded.wait(4), "replacement must reach Vosk and command dispatch"
        thread.join(3)
        assert not thread.is_alive()
        assert len(opened) == (1 if shutdown else 2)
        assert all(p.closed and p.process.poll() is not None for p in opened)
        if automatic:
            assert new.voice_recovery_count == 1
    finally:
        new.voice_control_stop_event.set()
        thread.join(3)
        new.voice_control_stop_event.clear()


def test_echo_filter_resumes_after_playback_and_interruption(new, monkeypatch):
    reset_state(new)
    data = b'\x01\x02' * 32
    monkeypatch.setattr(new.time, "time", lambda: 1000.)
    new._set_speech_active(True, "speaker output")
    assert new._capture_audio_progress(data, time.monotonic()) == bytes(len(data))
    new._set_speech_active(False)  # Both normal end and interruption clear this state.
    assert new._capture_audio_progress(data, time.monotonic()) == bytes(len(data))
    monkeypatch.setattr(new.time, "time", lambda: 1000.3)
    assert new._capture_audio_progress(data, time.monotonic()) == data
