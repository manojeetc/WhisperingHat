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
from types import SimpleNamespace

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


def _ocr_result(new, text, conf=90.0):
    words = [(word, conf) for word in text.split()]
    return text, words, new._ocr_text_metrics(text, words)


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


def test_grab_ocr_frame_prefers_still_when_opted_in(new, monkeypatch):
    monkeypatch.setattr(new, "OCR_USE_STILL", True)

    class FakeSource:
        def capture_still(self):
            return "STILL"

        def peek_latest(self):
            return "PEEKED"

        def read(self):
            return True, "READ"

    assert new._grab_ocr_frame(FakeSource()) == "STILL"


def test_grab_ocr_frame_skips_still_by_default(new, monkeypatch):
    monkeypatch.setattr(new, "OCR_USE_STILL", False)
    calls = []

    class FakeSource:
        def capture_still(self):
            calls.append("still")
            return "STILL"

        def peek_latest(self):
            return "PEEKED"

        def read(self):
            return True, "READ"

    assert new._grab_ocr_frame(FakeSource()) == "PEEKED"
    assert calls == []


def test_grab_ocr_frame_prefers_peek_without_still(new, monkeypatch):
    monkeypatch.setattr(new, "OCR_USE_STILL", False)
    class FakeSource:
        def peek_latest(self):
            return "PEEKED"

        def read(self):
            return True, "READ"

    assert new._grab_ocr_frame(FakeSource()) == "PEEKED"


def test_latest_frame_measures_high_res_still_gap(new):
    src = object.__new__(new.LatestFrameSource)
    src.condition = threading.Condition()
    src.sequence = 0
    src.latest_frame = None
    src.last_frame_monotonic = 10.0
    src.still_capture_started_monotonic = 10.0
    src.still_capture_active = True
    src.last_still_gap_s = None
    src.max_still_gap_s = 0.0

    # A frame arriving during the still window records the live gap.
    src._note_detection_frame("during", 12.0)
    assert src.max_still_gap_s == pytest.approx(2.0)
    assert src.still_capture_started_monotonic == pytest.approx(10.0)

    # The first frame after the still closes the full blind window.
    src.still_capture_active = False
    src._note_detection_frame("after", 13.0)
    assert src.last_still_gap_s == pytest.approx(3.0)
    assert src.max_still_gap_s == pytest.approx(3.0)
    assert src.still_capture_started_monotonic is None


def test_prepare_ocr_frame_resizes_without_upscaling(new, monkeypatch):
    monkeypatch.setattr(new, "OCR_MAX_WIDTH", 1280)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    prepared = new._prepare_ocr_frame(frame)
    assert prepared.shape == (720, 1280, 3)

    small = np.zeros((480, 640, 3), dtype=np.uint8)
    assert new._prepare_ocr_frame(small).shape == (480, 640, 3)


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
    monkeypatch.setattr(
        new, "_run_tesseract_tsv",
        lambda f: _ocr_result(new, "EXIT\n  only  "),
    )
    new._perform_ocr_read(object())
    assert spoken == [(2, "EXIT only")]


def test_perform_ocr_read_no_text(new, monkeypatch):
    spoken = _capture_speech(new, monkeypatch)
    monkeypatch.setattr(new, "_ocr_available", lambda: True)
    monkeypatch.setattr(new, "_grab_ocr_frame", lambda s: _frame())
    monkeypatch.setattr(
        new, "_run_tesseract_tsv",
        lambda f: ("", [], new._ocr_text_metrics("", [])),
    )
    new._perform_ocr_read(object())
    assert spoken == [(2, "I don't see any text.")]


def test_perform_ocr_read_rejects_gibberish(new, monkeypatch):
    spoken = _capture_speech(new, monkeypatch)
    monkeypatch.setattr(new, "_ocr_available", lambda: True)
    monkeypatch.setattr(new, "_grab_ocr_frame", lambda s: _frame())
    gibberish = "borotndd ood baal rrordve polvion sual inna bith"
    monkeypatch.setattr(
        new, "_run_tesseract_tsv",
        lambda f: _ocr_result(new, gibberish, conf=8.0),
    )
    new._perform_ocr_read(object())
    assert spoken == [(
        2,
        "I couldn't read the page. Hold it steady and try again.",
    )]

    # The rejected raw text must never be queued.
    assert all(gibberish not in message for _risk, message in spoken)


def test_perform_ocr_read_queues_short_chunks(new, monkeypatch):
    spoken = _capture_speech(new, monkeypatch)
    monkeypatch.setattr(new, "_ocr_available", lambda: True)
    monkeypatch.setattr(new, "_grab_ocr_frame", lambda s: _frame())
    text = (
        "The quick brown fox jumps over the lazy dog. " * 8
        + "This is the second part of the page. " * 8
    )
    monkeypatch.setattr(
        new, "_run_tesseract_tsv",
        lambda f: _ocr_result(new, text, conf=95.0),
    )
    new._perform_ocr_read(object())

    assert len(spoken) >= 2
    assert all(len(message) <= new.OCR_SPEECH_CHUNK_CHARS for _risk, message in spoken)
    assert len(spoken[0][1]) <= new.OCR_SPEECH_CHUNK_CHARS


def test_ocr_later_chunks_yield_to_routine_repeats(new, monkeypatch):
    calls = []

    def fake_direct(message, **kwargs):
        calls.append((message, kwargs.get("priority")))
        return True

    monkeypatch.setattr(new, "_ocr_available", lambda: True)
    monkeypatch.setattr(new, "_grab_ocr_frame", lambda s: _frame())
    monkeypatch.setattr(new, "_play_direct_response", fake_direct)
    text = "A sentence with enough words to read. " * 12
    monkeypatch.setattr(
        new, "_run_tesseract_tsv",
        lambda f: _ocr_result(new, text, conf=95.0),
    )
    new._perform_ocr_read(object())

    assert len(calls) >= 2
    assert calls[0][1] is None                 # first chunk keeps risk-2 priority
    assert all(priority == 0 for _message, priority in calls[1:])


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

    monkeypatch.setattr(new, "_run_tesseract_tsv", boom)
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


def test_parse_tesseract_tsv_and_quality_metrics(new):
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t92\tHello\n"
        "5\t1\t1\t1\t1\t2\t12\t0\t20\t10\t88\tworld\n"
        "5\t1\t1\t1\t2\t1\t0\t20\t10\t10\t85\tSecond\n"
    )
    text, words, metrics = new._parse_tesseract_tsv(tsv)
    assert text == "Hello world\nSecond"
    assert words == [("Hello", 92.0), ("world", 88.0), ("Second", 85.0)]
    assert metrics["word_count"] == 3
    assert metrics["mean_conf"] == pytest.approx(88.33333333333333)
    assert new._ocr_quality_ok(metrics) is True

    gibberish = "borotndd ood baal rrordve"
    _text, gibberish_words, bad_metrics = new._parse_tesseract_tsv(
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
        + "\n".join(
            f"5\t1\t1\t1\t1\t{i}\t0\t0\t10\t10\t8\t{word}"
            for i, word in enumerate(gibberish.split(), start=1)
        )
        + "\n"
    )
    assert bad_metrics["word_count"] == 4
    assert new._ocr_quality_ok(bad_metrics) is False


def test_split_ocr_speech_chunks_short_first_chunk(new):
    text = "A sentence with enough words to read. " * 12
    chunks = new._split_ocr_speech_chunks(text)
    assert chunks
    assert len(chunks[0]) <= new.OCR_SPEECH_CHUNK_CHARS
    assert len(chunks) <= new.OCR_MAX_SPEECH_CHUNKS
    assert all(len(chunk) <= new.OCR_SPEECH_CHUNK_CHARS for chunk in chunks)


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

def test_quality_rejects_high_confidence_gibberish(new):
    # Even if Tesseract is confident, a long block without normal English
    # function words is not safe to read aloud.
    sample = (
        "nae a bw borotndd ood baal orl rrordve polvion sual inna bith "
        "ovorl myot olde anve ont soni snd l Hon ton tiolqxo ow yin outnee "
        "aid oro cn aan gril orld iw Avotnirss Mi onlt oval int I boiwoqqn "
        "oburoblit orld divovwl tion a ai of yor We ga H ban ov qa qu Ho DO "
        "fy yosibntsiod ol wil Hyun Bay tte AO Oe wal ss gals ofl cw bern ff "
        "st giqiotnno wl y oh ek vod a volo cu onltl Aout hors oolov i banda "
        "aan OF AH aot bas aid oF pant Hohn unageb auoivord aidiag ol of "
        "matt inbawal vow od ban bodonay es naw oqo of AEB i ions omit wid "
        "dud v0 Awe od ninga bas"
    )
    _text, words, metrics = _ocr_result(new, sample, conf=95.0)
    assert metrics["common_word_ratio"] < new.OCR_QUALITY_MIN_COMMON_WORD_RATIO
    assert new._ocr_quality_ok(metrics) is False


def test_run_tesseract_tsv_requests_tsv_config(new, monkeypatch):
    captured = {}

    class FakeProc:
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return (
                b"level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
                b"left\ttop\twidth\theight\tconf\ttext\n"
                b"5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tHello\n",
                b"",
            )

        def kill(self):
            pass

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(new.subprocess, "Popen", fake_popen)
    text, words, metrics = new._run_tesseract_tsv(_frame())
    assert captured["cmd"][-1] == "tsv"
    assert text == "Hello"
    assert words == [("Hello", 90.0)]
    assert metrics["word_count"] == 1


# --- one-shot local image diagnostics ------------------------------------
@pytest.fixture
def dump_reader(new, monkeypatch, tmp_path):
    monkeypatch.setattr(new, "OCR_DUMP_ONCE", True)
    monkeypatch.setattr(new, "_ocr_dump_attempted", False)
    monkeypatch.setattr(new, "OCR_USE_STILL", True)
    monkeypatch.setattr(new, "OCR_MAX_WIDTH", 16)
    monkeypatch.setattr(new, "OCR_QUALITY_CHECK", True)
    monkeypatch.setattr(new, "_ocr_available", lambda: True)
    monkeypatch.setattr(new.tempfile, "gettempdir", lambda: str(tmp_path))
    original = np.random.default_rng(42).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    inputs = []

    class Source:
        def capture_still(self):
            return original

        def peek_latest(self):
            return original

    class Proc:
        returncode = 0

        def communicate(self, input=None, timeout=None):
            inputs.append(input)
            return b"", b""

    monkeypatch.setattr(new.subprocess, "Popen", lambda *a, **k: Proc())
    return Source(), original, inputs


def test_ocr_dump_off_writes_nothing(new, monkeypatch, tmp_path, dump_reader):
    source, _original, inputs = dump_reader
    monkeypatch.setattr(new, "OCR_DUMP_ONCE", False)
    _capture_speech(new, monkeypatch)
    new._perform_ocr_read(source)
    assert len(inputs) == 1
    assert list(tmp_path.iterdir()) == []
    assert new._ocr_dump_attempted is False


def test_ocr_dump_only_first_image(new, monkeypatch, tmp_path, dump_reader):
    source, _original, inputs = dump_reader
    _capture_speech(new, monkeypatch)
    new._perform_ocr_read(None)  # No image must not consume the one shot.
    assert new._ocr_dump_attempted is False
    new._perform_ocr_read(source)
    first_pair = {path: path.read_bytes() for path in tmp_path.rglob("*.png")}
    assert len(first_pair) == 2
    new._perform_ocr_read(source)
    assert len(inputs) == 2
    assert {path: path.read_bytes() for path in tmp_path.rglob("*.png")} == first_pair
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("result", ["empty", "low-confidence"])
def test_ocr_dump_pair_survives_rejection(
    new, monkeypatch, tmp_path, capsys, dump_reader, result
):
    source, original, inputs = dump_reader
    spoken = _capture_speech(new, monkeypatch)
    if result == "low-confidence":
        monkeypatch.setattr(
            new, "_parse_tesseract_tsv",
            lambda output: _ocr_result(new, "the words on this page", conf=44.0),
        )
    new._perform_ocr_read(source)
    original_path, = tmp_path.rglob("*-original.png")
    input_path, = tmp_path.rglob("*-tesseract-input.png")
    attempt = original_path.parent.name
    assert original_path.name == f"{attempt}-still-original.png"
    assert input_path.name == f"{attempt}-still-tesseract-input.png"
    assert np.array_equal(new.cv2.imread(str(original_path)), original)
    assert input_path.read_bytes() == inputs[0]  # Exact bytes sent to Tesseract.
    assert new.cv2.imread(str(input_path)).shape == (12, 16, 3)
    expected = (
        "I don't see any text." if result == "empty"
        else "I couldn't read the page. Hold it steady and try again."
    )
    assert spoken == [(2, expected)]
    log = capsys.readouterr().out
    for value in ("source=still", "original=32x24", "input=16x12",
                  str(original_path.resolve()), str(input_path.resolve())):
        assert value in log


@pytest.mark.parametrize("failure", ["none", "exception"])
def test_ocr_dump_labels_still_fallback_as_preview(
    new, monkeypatch, tmp_path, capsys, dump_reader, failure
):
    source, _original, _inputs = dump_reader
    _capture_speech(new, monkeypatch)

    def failed_still():
        if failure == "exception":
            raise RuntimeError("camera busy")
        return None

    monkeypatch.setattr(source, "capture_still", failed_still)
    new._perform_ocr_read(source)
    paths = list(tmp_path.rglob("*.png"))
    assert len(paths) == 2
    assert all("-preview-peek-" in path.name for path in paths)
    assert "source=preview-peek" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["original", "input", "git-temp"])
def test_ocr_dump_write_failure_keeps_ocr_running(
    new, monkeypatch, tmp_path, capsys, dump_reader, failure
):
    source, _original, inputs = dump_reader
    spoken = _capture_speech(new, monkeypatch)
    if failure == "original":
        monkeypatch.setattr(new.cv2, "imwrite", lambda *a: False)
    elif failure == "input":
        def fail_write(*args):
            raise OSError("disk full")
        monkeypatch.setattr(new.Path, "write_bytes", fail_write)
    else:
        (tmp_path / ".git").mkdir()
    new._perform_ocr_read(source)
    new._perform_ocr_read(source)
    assert len(inputs) == 2
    assert spoken == [(2, "I don't see any text.")] * 2
    assert new._ocr_dump_attempted is True
    log = capsys.readouterr().out
    assert log.count("failed:") == 1  # Failed writes also consume the one shot.
    if failure == "git-temp":
        assert list(tmp_path.rglob("*.png")) == []
        assert "outside Git working trees" in log


@pytest.mark.parametrize("origin", ["still", "peek", "read"])
@pytest.mark.parametrize("mirrored", [False, True])
def test_ocr_unmirrors_once_without_changing_capture(
    new, monkeypatch, tmp_path, dump_reader, origin, mirrored
):
    source, original, inputs = dump_reader
    source.ocr_mirrored = mirrored
    before = original.copy()
    monkeypatch.setattr(new, "OCR_USE_STILL", origin == "still")
    if origin == "read":
        monkeypatch.setattr(source, "peek_latest", lambda: None)
        monkeypatch.setattr(source, "read", lambda: (True, original), raising=False)
    _capture_speech(new, monkeypatch)
    new._perform_ocr_read(source)
    decoded = new.cv2.imdecode(np.frombuffer(inputs[0], dtype=np.uint8), new.cv2.IMREAD_COLOR)
    expected = new.cv2.resize(
        before[:, ::-1] if mirrored else before,
        (16, 12), interpolation=new.cv2.INTER_AREA,
    )
    assert np.array_equal(decoded, expected)  # Catches missing/double flips.
    assert np.array_equal(original, before)   # Shared detection pixels untouched.
    original_path, = tmp_path.rglob("*-original.png")
    assert np.array_equal(new.cv2.imread(str(original_path)), before)
    input_path, = tmp_path.rglob("*-tesseract-input.png")
    assert input_path.read_bytes() == inputs[0]
    # Also exercise the no-resize path, which used to return the source array.
    monkeypatch.setattr(new, "OCR_MAX_WIDTH", 32)
    assert np.array_equal(
        new._prepare_ocr_frame(original, unmirror=mirrored),
        before[:, ::-1] if mirrored else before,
    )
    assert np.array_equal(original, before)


@pytest.mark.parametrize("hflip,vflip", [(False, True), (False, False), (True, True), (True, False)])
def test_picamera_mirror_metadata_preserves_preview_and_still_transform(
    new, monkeypatch, hflip, vflip
):
    captured = {}
    frame = _frame()
    camera = SimpleNamespace(
        camera_properties={},
        create_preview_configuration=lambda **kw: captured.setdefault("preview", kw),
        create_still_configuration=lambda **kw: captured.setdefault("still", kw),
        configure=lambda config: None,
        start=lambda: None,
        capture_array=lambda name: frame,
        switch_mode_and_capture_array=lambda config, name: frame,
    )
    monkeypatch.setitem(new.sys.modules, "picamera2", SimpleNamespace(Picamera2=lambda **kw: camera))
    monkeypatch.setitem(new.sys.modules, "libcamera", SimpleNamespace(
        Transform=lambda **kw: kw, controls=SimpleNamespace(),
    ))
    monkeypatch.setattr(new, "CAMERA_HFLIP", hflip)
    monkeypatch.setattr(new, "CAMERA_VFLIP", vflip)
    monkeypatch.setattr(new, "CAMERA3_AF_MODE", "off")
    monkeypatch.setattr(new.time, "sleep", lambda duration: None)
    source = new.Picamera2Source()
    assert source.ocr_mirrored == (hflip != vflip)
    assert source.read()[1] is frame
    assert source.capture_still() is frame
    assert captured["preview"]["transform"] == {"hflip": int(hflip), "vflip": int(vflip)}
    assert captured["still"]["transform"] is captured["preview"]["transform"]
    # The standard newest-frame wrapper must pass through the OCR metadata.
    monkeypatch.setattr(new.threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: None))
    wrapped = new.LatestFrameSource(source)
    assert wrapped.ocr_mirrored == source.ocr_mirrored
