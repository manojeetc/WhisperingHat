"""Request-only vision checks; no model downloads or Pi hardware."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import voice_vision as vision


def test_caption_one_sentence_and_ocr_gate():
    assert vision.short_sentence("<s>A chair near a desk. Another sentence.</s>") == "A chair near a desk."
    with pytest.raises(ValueError):
        vision.short_sentence("")
    regions = vision.ordered_regions(np.zeros((2, 4, 2)), ["First line.", "Second line."], [.98, .96])
    assert [r["text"] for r in regions] == ["First line.", "Second line."]
    assert vision.ocr_reliable(regions)
    assert not vision.ocr_reliable([dict(text="junk", confidence=.79)])
    assert not vision.ocr_reliable([])
    assert not vision.ocr_reliable([dict(text="Clipped", confidence=.99,
        box=[[4, 4], [16, 4], [16, 19], [4, 19]])], (20, 20, 3))


@pytest.mark.parametrize("fails", [False, True])
def test_caption_handoff_does_not_wait(new, monkeypatch, fails):
    entered, finish = threading.Event(), threading.Event()
    spoken = []
    frame = np.zeros((8, 10, 3), np.uint8)
    def request(pixels, timeout):
        assert np.array_equal(pixels, frame)
        entered.set()
        assert finish.wait(2)
        if fails:
            raise TimeoutError("deadline")
        return dict(text="A room with a chair.")
    monkeypatch.setattr(new, "caption_model", SimpleNamespace(request=request))
    monkeypatch.setattr(new, "caption_worker_thread", None)
    monkeypatch.setattr(new, "_play_direct_response", lambda text, *a, **kw: spoken.append(text))
    new.ocr_stop_event.clear()
    new._respond_to_scene(frame, "Existing description.", 1)
    assert entered.wait(2)
    assert not spoken  # Main loop returned while inference remains blocked.
    finish.set()
    new.caption_worker_thread.join(2)
    assert spoken == ["Existing description." if fails else "A room with a chair."]


@pytest.mark.parametrize("result", ["ok", "unreliable", "timeout"])
def test_ppocr_full_resolution_unmirror_and_fallback(new, monkeypatch, result):
    original = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    before = original.copy()
    seen, spoken = [], []
    def request(frame, timeout):
        assert np.array_equal(frame, original[:, ::-1])
        if result == "timeout":
            raise TimeoutError("deadline")
        return dict(text="First line.\nSecond line.", reliable=result == "ok",
                    regions=[dict(text="First line.", confidence=.98), dict(text="Second line.", confidence=.99)])
    def tesseract(frame):
        seen.append(frame)
        return "", [], new._ocr_text_metrics("", [])
    monkeypatch.setattr(new, "ppocr_model", SimpleNamespace(request=request))
    monkeypatch.setattr(new, "_grab_ocr_frame", lambda source: original)
    monkeypatch.setattr(new, "_begin_ocr_dump", lambda *a: None)
    monkeypatch.setattr(new, "OCR_MAX_WIDTH", 16)
    monkeypatch.setattr(new, "_ocr_available", lambda: True)
    monkeypatch.setattr(new, "_run_tesseract_tsv", tesseract)
    monkeypatch.setattr(new, "_play_direct_response", lambda text, **kw: spoken.append(text) or True)
    new._perform_ocr_read(SimpleNamespace(ocr_mirrored=True))
    assert np.array_equal(original, before)
    if result == "ok":
        assert not seen
        assert spoken == ["First line. Second line."]
    else:
        assert len(seen) == 1 and seen[0].shape == (12, 16, 3)
        assert spoken == [new.OCR_FAILURE_MESSAGE]


def test_verbatim_chunks_do_not_truncate(new):
    text = "This is a complete line. " * 100
    assert " ".join(new._split_ocr_speech_chunks(text, verbatim=True)) == text.strip()


def test_shutdown_during_caption_discards_response(new, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    spoken = []
    def request(frame, timeout):
        entered.set()
        assert finish.wait(2)
        return dict(text="A room.")
    monkeypatch.setattr(new, "caption_model", SimpleNamespace(request=request))
    monkeypatch.setattr(new, "caption_worker_thread", None)
    monkeypatch.setattr(new, "_play_direct_response", lambda *a, **kw: spoken.append(a))
    new.ocr_stop_event.clear()
    new._respond_to_scene(None, "Fallback.", 1)
    assert entered.wait(2)
    new.ocr_stop_event.set()
    finish.set()
    new.caption_worker_thread.join(2)
    assert not spoken
    new.ocr_stop_event.clear()


def test_ocr_dump_write_failure_does_not_discard_neural_text(new, monkeypatch, tmp_path):
    spoken = []
    result = dict(text="Readable text.", reliable=True,
                  regions=[dict(text="Readable text.", confidence=.99)])
    monkeypatch.setattr(new, "ppocr_model", SimpleNamespace(request=lambda *a: result))
    monkeypatch.setattr(new, "_grab_ocr_frame", lambda s: np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setattr(new, "_begin_ocr_dump", lambda *a: tmp_path / "tesseract-input.png")
    monkeypatch.setattr(new.cv2, "imwrite", lambda *a: False)
    monkeypatch.setattr(new, "_play_direct_response", lambda text, **kw: spoken.append(text) or True)
    new._perform_ocr_read(None)
    assert spoken == ["Readable text."]


def test_request_timeout_stops_child_and_loading_falls_back():
    model = vision.RequestModel.__new__(vision.RequestModel)
    calls = []
    model.kind, model.ready, model.lock = "ocr", True, threading.Lock()
    model.pipe = SimpleNamespace(send=lambda frame: calls.append("send"), poll=lambda timeout: False,
                                 close=lambda: calls.append("close"))
    alive = [True]
    model.process = SimpleNamespace(is_alive=lambda: alive[0],
        terminate=lambda: (calls.append("terminate"), alive.__setitem__(0, False)),
        join=lambda timeout: calls.append("join"))
    with pytest.raises(TimeoutError):
        model.request("pixels", .01)
    assert calls == ["send", "terminate", "join", "close"]
    calls.clear()
    with pytest.raises(RuntimeError, match="loading"):
        model.request("pixels", .01)
    assert not calls


def test_concurrent_still_uses_next_camera_request_without_mode_switch(new, monkeypatch):
    captured, released = {}, []
    main = np.full((24, 32, 3), 90, np.uint8)
    yuv = np.full((12, 10), 128, np.uint8)  # I420 for 10x8 preview.
    request = SimpleNamespace(make_array=lambda stream: main if stream == "main" else yuv,
                              release=lambda: released.append(True))
    camera = SimpleNamespace(camera_properties={}, configure=lambda config: None, start=lambda: None,
        create_preview_configuration=lambda **kw: captured.update(kw), capture_request=lambda: request)
    monkeypatch.setitem(new.sys.modules, "picamera2", SimpleNamespace(Picamera2=lambda **kw: camera))
    monkeypatch.setitem(new.sys.modules, "libcamera", SimpleNamespace(Transform=lambda **kw: kw, controls=None))
    monkeypatch.setattr(new, "VISION_COMMANDS_ENABLED", True)
    monkeypatch.setattr(new, "USE_LATEST_FRAME_THREAD", True)
    monkeypatch.setattr(new, "CAMERA3_AF_MODE", "off")
    monkeypatch.setattr(new.time, "sleep", lambda seconds: None)
    source = new.Picamera2Source()
    result = []
    with source.still_condition:
        worker = threading.Thread(target=lambda: result.append(source.capture_still()))
        worker.start()
    # Synchronize with the request flag, not timing-dependent sleeps.
    with source.still_condition:
        assert source.still_condition.wait_for(lambda: source.still_requested, timeout=2)
    ok, preview = source.read()
    worker.join(2)
    assert ok and preview.shape == (8, 10, 3)
    assert np.array_equal(result[0], main)
    assert result[0] is not main and released == [True]
    assert captured["lores"]["format"] == "YUV420"
    assert source.read()[0]  # Preview continues after the still.
