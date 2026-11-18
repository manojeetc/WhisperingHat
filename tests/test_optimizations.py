"""Regression tests for the three authorized backend optimizations.

- YOLO Conv+BN fusion at load: same weights/outputs, fewer CPU ops per frame.
- Anti-aliased Vosk downsampling: an INTENTIONAL audio-prep change. Out-of-band
  content (>8 kHz) is now low-passed away instead of aliasing into the speech
  band. In-band speech, the 16 kHz fast path, output length and format are
  preserved.
- XNNPACK needs no code change: the LiteRT/TFLite Python interpreter already
  applies it by default (AUTO/BUILTIN op resolver). Verified separately.

Only `ultralytics` is stubbed; YOLO is monkeypatched per-test with a fake that
records the fuse() call. Resampling tests run real numpy on synthetic tones.
"""
import numpy as np
import pytest

from conftest import original_linear_resample


class _Audio:
    def __init__(self, frame_data, sample_width, sample_rate):
        self.frame_data = frame_data
        self.sample_width = sample_width
        self.sample_rate = sample_rate


def _rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


# --- YOLO Conv+BN fusion --------------------------------------------------
def test_load_detection_model_fuses_pytorch(new, monkeypatch):
    record = {}

    class FakeYOLO:
        def __init__(self, name):
            record["name"] = name

        def fuse(self, verbose=True, imgsz=640):
            record["fuse_verbose"] = verbose
            return self

    monkeypatch.setattr(new, "YOLO", FakeYOLO)
    model = new.load_detection_model()
    assert isinstance(model, FakeYOLO)
    assert record["name"] == new.MODEL_NAME
    assert record["fuse_verbose"] is False  # fused, without extra startup noise


def test_load_detection_model_skips_fuse_for_non_pytorch(new, monkeypatch):
    class FakeYOLO:
        def __init__(self, name):
            pass

        def fuse(self, verbose=True, imgsz=640):
            raise TypeError("fuse() only supports PyTorch models")

    monkeypatch.setattr(new, "YOLO", FakeYOLO)
    model = new.load_detection_model()  # must not raise; NCNN etc. already fused
    assert isinstance(model, FakeYOLO)


# --- Anti-aliased downsampling -------------------------------------------
def test_downsample_removes_out_of_band_tone(new):
    rate, target = 44100, 16000
    t = np.arange(int(rate * 0.5), dtype=np.float64) / rate
    amp = 16000.0
    # 12 kHz is above the 8 kHz output Nyquist: it must not survive downsampling.
    tone = (amp * np.sin(2 * np.pi * 12000 * t)).astype(np.int16)
    samples = tone.astype(np.float32)

    old = original_linear_resample(samples, rate, target)
    pcm = new._convert_audio_to_16khz_mono_int16(_Audio(tone.tobytes(), 2, rate))
    got = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)

    assert got.size == old.size
    assert _rms(old) > 0.05 * amp          # linear interp really does alias it through
    assert _rms(got) < 0.1 * _rms(old)     # new path removes that aliased energy


def test_downsample_preserves_in_band_tone(new):
    rate, target = 44100, 16000
    t = np.arange(int(rate * 0.5), dtype=np.float64) / rate
    amp = 16000.0
    tone = (amp * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)  # 1 kHz speech band
    samples = tone.astype(np.float32)

    old = original_linear_resample(samples, rate, target)
    pcm = new._convert_audio_to_16khz_mono_int16(_Audio(tone.tobytes(), 2, rate))
    got = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)

    assert got.size == old.size
    assert _rms(got) == pytest.approx(_rms(old), rel=0.15)
    assert _rms(got) > 0.7 * amp / np.sqrt(2)


def test_resample_empty_and_silence(new):
    assert new._convert_audio_to_16khz_mono_int16(_Audio(b"", 2, 44100)) == b""
    silence = np.zeros(4410, dtype=np.int16).tobytes()
    out = new._convert_audio_to_16khz_mono_int16(_Audio(silence, 2, 44100))
    assert np.all(np.frombuffer(out, dtype=np.int16) == 0)


def test_resample_short_chunk_keeps_length(new):
    # Fewer samples than FIR taps: must not pad/change length (np.convolve 'same'
    # would otherwise grow a tiny buffer to the kernel length).
    raw = np.arange(10, dtype=np.int16).tobytes()
    out = new._convert_audio_to_16khz_mono_int16(_Audio(raw, 2, 44100))
    assert len(out) // 2 == max(1, round(10 * 16000 / 44100))


def test_fast_path_still_unchanged(new):
    raw = np.arange(-100, 100, dtype=np.int16).tobytes()
    assert new._convert_audio_to_16khz_mono_int16(_Audio(raw, 2, 16000)) == raw

# --- Opt-in broad-vocabulary detector ------------------------------------
def test_open_vocab_yoloe_sets_fixed_classes(new, monkeypatch):
    record = {}

    class FakeYOLOE:
        def __init__(self, name):
            record["name"] = name
            self.names = {0: "chair", 1: "stool"}

        def get_text_pe(self, names):
            record["prompt"] = list(names)
            return "embeddings"

        def set_classes(self, names, embeddings):
            record["classes"] = list(names)
            record["embeddings"] = embeddings

        def fuse(self, verbose=False):
            record["fused"] = True

    monkeypatch.setattr(new, "YOLOE", FakeYOLOE)
    monkeypatch.setattr(new, "OPEN_VOCAB_ENABLED", True)
    monkeypatch.setattr(new, "OPEN_VOCAB_MODEL", "yoloe-11s-seg.pt")
    monkeypatch.setattr(new, "OPEN_VOCAB_PROMPT_FREE", False)
    monkeypatch.setattr(new, "OPEN_VOCAB_CLASSES", ("chair", "stool"))
    monkeypatch.setattr(new, "OPEN_VOCAB_IMG_SIZE", 384)
    monkeypatch.setattr(new, "DETECTION_CLASSES", new.YOLO_CLASSES_TO_KEEP)
    monkeypatch.setattr(new, "DETECTION_IMG_SIZE", 320)

    model = new.load_detection_model()

    assert isinstance(model, FakeYOLOE)
    assert record["classes"] == ["chair", "stool"]
    assert record["embeddings"] == "embeddings"
    assert record["fused"] is True
    assert new.DETECTION_CLASSES is None
    assert new.DETECTION_IMG_SIZE == 384


def test_open_vocab_prompt_free_does_not_set_classes(new, monkeypatch):
    record = {}

    class FakeYOLOE:
        def __init__(self, name):
            self.names = {0: "chair"}

        def set_classes(self, *args):
            record["set_classes"] = True

        def fuse(self, verbose=False):
            pass

    monkeypatch.setattr(new, "YOLOE", FakeYOLOE)
    monkeypatch.setattr(new, "OPEN_VOCAB_ENABLED", True)
    monkeypatch.setattr(new, "OPEN_VOCAB_MODEL", "yoloe-11s-seg-pf.pt")
    monkeypatch.setattr(new, "OPEN_VOCAB_PROMPT_FREE", False)

    new.load_detection_model()

    assert "set_classes" not in record


def test_run_yolo_omits_coco_filter_for_open_vocab(new, monkeypatch):
    calls = {}

    class FakeModel:
        def predict(self, frame, **kwargs):
            calls.update(kwargs)
            return "result"

    monkeypatch.setattr(new, "USE_TRACKING", False)
    monkeypatch.setattr(new, "DETECTION_CLASSES", None)
    monkeypatch.setattr(new, "DETECTION_IMG_SIZE", 384)

    assert new.run_yolo(FakeModel(), object()) == "result"
    assert "classes" not in calls
    assert calls["imgsz"] == 384
