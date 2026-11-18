import json

import cv2
import numpy as np
import pytest

from naming_capture import NamingCapture


def test_capped_lossless_pair_and_valid_json(tmp_path):
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[0, 1] = [3, 21, 234]
    original = frame.copy()
    writer = NamingCapture(tmp_path, limit=1)
    try:
        assert writer.submit(frame, dict(frame_number=3, grid=np.array([[np.nan, 600.0]])))
        frame.fill(255)  # queued bytes must remain the exact naming pixels
        assert not writer.submit(frame, dict(frame_number=4))
    finally:
        writer.close()
    images, contexts = list(tmp_path.rglob("*.png")), list(tmp_path.rglob("*.json"))
    assert len(images) == len(contexts) == 1
    np.testing.assert_array_equal(cv2.imread(str(images[0])), original)
    metadata = json.loads(contexts[0].read_text(encoding="utf-8"))
    assert metadata["grid"] == [[None, 600.0]]
    assert len(metadata["image_sha256"]) == 64


def test_full_queue_drops_without_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(NamingCapture, "_write", lambda self: self.stop.wait())
    writer = NamingCapture(tmp_path)
    try:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        assert writer.submit(frame, dict(frame_number=1))
        assert writer.submit(frame, dict(frame_number=2))
        assert not writer.submit(frame, dict(frame_number=3))
        assert writer.count == 2
    finally:
        writer.close()


def test_write_failure_does_not_escape_to_app(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    writer = NamingCapture(tmp_path)
    assert writer.submit(np.zeros((2, 2, 3), dtype=np.uint8), dict(frame_number=1))
    writer.close()
    assert "write failed" in capsys.readouterr().out


def test_invalid_destination_and_cap_are_rejected(tmp_path):
    (tmp_path / ".git").touch()
    with pytest.raises(ValueError, match="outside Git"):
        NamingCapture(tmp_path / "images")
    with pytest.raises(ValueError, match="between 1 and 20"):
        NamingCapture(tmp_path.parent, limit=21)
