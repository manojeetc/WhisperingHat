"""Generic geometry and reporting regressions; private photographs stay outside Git."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import ocr_region_compare as compare


def test_straighten_preserves_handedness_and_adds_white_margin():
    image = np.full((80, 160, 3), 255, np.uint8)
    image[10:70, 10:70] = (0, 0, 255)
    image[10:70, 90:150] = (255, 0, 0)
    crop = compare.straighten(image, [[10, 10], [149, 10], [149, 69], [10, 69]])
    assert tuple(crop[20, 20]) == (0, 0, 255)
    assert tuple(crop[20, -20]) == (255, 0, 0)
    assert (crop[:8] == 255).all()


@pytest.mark.parametrize("quad", [
    [[0, 0], [5, 5], [0, 5], [5, 0]],
    [[0, 0], [10, 0], [10, 5], [0, 5]],
    [[0, 0], [float("nan"), 0], [5, 5], [0, 5]],
])
def test_manual_control_rejects_crossed_outside_or_nonfinite_corners(quad):
    with pytest.raises(ValueError):
        compare.straighten(np.zeros((10, 10, 3), np.uint8), quad)


def test_selector_finds_rotated_generic_line_without_ocr():
    image = np.full((320, 640, 3), 255, np.uint8)
    cv2.putText(image, "LOCAL TEXT LINE", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    image = cv2.warpAffine(image, cv2.getRotationMatrix2D((320, 160), 25, 1),
                           (640, 320), borderValue=(255, 255, 255))
    quad = compare.select_region(image)
    assert quad is not None
    direction = quad[1] - quad[0]
    assert abs(np.degrees(np.arctan2(direction[1], direction[0])) + 25) < 8
    crop = compare.straighten(image, quad)
    assert crop.shape[1] > crop.shape[0] * 2
    assert compare.select_region(np.full((320, 640, 3), 255, np.uint8)) is None


def test_reference_reports_omissions_and_reordered_lines_not_confidence():
    result = compare.text_difference("First printed line\nSecond printed line", "Second printed line\nFirst printed line")
    assert result["reordered_exact_lines"] is True
    result = compare.text_difference("First printed line\nSecond printed line", "First line\nstray words")
    assert any("printed" in edit["expected"] for edit in result["word_edits"])
    assert result["reference_lines"][1]["word_position"] is None
    assert result["line_order_assessable"] is False
    assert compare.text_difference("One physical line", "one physical\nline.")["word_edits"] == []


def test_cli_backend_preserves_tsv_order_and_filters_empty_tokens(monkeypatch):
    calls = []
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t90\tFIRST\n"
        "5\t1\t1\t1\t2\t1\t0\t5\t1\t1\t80\tSECOND\n"
        "5\t1\t1\t1\t2\t2\t0\t5\t1\t1\t95\t \n"
    )
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="tesseract test\n" if "--version" in command else tsv)
    monkeypatch.setattr(compare.subprocess, "run", run)
    text, words, _version = compare.recognize(Path("image.png"), 6, "cli", None, "eng")
    assert text == "FIRST\nSECOND"
    assert [word["text"] for word in words] == ["FIRST", "SECOND"]
    assert calls[0][0][-3:] == ["--psm", "6", "tsv"]
    assert calls[0][1]["timeout"] == 30 and calls[0][1]["encoding"] == "utf-8"


def test_report_saves_selected_regions_and_actual_reference_edits(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    cv2.imwrite(str(source), np.full((80, 160, 3), 255, np.uint8))
    original_bytes = source.read_bytes()
    monkeypatch.setattr(compare, "recognize", lambda *args: ("wrong word", [], "fake engine"))
    directory = compare.compare(source, tmp_path / "output", reference="printed paragraph")
    report = json.loads((directory / "report.json").read_text())
    assert report["automatic_found"] is False
    assert len(report["results"]) == 6  # Four full-frame modes, two center modes.
    assert all(row["comparison"]["word_edits"] for row in report["results"])
    assert all(row["total_ms"] >= row["selection_ms"] >= 0 for row in report["results"])
    assert (directory / "center.png").exists() and (directory / "regions.png").exists()
    assert source.read_bytes() == original_bytes


def test_private_crops_refused_inside_git_tree(tmp_path):
    source = tmp_path / "input.png"
    cv2.imwrite(str(source), np.full((80, 160, 3), 255, np.uint8))
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="outside Git"):
        compare.compare(source, tmp_path / "private")
    assert not (tmp_path / "private").exists()


def test_all_engine_failures_are_saved_and_raise(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    cv2.imwrite(str(source), np.full((80, 160, 3), 255, np.uint8))
    def unavailable(*args):
        raise FileNotFoundError("Tesseract unavailable")
    monkeypatch.setattr(compare, "recognize", unavailable)
    with pytest.raises(RuntimeError, match="all OCR attempts failed"):
        compare.compare(source, tmp_path / "output")
    path, = (tmp_path / "output").glob("*/report.json")
    assert all("error" in row for row in json.loads(path.read_text())["results"])
