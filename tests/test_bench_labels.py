"""The benchmark must report correct/wrong/generic/missed separately."""
import json

import bench_pi


def test_named_detection_outcomes_are_separate(tmp_path):
    labels = {
        "chair.jpg": {"expected": ["chair"], "tof_visible": True},
        "box.jpg": {"expected": ["cardboard box", "box"], "tof_visible": True},
        "basket.jpg": {"expected": ["laundry basket"], "tof_visible": True},
        "trash.jpg": {"expected": ["trash can", "bin"], "tof_visible": True},
        "stool.jpg": {"expected": ["stool"], "tof_visible": False},
        "unlabeled.jpg": [],
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels), encoding="utf-8")
    detections = {
        "chair.jpg": ["chair"],
        "box.jpg": ["suitcase"],
        "basket.jpg": [],
        "trash.jpg": [],
        "stool.jpg": [],
        "unlabeled.jpg": ["person"],
    }

    result = bench_pi._evaluate_named_detections(str(path), detections)

    assert result["counts"] == {
        "correct": 1,
        "wrong": 1,
        "generic": 2,
        "missed": 1,
        "generic_or_missed": 0,
        "unlabeled": 1,
    }
    outcomes = {row["image"]: row["outcome"] for row in result["per_image"]}
    assert outcomes["chair.jpg"] == "correct"
    assert outcomes["box.jpg"] == "wrong"
    assert outcomes["basket.jpg"] == "generic"
    assert outcomes["trash.jpg"] == "generic"
    assert outcomes["stool.jpg"] == "missed"


def test_app_log_latency_separates_named_and_generic_hazards(tmp_path):
    log = tmp_path / "app.log"
    log.write_text(
        "[latency] hazard_to_tts_start=120.0ms message='chair close in path'\n"
        "[latency] hazard_to_tts_start=180.0ms message='chair close in path'\n"
        "[latency] hazard_to_tts_start=90.0ms message='unknown obstacle in path'\n",
        encoding="utf-8",
    )

    result = bench_pi._summarize_app_latency_logs([str(log)])

    assert result["hazard"]["count"] == 3
    assert result["hazard_named"]["count"] == 2
    assert result["hazard_named"]["median_ms"] == 150.0
    assert result["hazard_generic"]["count"] == 1
    assert result["hazard_generic"]["median_ms"] == 90.0
