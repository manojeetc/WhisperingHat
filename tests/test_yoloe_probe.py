"""Offline probe checks: no weights, downloads, hardware, tracking or speech."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import yoloe_probe as probe
from conftest import make_results


def test_inputs_deduplicate_and_exclude_ocr(tmp_path):
    original = tmp_path / "capture-original.png"
    original.touch()
    ocr = tmp_path / "capture-tesseract-input.png"
    ocr.touch()
    (tmp_path / "report.json").touch()
    files, skipped = probe.images_from([tmp_path, original])
    assert files == [original]
    assert skipped == [str(ocr)]
    with pytest.raises(ValueError, match="OCR Tesseract inputs are excluded"):
        probe.images_from([ocr])
    with pytest.raises(ValueError, match="does not exist"):
        probe.images_from([tmp_path / "missing.png"])


def test_private_output_and_weights_cannot_be_in_git(tmp_path):
    checkout = tmp_path / "repo"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: elsewhere")  # worktrees use a file
    with pytest.raises(ValueError, match="outside Git"):
        probe.outside_git(checkout / "new" / "weights.pt")
    assert probe.outside_git(tmp_path / "private") == tmp_path / "private"


@pytest.mark.parametrize("checkpoint", ["missing/subdir/model.pt", "yoloe-11s-seg-pf.pt"])
def test_invalid_checkpoints_fail_before_model_loading(tmp_path, monkeypatch, checkpoint):
    image = tmp_path / "original.png"
    image.touch()
    monkeypatch.setattr("sys.argv", ["yoloe_probe.py", str(image),
                                    "--cache", str(tmp_path / "cache"),
                                    "--output", str(tmp_path / "output"),
                                    "--checkpoint", checkpoint])
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2


def test_bed_is_filtered_and_not_a_known_name(new):
    checks = probe.candidate_checks(new, "bed", [250, 250, 390, 470], [640, 480],
                                    new.YOLO_CLASSES_TO_KEEP, 59)
    assert not checks["live_class_filter"]["accepted"]
    assert not checks["known_class_set"]["accepted"]
    assert checks["corridor"]["in_corridor"]
    assert "class excluded" in checks["camera_known_distance_path"]["reason"]
    assert checks["tof_association"]["status"] == "not evaluated"
    assert checks["unknown_arbitration"]["status"] == "not evaluated"
    assert checks["spoken_alert"]["status"] == "not evaluated"


def test_exact_prompt_names_and_corridor_are_separate_checks(new):
    chair = probe.candidate_checks(new, "chair", [250, 250, 390, 470], [640, 480],
                                   new.YOLO_CLASSES_TO_KEEP, 56)
    assert chair["camera_known_distance_path"]["accepted"]
    bag = probe.candidate_checks(new, "bag of chips", [250, 250, 390, 470], [640, 480], None, 2)
    assert bag["live_class_filter"]["accepted"]  # open-vocabulary IDs are not COCO IDs
    assert not bag["known_class_set"]["accepted"]
    outside = probe.candidate_checks(new, "chair", [0, 0, 30, 40], [640, 480], None, 1)
    assert not outside["camera_known_distance_path"]["accepted"]
    assert "outside corridor" in outside["camera_known_distance_path"]["reason"]


def context_for(app):
    return dict(frame_size=[640, 480],
                association_config={key: getattr(app, key) for key in probe.CONTEXT_CONFIG},
                tof_snapshot=dict(grid_mm=np.full((8, 8), 600).tolist(), fresh=True, age_seconds=.05))


def test_synchronized_context_uses_real_association_reasons(new, monkeypatch):
    monkeypatch.setattr(new, "ENABLE_TOF", True)
    monkeypatch.setattr(new, "TOF_MOUNT_CALIBRATED", True)
    snapshot = probe.snapshot_from(context_for(new), new, [640, 480])
    matched = probe.candidate_checks(new, "chair", [250, 250, 390, 470], [640, 480], None, 0, snapshot)
    assert matched["tof_association"]["status"] == "accepted"
    assert matched["tof_association"]["reason"].startswith("matched")
    assert matched["tof_association"]["match"]["distance_m"] == pytest.approx(.6)
    snapshot["fresh"] = False
    stale = probe.candidate_checks(new, "chair", [250, 250, 390, 470], [640, 480], None, 0, snapshot)
    assert stale["tof_association"]["status"] == "rejected"
    assert stale["tof_association"]["reason"] == "tof_not_fresh"
    assert stale["spoken_alert"]["status"] == "not evaluated"
    json.dumps(stale, allow_nan=False)


@pytest.mark.parametrize("invalid", ["size", "config", "grid", "age", "fresh", "negative", "object"])
def test_invalid_sensor_context_is_not_guessed(new, invalid):
    context = context_for(new)
    if invalid == "size":
        context["frame_size"] = [1920, 1440]
    elif invalid == "config":
        context["association_config"] = {}
    elif invalid == "grid":
        context["tof_snapshot"]["grid_mm"] = [[600]]
    elif invalid == "age":
        context["tof_snapshot"]["age_seconds"] = new.TOF_STALE_SECONDS + 1
    elif invalid == "fresh":
        context["tof_snapshot"]["fresh"] = "true"
    elif invalid == "negative":
        context["tof_snapshot"]["grid_mm"][0][0] = -1
    else:
        context = []
    with pytest.raises(ValueError):
        probe.snapshot_from(context, new, [640, 480])


def test_same_pixels_and_live_settings_with_raw_filter_diagnostic(new, tmp_path, monkeypatch):
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    pixels[30:80, 10:90] = [17, 103, 221]  # asymmetric BGR; catches flips/conversions
    path = tmp_path / "original.png"
    assert cv2.imwrite(str(path), pixels)

    class FakeModel:
        def __init__(self, names, class_id):
            self.names, self.class_id, self.calls = names, class_id, []
            self.predictor = SimpleNamespace(args=SimpleNamespace(conf=.4, imgsz=320))

        def predict(self, frame, **kwargs):
            np.testing.assert_array_equal(frame, pixels)
            self.calls.append(kwargs)
            if kwargs["classes"] is not None and self.class_id not in kwargs["classes"]:
                return []
            return make_results([[250, 250, 390, 470]], [.88], [self.class_id], None)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline probe must not evaluate live risk or speech")

    for name in ("score_risk", "should_alert", "alert_text"):
        monkeypatch.setattr(new, name, forbidden)
    baseline = FakeModel({59: "bed"}, 59)
    yoloe = FakeModel({0: "laundry basket"}, 0)
    args = SimpleNamespace(imgsz=320, warmup=2, repeats=3)
    report = probe.run(new, baseline, yoloe, [path], tmp_path, args, {})
    item = report["frames"][0]
    assert item["live_alert_replay"]["status"] == "not evaluated"
    assert not item["tof_context_supplied"]
    live, raw, prompted = [item["models"][name] for name in ("baseline_live", "baseline_unfiltered", "yoloe")]
    assert live["detections"] == []
    assert raw["detections"][0]["label"] == "bed"
    assert not raw["detections"][0]["checks"]["live_class_filter"]["accepted"]
    assert prompted["detections"][0]["label"] == "laundry basket"
    assert len(baseline.calls) == 10 and len(yoloe.calls) == 5
    for call in baseline.calls + yoloe.calls:
        assert call["conf"] == new.CONFIDENCE == .4
        assert call["imgsz"] == 320 and call["device"] == "cpu"
        assert not call["save"]
    assert baseline.calls[0]["classes"] == new.YOLO_CLASSES_TO_KEEP
    assert baseline.calls[-1]["classes"] is None
    assert len(raw["warmed_ms"]) == 3 and raw["median_ms"] >= 0
    assert cv2.imread(item["annotated"]).shape == (524, 1920, 3)
    assert len(item["input_sha256"]) == 64
    json.dumps(report, allow_nan=False)
