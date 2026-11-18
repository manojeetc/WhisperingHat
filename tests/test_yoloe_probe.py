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
                association_config=app.naming_replay_config(),
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
    medium = FakeModel({56: "chair"}, 56)
    yoloe = FakeModel({0: "laundry basket"}, 0)
    args = SimpleNamespace(warmup=2, repeats=3, ambiguity_iou=.75, ambiguity_score_gap=.1)
    variants = probe.comparison_variants(baseline, medium, yoloe, [320, 384])
    report = probe.run(new, variants, [path], tmp_path, args, {})
    item = report["frames"][0]
    assert item["live_alert_replay"]["status"] == "not evaluated"
    assert not item["tof_context_supplied"]
    live, raw, prompted = [item["models"][name] for name in ("yolo11n_320", "yolo11n_640", "yoloe_640")]
    assert live["detections"][0]["label"] == "bed"
    assert raw["detections"][0]["label"] == "bed"
    assert prompted["detections"][0]["label"] == "laundry basket"
    assert len(baseline.calls) == 15 and len(yoloe.calls) == len(medium.calls) == 5
    for call in baseline.calls + medium.calls + yoloe.calls:
        assert call["conf"] == new.CONFIDENCE == .4
        assert call["imgsz"] in (320, 384, 640) and call["device"] == "cpu"
        assert not call["save"]
        assert call["classes"] is None
    assert baseline.calls[0]["imgsz"] == 320
    assert baseline.calls[5]["imgsz"] == 640
    assert baseline.calls[10]["imgsz"] == 384
    assert baseline.calls[-1]["classes"] is None
    assert len(raw["warmed_ms"]) == 3 and raw["median_ms"] >= 0
    assert cv2.imread(item["annotated"]).shape == (1632, 1280, 3)
    assert report["counts"]["yolo11m_640"]["not_evaluated"] == 1
    assert len(item["input_sha256"]) == 64
    json.dumps(report, allow_nan=False)


def row(label, confidence=.6, box=(250, 250, 390, 470)):
    return dict(label=label, confidence=confidence, xyxy=list(box))


def test_current_replay_and_ambiguity_tradeoff_counts(new):
    context = context_for(new)
    context["hazard"] = dict(approved=True, source="tof_unknown", distance_m=.6)
    snapshot = probe.snapshot_from(context, new, [640, 480])
    args = SimpleNamespace(ambiguity_iou=.75, ambiguity_score_gap=.1)
    frames = []
    cases = [
        ([row("laptop", .62), row("tv", .6, (251, 251, 391, 470))], ["bed"], "wrong_names", True, False),
        ([row("laptop", .62), row("tv", .6)], ["laptop"], "correct_names", False, True),
        ([row("chair", .9)], ["chair"], "correct_names", False, False),
        ([row("bag of chips", .99, (0, 250, 80, 470))], [], "abstentions", False, False),
    ]
    for detections, expected, grade, prevented, lost in cases:
        context["expected_names"] = expected
        result = dict(detections=detections)
        naming = probe.replay(new, result, [640, 480], context, snapshot, args)
        assert naming["current_grade"] == grade
        assert naming["wrong_name_prevented"] is prevented
        assert naming["correct_name_lost"] is lost
        result["naming_replay"] = naming
        frames.append(dict(models=dict(model=result)))
    totals = probe.summarize(frames)["model"]
    assert totals["current"] == dict(correct_names=2, wrong_names=1, abstentions=1, unscored_names=0)
    assert totals["ambiguity"] == dict(correct_names=1, wrong_names=0, abstentions=3, unscored_names=0)
    assert totals["wrong_names_prevented"] == totals["correct_names_lost"] == 1
    assert totals["labeled_frames"] == 4
    json.dumps(probe.json_safe(frames), allow_nan=False)  # live-style float32 box diagnostics


@pytest.mark.parametrize("other_label,other_score,other_box,accepted,abstains", [
    ("tv", .5, (251, 251, 391, 470), True, True),
    ("laptop", .59, (251, 251, 391, 470), True, False),
    ("tv", .41, (251, 251, 391, 470), True, False),
    ("tv", .59, (400, 250, 540, 470), True, False),
    ("tv", .59, (251, 251, 391, 470), False, False),
])
def test_ambiguity_requires_different_eligible_labels_similar_scores_and_boxes(other_label, other_score, other_box, accepted, abstains):
    winner = dict(**row("laptop", float(np.float32(.6))), distance_m=.6, accepted=True)
    other = dict(**row(other_label, other_score, other_box), distance_m=.6, accepted=accepted)
    guarded = probe.ambiguity_replay(dict(candidates=[winner, other]))
    assert (guarded["selected_name"] is None) is abstains


def test_missing_context_is_not_an_abstention_or_ground_truth(new):
    result = dict(detections=[])
    args = SimpleNamespace(ambiguity_iou=.75, ambiguity_score_gap=.1)
    assert probe.replay(new, result, [640, 480], None, None, args)["status"] == "not evaluated"
    assert probe.grade("bed", None) == "unscored_names"
    assert probe.grade(None, ["bed"]) == "abstentions"
    assert probe.box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0


def test_capture_sidecar_is_loaded_and_annotation_overlay_keeps_sensor_data(new, tmp_path):
    path = tmp_path / "frame-000003.png"
    path.write_bytes(b"test")
    context = dict(context_for(new), kind="whisper_naming_capture_v1", expected_names=None)
    path.with_suffix(".json").write_text(json.dumps(context), encoding="utf-8")
    loaded = probe.load_contexts([path], {path.name: dict(expected_names=["bed"])})[str(path)]
    assert loaded["expected_names"] == ["bed"]
    assert loaded["tof_snapshot"] == context["tof_snapshot"]
    context["image_sha256"] = "wrong"
    path.with_suffix(".json").write_text(json.dumps(context), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        probe.load_contexts([path], {})


def test_tensor_observer_uses_active_backend_and_is_removed_between_sizes(new):
    class Core:
        def __init__(self):
            self.hooks = []

        def register_forward_pre_hook(self, callback):
            self.hooks.append(callback)
            return SimpleNamespace(remove=lambda: self.hooks.remove(callback))

    class Model:
        names = {}

        def __init__(self):
            self.model = Core()  # not the active copied/fused backend
            self.predictor = SimpleNamespace(model=SimpleNamespace(model=Core()), args=None)

        def predict(self, frame, **kwargs):
            shape = (1, 3, 256, 320) if kwargs["imgsz"] == 320 else (1, 3, 480, 640)
            for callback in self.predictor.model.model.hooks:
                callback(None, [SimpleNamespace(shape=shape)])
            return []

    model = Model()
    args = SimpleNamespace(warmup=1, repeats=2)
    for size, expected in [(320, (1, 3, 256, 320)), (640, (1, 3, 480, 640))]:
        result = probe.benchmark(new, model, np.zeros((480, 640, 3), np.uint8), None, args, None, size)
        assert result["tensor_shapes"] == [expected]
        assert model.model.hooks == model.predictor.model.model.hooks == []
