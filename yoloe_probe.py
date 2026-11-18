"""Offline naming comparison. Never starts sensors, tracking, alerts, or speech."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import cv2
import numpy as np
from naming_capture import json_safe


REPO = Path(__file__).resolve().parent
DEFAULT_LABELS = "bed,chair,bag of chips,cardboard box,laundry basket"


def outside_git(path):
    path = Path(path).resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError(f"weights and outputs must stay outside Git: {path}")
    return path


def images_from(paths):
    files, skipped = set(), []
    for path in map(Path, paths):
        if not path.exists():
            raise ValueError(f"input does not exist: {path}")
        for item in (path.rglob("*") if path.is_dir() else [path]):
            if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                if "tesseract-input" in item.stem.lower():
                    skipped.append(str(item.resolve()))
                else:
                    files.add(item.resolve())
    if not files:
        raise ValueError("no original images found; OCR Tesseract inputs are excluded")
    return sorted(files), skipped


def snapshot_from(context, app, shape):
    if context is None:
        return None
    if isinstance(context, dict) and "tof_snapshot" not in context:
        return None
    if not isinstance(context, dict) or not isinstance(context.get("tof_snapshot"), dict):
        raise ValueError("each context requires a tof_snapshot object")
    expected_config = app.naming_replay_config()
    if context.get("association_config") != expected_config or context.get("frame_size") != list(shape):
        raise ValueError("context must match the checkout's association_config and the input frame_size")
    snapshot = dict(context["tof_snapshot"])
    if snapshot.get("fresh") is False and snapshot.get("grid_mm") is None:
        return None
    grid = np.asarray(snapshot.get("grid_mm"), dtype=np.float32)
    age = snapshot.get("age_seconds")
    if grid.shape != (8, 8) or np.isinf(grid).any() or (grid[np.isfinite(grid)] <= 0).any():
        raise ValueError("grid_mm must be 8x8, positive millimetres or null for invalid cells")
    if type(snapshot.get("fresh")) is not bool or type(age) not in (int, float) or not np.isfinite(age) or age < 0:
        raise ValueError("tof_snapshot requires a boolean fresh flag and nonnegative finite age_seconds")
    if snapshot["fresh"] and age > app.TOF_STALE_SECONDS:
        raise ValueError("fresh snapshot contradicts age_seconds")
    snapshot["grid_mm"] = grid
    return snapshot


def candidate_checks(app, label, box, shape, allowed_ids, class_id, snapshot=None):
    width, height = shape
    cx = (box[0] + box[2]) / 2
    inside, position = app.corridor_geometry_for_point(cx, box[3], width, height)
    class_allowed = allowed_ids is None or class_id in allowed_ids
    known = label in app.KNOWN_OBJECT_CLASSES
    rejected = [reason for passed, reason in (
        (class_allowed, "class excluded by live whitelist"),
        (known, "exact name absent from known classes"),
        (inside, "box outside corridor"),
    ) if not passed]
    checks = dict(
        live_class_filter=dict(accepted=class_allowed, reason="allowed" if class_allowed else "COCO ID excluded by live class whitelist"),
        known_class_set=dict(accepted=known, reason="exact name in KNOWN_OBJECT_CLASSES" if known else "exact name absent; no synonym normalization"),
        corridor=dict(in_corridor=bool(inside), position=position),
        camera_known_distance_path=dict(accepted=bool(class_allowed and known and inside),
                                        reason="; ".join(rejected) if rejected else "eligible; camera distance not evaluated"),
        tof_association=dict(status="not evaluated", reason="no synchronized ToF grid and calibration context"),
        unknown_arbitration=dict(status="not evaluated", reason="requires contemporaneous smoothed path/known distances and temporal state"),
        spoken_alert=dict(status="not evaluated", reason="no tracking/motion, risk, competing hazards, stabilization or speech history"),
    )
    if snapshot is not None:
        diagnostics = {}
        match = app.tof_distance_for_box(*box, width, height, snapshot=snapshot, diagnostics=diagnostics)
        checks["tof_association"] = dict(status="accepted" if match else "rejected", match=match, **diagnostics)
    return checks


def benchmark(app, model, frame, classes, args, snapshot, imgsz, legacy_classes=None):
    kwargs = dict(verbose=False, conf=app.CONFIDENCE, imgsz=imgsz,
                  device="cpu", classes=classes, save=False)
    # Unrelated stills have no valid persistent tracker history. Both models use predict.
    shapes, samples = [], []
    for _ in range(args.warmup):
        model.predict(frame, **kwargs)
    # AutoBackend can copy/fuse the supplied module during setup. Observe the
    # active backend after warmup, including when the nano model changes size.
    core = getattr(getattr(getattr(model, "predictor", None), "model", None), "model", None)
    hook = core.register_forward_pre_hook(lambda module, inputs: shapes.append(list(inputs[0].shape))) if hasattr(core, "register_forward_pre_hook") else None
    try:
        for _ in range(args.repeats):
            started = time.perf_counter()
            results = model.predict(frame, **kwargs)
            samples.append((time.perf_counter() - started) * 1000)
    finally:
        if hook is not None:
            hook.remove()
    names = model.names
    ids = {name: int(index) for index, name in (names.items() if isinstance(names, dict) else enumerate(names))}
    candidates = []
    for label, confidence, *rest in app.extract_detections(results, names):
        box = [float(value) for value in rest[:4]]
        candidates.append(dict(label=label, class_id=ids[label], confidence=float(confidence), xyxy=box,
                               legacy_other_feature_checks=candidate_checks(app, label, box, (frame.shape[1], frame.shape[0]),
                                                                            legacy_classes, ids[label], snapshot)))
    effective = getattr(getattr(model, "predictor", None), "args", None)
    return dict(warmed_ms=samples, median_ms=statistics.median(samples), detections=candidates,
                requested_imgsz=imgsz, tensor_shapes=sorted(set(tuple(shape) for shape in shapes)),
                prediction_args={key: getattr(effective, key, None) for key in ("conf", "imgsz", "iou", "max_det", "rect", "half")},
                classes=classes)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.0


def ambiguity_replay(evaluation, iou_threshold=.75, score_gap=.10):
    candidates = [row for row in evaluation["candidates"] if row["accepted"]]
    if not candidates:
        return dict(selected_name=None, conflicts=[], reason="no accepted name")
    winner = min(candidates, key=lambda row: (row["distance_m"], -row["confidence"], row["label"]))
    conflicts = []
    for other in candidates:
        iou = box_iou(winner["xyxy"], other["xyxy"])
        gap = abs(winner["confidence"] - other["confidence"])
        if other["label"] != winner["label"] and iou >= iou_threshold and gap <= score_gap + 1e-6:
            conflicts.append(dict(label=other["label"], confidence=other["confidence"], iou=iou, score_gap=gap))
    return dict(selected_name=None if conflicts else winner["label"], conflicts=conflicts,
                reason="ambiguous overlapping labels" if conflicts else "current name retained")


def grade(name, expected):
    if name is None:
        return "abstentions"
    if expected is None:
        return "unscored_names"
    return "correct_names" if name in expected else "wrong_names"


def replay(app, result, shape, context, snapshot, args):
    hazard = (context or {}).get("hazard", {})
    if snapshot is None or not snapshot["fresh"] or hazard.get("approved") is not True or hazard.get("source") not in {"tof", "tof_unknown"}:
        return dict(status="not evaluated", reason="requires an approved hazard, synchronized ToF and matching configuration")
    distance = hazard.get("distance_m")
    if type(distance) not in (int, float) or not np.isfinite(distance) or distance <= 0:
        raise ValueError("approved hazard requires a positive finite distance_m")
    expected = context.get("expected_names")
    if expected is not None and (not isinstance(expected, list) or not all(isinstance(name, str) and name.strip() for name in expected)):
        raise ValueError("expected_names must be null (unlabeled), [] (generic desired), or a list of acceptable exact labels")
    rows = result["detections"]
    detections = [(row["label"], row["confidence"], *np.asarray(row["xyxy"], dtype=np.float32), -1) for row in rows]
    evaluation = app.evaluate_obstacle_names(detections, *shape, distance, snapshot, app.tof_overlap_rectangle(*shape))
    guarded = ambiguity_replay(evaluation, args.ambiguity_iou, args.ambiguity_score_gap)
    current_name, guarded_name = evaluation["selected_name"], guarded["selected_name"]
    current_grade, guarded_grade = grade(current_name, expected), grade(guarded_name, expected)
    return dict(status="evaluated", scope="conditional naming only; original hazard approval held fixed", hazard=hazard,
                observed_spoken=context.get("spoken_message"), speech_queued=context.get("speech_queued"),
                expected_names=expected, current=evaluation, ambiguity=guarded,
                current_grade=current_grade, ambiguity_grade=guarded_grade,
                wrong_name_prevented=current_grade == "wrong_names" and guarded_name is None,
                correct_name_lost=current_grade == "correct_names" and guarded_name is None,
                selected_text_current=current_name or "unknown obstacle",
                selected_text_ambiguity=guarded_name or "unknown obstacle")


def summarize(frames):
    totals = {}
    for frame in frames:
        for name, result in frame["models"].items():
            counts = totals.setdefault(name, dict(current=dict(correct_names=0, wrong_names=0, abstentions=0, unscored_names=0),
                                                  ambiguity=dict(correct_names=0, wrong_names=0, abstentions=0, unscored_names=0),
                                                  not_evaluated=0, labeled_frames=0, wrong_names_prevented=0, correct_names_lost=0))
            naming = result["naming_replay"]
            if naming["status"] != "evaluated":
                counts["not_evaluated"] += 1
                continue
            for policy in ("current", "ambiguity"):
                counts[policy][naming[f"{policy}_grade"]] += 1
            counts["labeled_frames"] += naming["expected_names"] is not None
            counts["wrong_names_prevented"] += naming["wrong_name_prevented"]
            counts["correct_names_lost"] += naming["correct_name_lost"]
    return totals


def load_contexts(files, manifest):
    contexts = {}
    counts = Counter(path.name for path in files)
    for path in files:
        if counts[path.name] > 1 and path.name in manifest and str(path) not in manifest:
            raise ValueError(f"ambiguous annotation filename; use absolute image paths: {path.name}")
        sidecar = path.with_suffix(".json")
        context = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        if not isinstance(context, dict) or (context and context.get("kind") != "whisper_naming_capture_v1"):
            raise ValueError(f"not a naming capture sidecar: {sidecar}")
        if context.get("image_sha256") and context["image_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"capture image hash mismatch: {path}")
        patch = manifest.get(str(path), manifest.get(path.name, {}))
        if not isinstance(patch, dict):
            raise ValueError("context entries must be objects")
        context.update(patch)
        if context:
            contexts[str(path)] = context
    return contexts


def comparison_variants(nano, medium, yoloe, live_sizes):
    variants = [("yolo11n_320", nano, 320), ("yolo11n_640", nano, 640),
                ("yolo11m_640", medium, 640), ("yoloe_640", yoloe, 640)]
    variants.extend((f"yolo11n_live_{size}", nano, size) for size in sorted(set(live_sizes) - {320, 640}))
    return variants


def annotated(frame, result, title):
    scale = 640 / frame.shape[1]
    image = cv2.resize(frame, (640, round(frame.shape[0] * scale)))
    for row in result["detections"]:
        color = (0, 210, 0)
        x1, y1, x2, y2 = [int(value * scale) for value in row["xyxy"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{row['label']} {row['confidence']:.2f}",
                    (max(0, x1), max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    image = cv2.copyMakeBorder(image, 64, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(image, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    cv2.putText(image, f"CPU predict median {result['median_ms']:.1f} ms", (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    naming = result["naming_replay"]
    message = "naming: not evaluated" if naming["status"] != "evaluated" else f"name={naming['current']['selected_name'] or 'generic'} guarded={naming['ambiguity']['selected_name'] or 'generic'}"
    cv2.putText(image, message, (8, 55), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return image


def run(app, variants, files, output, args, contexts):
    report = dict(association_config=app.naming_replay_config(), frames=[])
    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"could not decode original frame: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        shape = [frame.shape[1], frame.shape[0]]
        context = contexts.get(str(path), contexts.get(path.name))
        snapshot = snapshot_from(context, app, shape)
        item = dict(input=str(path), input_sha256=digest, frame_size=shape,
                    provenance=("synthetic fixture; not a hat frame" if (context or {}).get("synthetic_fixture") else
                                "captured naming frame" if (context or {}).get("kind") == "whisper_naming_capture_v1" else
                                "unverified original; not established as an alert frame"),
                    tof_context_supplied=snapshot is not None,
                    live_alert_replay=dict(status="not evaluated", reason="still-image predictions lack temporal sensor/tracking and alert history"),
                    models={})
        panels = []
        for name, model, imgsz in variants:
            result = benchmark(app, model, frame, None, args, snapshot, imgsz,
                               legacy_classes=app.YOLO_CLASSES_TO_KEEP if name.startswith("yolo11") else None)
            result["naming_replay"] = replay(app, result, shape, context, snapshot, args)
            item["models"][name] = result
            panels.append(annotated(frame, result, name))
            print(f"{path.name} {name}: {result['median_ms']:.1f} ms "
                  + str([(d['label'], round(d['confidence'], 3)) for d in result['detections']]), flush=True)
        annotated_path = output / f"{path.stem}-{digest[:8]}-comparison.jpg"
        # Two columns keep the requested four-way comparison legible.
        if len(panels) % 2:
            panels.append(np.zeros_like(panels[0]))
        grid = np.vstack([np.hstack(panels[index:index + 2]) for index in range(0, len(panels), 2)])
        if not cv2.imwrite(str(annotated_path), grid):
            raise OSError(f"could not write {annotated_path}")
        item["annotated"] = str(annotated_path)
        report["frames"].append(item)
    report["counts"] = summarize(report["frames"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="original image(s) or folder(s)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "whisper-yoloe-cache")
    parser.add_argument("--checkpoint", default="yoloe-11s-seg.pt")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="comma-separated text prompts")
    parser.add_argument("--live-imgsz", type=int, help="additional baseline size; defaults to current checkout runtime configuration")
    parser.add_argument("--ambiguity-iou", type=float, default=.75)
    parser.add_argument("--ambiguity-score-gap", type=float, default=.10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--context", type=Path, help="optional synchronized ToF context JSON; never inferred from a photo")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or (args.live_imgsz is not None and args.live_imgsz < 32):
        parser.error("warmup/repeats must be positive; live-imgsz must be at least 32")
    if not 0 < args.ambiguity_iou <= 1 or not 0 <= args.ambiguity_score_gap <= 1:
        parser.error("ambiguity-iou must be in (0, 1]; ambiguity-score-gap in [0, 1]")
    labels = [label.strip() for label in args.labels.split(",")]
    if not all(labels) or len(set(labels)) != len(labels):
        parser.error("labels must be nonempty and unique")
    files, skipped = images_from(args.inputs)
    contexts = json.loads(args.context.resolve().read_text(encoding="utf-8")) if args.context else {}
    if not isinstance(contexts, dict):
        parser.error("context must map image paths or filenames to context objects")
    contexts = load_contexts(files, contexts)
    cache, output_root = outside_git(args.cache), outside_git(args.output)
    for directory in (cache, output_root):
        directory.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="yoloe-probe-", dir=output_root))
    requested = Path(args.checkpoint)
    if not requested.exists() and (requested.is_absolute() or requested.parent != Path(".")):
        parser.error(f"checkpoint file does not exist: {requested}")
    checkpoint = requested.resolve() if requested.exists() else cache / requested.name
    outside_git(checkpoint.parent)
    if "-pf" in checkpoint.stem:
        parser.error("use a text-prompt checkpoint, not a prompt-free -pf checkpoint")
    # Ultralytics downloads the prompt encoder into cwd. Keep every new asset out of Git.
    os.environ["YOLO_CONFIG_DIR"] = str(cache / "settings")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["ULTRALYTICS_AUTOINSTALL"] = "False"
    os.chdir(cache)
    import whisper as app
    import torch
    import ultralytics
    from ultralytics import YOLO, YOLOE
    checkout_size = app.OPEN_VOCAB_IMG_SIZE if app.OPEN_VOCAB_ENABLED else app.IMG_SIZE
    live_sizes = [checkout_size]
    if args.live_imgsz is not None:
        live_sizes.append(args.live_imgsz)
    for context in contexts.values():
        size = context.get("naming", {}).get("imgsz")
        if size is not None:
            if type(size) is not int or size < 32:
                parser.error("captured naming imgsz must be an integer of at least 32")
            live_sizes.append(size)
    baseline_path = REPO / app.MODEL_PROFILES["baseline"]["model"]
    if not baseline_path.is_file():
        raise FileNotFoundError(f"repo baseline weights missing: {baseline_path}")
    baseline = YOLO(str(baseline_path))
    medium_path = cache / "yolo11m.pt"
    medium = YOLO(str(medium_path))
    yoloe = YOLOE(str(checkpoint))
    yoloe.set_classes(labels, yoloe.get_text_pe(labels))
    for model in (baseline, medium, yoloe):
        model.fuse(verbose=False)
    report = run(app, comparison_variants(baseline, medium, yoloe, live_sizes), files, output, args, contexts)
    report.update(labels=labels, checkpoint=str(checkpoint), baseline=str(baseline_path),
                  confidence=app.CONFIDENCE, live_sizes=sorted(set(live_sizes)), warmup=args.warmup, repeats=args.repeats,
                  live_configuration=dict(model=app.MODEL_NAME, configured_imgsz=checkout_size,
                                          camera_frame_size=[app.CAMERA_WIDTH, app.CAMERA_HEIGHT],
                                          source="checkout configuration; captured sidecars verify actual Pi naming input/settings"),
                  captured_live_settings={path: ctx.get("naming") for path, ctx in contexts.items()},
                  ambiguity_rule=dict(iou=args.ambiguity_iou, absolute_confidence_gap=args.ambiguity_score_gap, scope="offline only"),
                  ultralytics=ultralytics.__version__, torch=torch.__version__, opencv=cv2.__version__,
                  platform=platform.platform(), python=platform.python_version(), device="cpu", torch_threads=torch.get_num_threads(),
                  baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                  medium=str(medium_path), medium_sha256=hashlib.sha256(medium_path.read_bytes()).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), skipped_ocr_inputs=skipped,
                  baseline_vocabulary=baseline.names, live_class_filter=app.YOLO_CLASSES_TO_KEEP,
                  known_object_classes=sorted(app.KNOWN_OBJECT_CLASSES),
                  prompt_known_names={label: label in app.KNOWN_OBJECT_CLASSES for label in labels},
                  checks_scope="legacy_other_feature_checks describe existing tracking/risk features; only naming_replay evaluates the unrestricted naming pass",
                  timing="warmed predict wall time including preprocessing, inference, NMS/masks and Results; excludes loading, prompt encoding, drawing and I/O",
                  scope="host CPU prediction and conditional naming replay only; no hazard or speech-queue replay, no Pi time-to-speech evaluation")
    (output / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report and private annotations: {output}", flush=True)


if __name__ == "__main__":
    main()
