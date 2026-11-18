"""Offline naming comparison. Never starts sensors, tracking, alerts, or speech."""
import argparse
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


REPO = Path(__file__).resolve().parent
DEFAULT_LABELS = "bed,chair,bag of chips,cardboard box,laundry basket"
CONTEXT_CONFIG = (
    "ENABLE_TOF", "TOF_MOUNT_CALIBRATED", "TOF_STALE_SECONDS",
    "TOF_OVERLAP_WIDTH_FRACTION", "TOF_OVERLAP_HEIGHT_FRACTION",
    "TOF_OVERLAP_CENTER_X_OFFSET", "TOF_OVERLAP_CENTER_Y_OFFSET",
    "TOF_BOX_X_INSET_FRACTION", "TOF_BOX_Y_START_FRACTION",
    "TOF_CELL_PADDING", "TOF_MIN_VALID_CELLS", "TOF_DISTANCE_PERCENTILE",
    "TOF_BOX_CLAMP_LOW_OBJECTS",
)


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
    if not isinstance(context, dict) or not isinstance(context.get("tof_snapshot"), dict):
        raise ValueError("each context requires a tof_snapshot object")
    expected_config = {key: getattr(app, key) for key in CONTEXT_CONFIG}
    if context.get("association_config") != expected_config or context.get("frame_size") != list(shape):
        raise ValueError("context must match the checkout's association_config and the input frame_size")
    snapshot = dict(context["tof_snapshot"])
    grid = np.asarray(snapshot["grid_mm"], dtype=np.float32)
    age = snapshot.get("age_seconds")
    if grid.shape != (8, 8) or np.isinf(grid).any() or (grid[np.isfinite(grid)] <= 0).any():
        raise ValueError("grid_mm must be 8x8, positive millimetres or null for invalid cells")
    if type(snapshot.get("fresh")) is not bool or not isinstance(age, (int, float)) or not np.isfinite(age) or age < 0:
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


def benchmark(app, model, frame, classes, args, snapshot):
    kwargs = dict(verbose=False, conf=app.CONFIDENCE, imgsz=args.imgsz,
                  device="cpu", classes=classes, save=False)
    # Unrelated stills have no valid persistent tracker history. Both models use predict.
    for _ in range(args.warmup):
        model.predict(frame, **kwargs)
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        results = model.predict(frame, **kwargs)
        samples.append((time.perf_counter() - started) * 1000)
    names = model.names
    ids = {name: int(index) for index, name in (names.items() if isinstance(names, dict) else enumerate(names))}
    candidates = []
    for label, confidence, *rest in app.extract_detections(results, names):
        box = [float(value) for value in rest[:4]]
        candidates.append(dict(label=label, class_id=ids[label], confidence=float(confidence), xyxy=box,
                               checks=candidate_checks(app, label, box, (frame.shape[1], frame.shape[0]),
                                                       classes, ids[label], snapshot)))
    effective = getattr(getattr(model, "predictor", None), "args", None)
    return dict(warmed_ms=samples, median_ms=statistics.median(samples), detections=candidates,
                prediction_args={key: getattr(effective, key, None) for key in ("conf", "imgsz", "iou", "max_det", "rect", "half")},
                classes=classes)


def annotated(frame, result, title):
    scale = 640 / frame.shape[1]
    image = cv2.resize(frame, (640, round(frame.shape[0] * scale)))
    for row in result["detections"]:
        allowed = row["checks"]["live_class_filter"]["accepted"]
        color = (0, 210, 0) if allowed else (0, 140, 255)
        x1, y1, x2, y2 = [int(value * scale) for value in row["xyxy"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{row['label']} {row['confidence']:.2f}" + (" FILTERED" if not allowed else ""),
                    (max(0, x1), max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    image = cv2.copyMakeBorder(image, 44, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(image, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    cv2.putText(image, f"CPU predict median {result['median_ms']:.1f} ms", (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    return image


def run(app, baseline, yoloe, files, output, args, contexts):
    report = dict(association_config={key: getattr(app, key) for key in CONTEXT_CONFIG}, frames=[])
    variants = (("baseline_live", baseline, app.YOLO_CLASSES_TO_KEEP),
                ("baseline_unfiltered", baseline, None), ("yoloe", yoloe, None))
    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"could not decode original frame: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        shape = [frame.shape[1], frame.shape[0]]
        snapshot = snapshot_from(contexts.get(str(path), contexts.get(path.name)), app, shape)
        item = dict(input=str(path), input_sha256=digest, frame_size=shape,
                    provenance="saved original; whether it coincides with an unknown warning is not established",
                    tof_context_supplied=snapshot is not None,
                    live_alert_replay=dict(status="not evaluated", reason="still-image predictions lack temporal sensor/tracking and alert history"),
                    models={})
        panels = []
        for name, model, classes in variants:
            result = benchmark(app, model, frame, classes, args, snapshot)
            # Explain how the raw diagnostic differs from the actual live whitelist.
            if name == "baseline_unfiltered":
                for row in result["detections"]:
                    row["checks"] = candidate_checks(app, row["label"], row["xyxy"], shape,
                                                      app.YOLO_CLASSES_TO_KEEP, row["class_id"], snapshot)
            item["models"][name] = result
            panels.append(annotated(frame, result, name))
            print(f"{path.name} {name}: {result['median_ms']:.1f} ms "
                  + str([(d['label'], round(d['confidence'], 3)) for d in result['detections']]), flush=True)
        annotated_path = output / f"{path.stem}-{digest[:8]}-comparison.jpg"
        if not cv2.imwrite(str(annotated_path), np.hstack(panels)):
            raise OSError(f"could not write {annotated_path}")
        item["annotated"] = str(annotated_path)
        report["frames"].append(item)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="original image(s) or folder(s)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "whisper-yoloe-cache")
    parser.add_argument("--checkpoint", default="yoloe-11s-seg.pt")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="comma-separated text prompts")
    parser.add_argument("--imgsz", type=int, default=None, help="defaults to repo baseline 320")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--context", type=Path, help="optional synchronized ToF context JSON; never inferred from a photo")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or (args.imgsz is not None and args.imgsz < 32):
        parser.error("warmup/repeats must be positive; imgsz must be at least 32")
    labels = [label.strip() for label in args.labels.split(",")]
    if not all(labels) or len(set(labels)) != len(labels):
        parser.error("labels must be nonempty and unique")
    files, skipped = images_from(args.inputs)
    contexts = json.loads(args.context.resolve().read_text(encoding="utf-8")) if args.context else {}
    if not isinstance(contexts, dict):
        parser.error("context must map image paths or filenames to context objects")
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
    args.imgsz = args.imgsz or app.MODEL_PROFILES["baseline"]["img_size"]
    baseline_path = REPO / app.MODEL_PROFILES["baseline"]["model"]
    if not baseline_path.is_file():
        raise FileNotFoundError(f"repo baseline weights missing: {baseline_path}")
    baseline = YOLO(str(baseline_path))
    yoloe = YOLOE(str(checkpoint))
    yoloe.set_classes(labels, yoloe.get_text_pe(labels))
    for model in (baseline, yoloe):
        model.fuse(verbose=False)
    report = run(app, baseline, yoloe, files, output, args, contexts)
    report.update(labels=labels, checkpoint=str(checkpoint), baseline=str(baseline_path),
                  confidence=app.CONFIDENCE, imgsz=args.imgsz, warmup=args.warmup, repeats=args.repeats,
                  ultralytics=ultralytics.__version__, torch=torch.__version__, opencv=cv2.__version__,
                  platform=platform.platform(), python=platform.python_version(), device="cpu", torch_threads=torch.get_num_threads(),
                  baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), skipped_ocr_inputs=skipped,
                  baseline_vocabulary=baseline.names, live_class_filter=app.YOLO_CLASSES_TO_KEEP,
                  known_object_classes=sorted(app.KNOWN_OBJECT_CLASSES),
                  prompt_known_names={label: label in app.KNOWN_OBJECT_CLASSES for label in labels},
                  timing="warmed predict wall time including preprocessing, inference, NMS/masks and Results; excludes loading, prompt encoding, drawing and I/O",
                  scope="Windows/host CPU prediction only; no tracker, risk, alert, speech, or Pi performance evaluation")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report and private annotations: {output}", flush=True)


if __name__ == "__main__":
    main()
