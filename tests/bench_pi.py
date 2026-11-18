"""Reproducible Raspberry Pi 5 benchmark for Whispering Hat.

This script is intentionally explicit about what it measures. It does NOT
pretend that an x86 desktop run is a Pi measurement. Run it on the Pi 5 and
keep the JSON output with the commit hash.

Examples
--------
Baseline vs YOLO11s/YOLO11m on the same captured frames:

    python tests/bench_pi.py \
        --models yolo11n.pt,yolo11s.pt,yolo11m.pt \
        --images bench/scenes \
        --imgsz 320 \
        --conf 0.40 \
        --loops 10 \
        --val-data /path/to/labeled-scenes.yaml \
        --json-out bench/results-$(date +%Y%m%d-%H%M).json

Combined load (detection + TTS + offline voice listening):

    python tests/bench_pi.py \
        --models yolo11n.pt \
        --images bench/scenes \
        --combined-seconds 60 \
        --voice \
        --json-out bench/results-combined.json

Scene images must be captured with the production camera settings (640x480,
same autofocus/exposure, no HUD overlay). Use the same directory for every
candidate, otherwise the comparison is not valid.
"""
import argparse
import json
import platform
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))


def _load_hat_module():
    import importlib.util

    # The real benchmark runs where ultralytics is installed. If this is an
    # x86 host without it, fail clearly rather than producing fake numbers.
    spec = importlib.util.spec_from_file_location("whisper_bench_target", REPO / "whisper.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["whisper_bench_target"] = module
    spec.loader.exec_module(module)
    return module


def _percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (pct / 100.0) * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _run_command(command):
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return completed.stdout.strip() or completed.stderr.strip()
    except Exception as error:
        return f"unavailable: {type(error).__name__}: {error}"


def _thermal_snapshot():
    temp_text = _run_command(["vcgencmd", "measure_temp"])
    throttled_text = _run_command(["vcgencmd", "get_throttled"])
    match = re.search(r"temp=([0-9.]+)", temp_text)
    return {
        "soc_temp_c": float(match.group(1)) if match else None,
        "get_throttled": throttled_text,
    }


def _peak_ram_mb():
    try:
        status = Path("/proc/self/status").read_text()
        match = re.search(r"^VmHWM:\s+(\d+)\s+kB", status, re.MULTILINE)
        if match:
            return round(int(match.group(1)) / 1024.0, 1)
    except Exception:
        pass
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        return None


def _current_rss_mb():
    try:
        status = Path("/proc/self/status").read_text()
        match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.MULTILINE)
        if match:
            return round(int(match.group(1)) / 1024.0, 1)
    except Exception:
        pass
    return _peak_ram_mb()


def _model_size_mb(model_ref):
    path = Path(model_ref)
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 1)
    if path.is_dir():
        total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        return round(total / (1024 * 1024), 1)
    return None


def _image_files(directory):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [
        path for path in sorted(Path(directory).rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
    ]
    if not files:
        raise SystemExit(f"No images found in {directory}")
    return files


def _load_image(path):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Could not decode image: {path}")
    return image


def _latency_stats(samples_ms):
    return {
        "count": len(samples_ms),
        "median_ms": round(statistics.median(samples_ms), 2) if samples_ms else None,
        "p95_ms": round(_percentile(samples_ms, 95), 2) if samples_ms else None,
        "min_ms": round(min(samples_ms), 2) if samples_ms else None,
        "max_ms": round(max(samples_ms), 2) if samples_ms else None,
        "fps_from_median": round(1000.0 / statistics.median(samples_ms), 2) if samples_ms else None,
    }


def _benchmark_model(hat, model_ref, images, args):
    from ultralytics import YOLO

    load_started = time.perf_counter()
    model = YOLO(model_ref)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    try:
        model.fuse(verbose=False)
    except TypeError:
        pass

    def infer(frame):
        if args.tracking:
            return model.track(
                frame,
                persist=True,
                tracker=hat.TRACKER_NAME,
                verbose=False,
                conf=args.conf,
                imgsz=args.imgsz,
                device="cpu",
                classes=hat.YOLO_CLASSES_TO_KEEP,
            )
        return model.predict(
            frame,
            verbose=False,
            conf=args.conf,
            imgsz=args.imgsz,
            device="cpu",
            classes=hat.YOLO_CLASSES_TO_KEEP,
        )

    # Preload frames so candidate comparisons do not include disk variance.
    frames = [_load_image(image_path) for image_path in images]

    # Warm up once; the first call includes lazy allocation/compilation.
    for _ in range(max(0, args.warmup)):
        infer(frames[0])

    latencies = []
    class_counts = {}
    detection_counts = []
    for _ in range(max(1, args.loops)):
        for frame in frames:
            started = time.perf_counter()
            results = infer(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed_ms)
            detections = hat.extract_detections(results, model.names)
            detection_counts.append(len(detections))
            for class_name, _conf, *_rest in detections:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    quality = None
    if args.val_data:
        try:
            val_result = model.val(
                data=args.val_data,
                imgsz=args.imgsz,
                conf=args.conf,
                split=args.val_split,
                device="cpu",
                verbose=False,
            )
            quality = {
                key: (float(value) if isinstance(value, (int, float)) else value)
                for key, value in (getattr(val_result, "results_dict", {}) or {}).items()
            }
        except Exception as error:
            quality = {"error": f"{type(error).__name__}: {error}"}

    return {
        "model": str(model_ref),
        "model_size_mb": _model_size_mb(model_ref),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "tracking": bool(args.tracking),
        "latency_scope": "model call (preprocess+inference+postprocess); frames preloaded, camera capture excluded",
        "load_ms": round(load_ms, 1),
        "inference": _latency_stats(latencies),
        "detections_per_frame": {
            "mean": round(statistics.mean(detection_counts), 2) if detection_counts else 0,
            "max": max(detection_counts) if detection_counts else 0,
        },
        "class_counts": dict(sorted(class_counts.items())),
        "quality_on_labeled_set": quality,
    }


def _benchmark_ocr(hat, image_paths, loops):
    if not hat._ocr_available():
        return {"available": False, "reason": "tesseract binary not on PATH"}
    samples = []
    for _ in range(max(1, loops)):
        for image_path in image_paths:
            frame = _load_image(image_path)
            started = time.perf_counter()
            try:
                text = hat._run_tesseract(frame)
                error = None
            except Exception as exc:
                text = ""
                error = f"{type(exc).__name__}: {exc}"
            samples.append((time.perf_counter() - started) * 1000.0)
            if error:
                return {"available": True, "error": error, "latency": _latency_stats(samples)}
    return {
        "available": True,
        "latency": _latency_stats(samples),
        "last_text_chars": len(text),
    }


def _benchmark_tts(hat, phrase, loops):
    if not phrase:
        return None
    program, player = hat._get_speech_binaries()
    if program is None or player is None:
        return {
            "available": False,
            "reason": "espeak-ng/espeak or pw-play not found",
        }
    samples = []
    for _ in range(max(1, loops)):
        started = time.perf_counter()
        hat._speak_message_now(phrase)
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "available": True,
        "phrase": phrase,
        "wall_time": _latency_stats(samples),
        "note": "Wall time covers synthesis plus full playback; first-audio latency is not separated.",
    }


def _summarize_app_latency_logs(paths):
    """Parse opt-in [latency] lines from the app's Pi debug logs."""
    import re

    samples = {}
    pattern = re.compile(r"\[latency\]\s+([a-z_]+)_to_tts_start=([0-9.]+)ms")
    for path in paths:
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            samples.setdefault("_errors", []).append(f"{path}: {error}")
            continue
        for line in lines:
            match = pattern.search(line)
            if match:
                samples.setdefault(match.group(1), []).append(float(match.group(2)))
    summary = {
        key: _latency_stats(values)
        for key, values in samples.items()
        if key != "_errors"
    }
    if "_errors" in samples:
        summary["errors"] = samples["_errors"]
    return summary


def _combined_load(hat, model_ref, images, args):
    from ultralytics import YOLO

    model = YOLO(model_ref)
    try:
        model.fuse(verbose=False)
    except TypeError:
        pass
    frames = [_load_image(path) for path in images]
    stop = threading.Event()
    counters = {"frames": 0, "tts": 0, "voice_started": False}
    infer_latencies = []

    def detection_loop():
        index = 0
        while not stop.is_set():
            frame = frames[index % len(frames)]
            index += 1
            started = time.perf_counter()
            model.track(
                frame,
                persist=True,
                tracker=hat.TRACKER_NAME,
                verbose=False,
                conf=args.conf,
                imgsz=args.imgsz,
                device="cpu",
                classes=hat.YOLO_CLASSES_TO_KEEP,
            )
            infer_latencies.append((time.perf_counter() - started) * 1000.0)
            counters["frames"] += 1

    def tts_loop():
        while not stop.is_set():
            if hat._get_speech_binaries()[0] is not None:
                hat._speak_message_now("Benchmark alert. Person ahead.")
                counters["tts"] += 1
            stop.wait(max(0.1, args.tts_interval))

    before = _thermal_snapshot()
    threads = [
        threading.Thread(target=detection_loop, name="bench-detection", daemon=True),
        threading.Thread(target=tts_loop, name="bench-tts", daemon=True),
    ]
    for thread in threads:
        thread.start()

    if args.voice:
        hat.start_voice_control()
        counters["voice_started"] = True

    started = time.perf_counter()
    try:
        while time.perf_counter() - started < args.combined_seconds:
            if not any(thread.is_alive() for thread in threads):
                break
            time.sleep(0.5)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5.0)
        if counters["voice_started"]:
            hat.stop_voice_control()

    elapsed = max(1e-9, time.perf_counter() - started)
    after = _thermal_snapshot()
    return {
        "seconds": round(elapsed, 1),
        "detection_frames": counters["frames"],
        "detection_fps": round(counters["frames"] / elapsed, 2),
        "detection_latency": _latency_stats(infer_latencies),
        "tts_utterances": counters["tts"],
        "voice_control_started": counters["voice_started"],
        "rss_end_mb": _current_rss_mb(),
        "peak_ram_mb": _peak_ram_mb(),
        "thermal_before": before,
        "thermal_after": after,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="yolo11n.pt", help="Comma-separated model names/paths")
    parser.add_argument("--images", required=True, help="Directory of production-camera frames")
    parser.add_argument("--loops", type=int, default=10, help="Passes over the image set")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference size (same for all candidates)")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--tracking", action="store_true", default=True)
    parser.add_argument("--no-tracking", dest="tracking", action="store_false")
    parser.add_argument("--val-data", default=None, help="Optional labeled YOLO data YAML")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--ocr", action="store_true", help="Benchmark Tesseract on the same frames")
    parser.add_argument("--tts-phrase", default=None, help="Benchmark one full TTS playback")
    parser.add_argument("--tts-loops", type=int, default=3)
    parser.add_argument("--combined-seconds", type=float, default=0.0)
    parser.add_argument("--tts-interval", type=float, default=3.0)
    parser.add_argument("--voice", action="store_true", help="Start offline voice listening during combined load")
    parser.add_argument("--app-log", action="append", default=[], help="App log with [latency] lines (repeatable)")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    images = _image_files(args.images)
    hat = _load_hat_module()

    import ultralytics

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "ultralytics": getattr(ultralytics, "__version__", "unknown"),
        "commit": _run_command(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "camera_source_note": "Frames must be captured with the production camera settings; verify 640x480.",
        "images": [str(path) for path in images],
        "arguments": vars(args),
        "models": [],
    }

    for model_ref in [item.strip() for item in args.models.split(",") if item.strip()]:
        print(f"[bench] {model_ref}", flush=True)
        result["models"].append(_benchmark_model(hat, model_ref, images, args))

    if args.ocr:
        result["ocr"] = _benchmark_ocr(hat, images, args.loops)

    if args.tts_phrase:
        result["tts"] = _benchmark_tts(hat, args.tts_phrase, args.tts_loops)

    if args.combined_seconds > 0:
        print("[bench] combined load", flush=True)
        result["combined"] = _combined_load(
            hat, result["models"][0]["model"], images, args
        )

    if args.app_log:
        result["end_to_end_from_app_log"] = _summarize_app_latency_logs(args.app_log)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"[bench] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
