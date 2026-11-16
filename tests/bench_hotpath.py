"""Indicative microbenchmark of the per-frame Python overhead that was reduced.

This isolates the CPU-side work around YOLO (parsing, corridor geometry, ToF
snapshots, risk scoring, cleanup) -- NOT YOLO inference itself, which is
unchanged. Numbers are from this x86 host and are only indicative; the relative
reduction in tensor conversions, lock acquisitions and snapshot calls is what
transfers to the Pi. Run: python tests/bench_hotpath.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: E402
import numpy as np  # noqa: E402

conftest._stub_ultralytics()
mod = conftest._load_module("whisper_bench", conftest.REPO / "whisper.py")

# Isolate logic: disable printing/logging/window so we time only the hot path.
mod.SHOW_WINDOW = False
mod.LOG_CSV = False
mod.LOG_ALL_DETECTIONS = False
mod.TOF_PRINT_SUMMARY = False
mod.PRINT_ALERT_DEBUG = False
mod.PRINT_EMOTION_TO_TERMINAL = False

W, H = 640, 480
NAMES = conftest.NAMES
rng = np.random.default_rng(0)
xyxy = np.asarray([[100, 80, 220, 400], [300, 120, 380, 360],
                   [420, 200, 560, 460], [50, 250, 160, 470]], dtype=np.float32)
confs = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
clss = np.asarray([0, 0, 2, 56], dtype=np.float32)
ids = np.asarray([1, 2, 3, 4], dtype=np.float32)
results = conftest.make_results(xyxy, confs, clss, ids)

grid = rng.uniform(300, 3500, (8, 8)).astype(np.float32)
grid[rng.random((8, 8)) < 0.2] = np.nan
grid.setflags(write=False)

# Count get_tof_snapshot calls.
_snap_calls = {"n": 0}
_real_snap = mod.get_tof_snapshot


def _counting_snap(*a, **k):
    _snap_calls["n"] += 1
    return _real_snap(*a, **k)


mod.get_tof_snapshot = _counting_snap


def seed():
    conftest.reset_state(mod, grid)
    for tid in (1, 2, 3, 4):
        mod.track_history[tid].append({"time": 999.8, "cx": 300.0, "cy": 240.0, "area": 9000.0, "class_name": "person"})
        mod.track_history[tid].append({"time": 999.9, "cx": 310.0, "cy": 245.0, "area": 10000.0, "class_name": "person"})


def original_cycle():
    """Old main loop: re-parse boxes every frame, recompute corridor + snapshot per box, cleanup every frame."""
    cache = results
    for detection_ran in (True, False, False):
        for box in cache[0].boxes:
            cls_id = int(box.cls[0])
            class_name = NAMES[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = mod.center_of_box(x1, y1, x2, y2)
            area = mod.box_area(x1, y1, x2, y2)
            track_id = int(box.id[0]) if box.id is not None else -1
            corridor_position = mod.corridor_position_word(cx, y2, W, H)
            alert_key = mod.make_alert_key(track_id, class_name, corridor_position)
            distance_key = mod.make_distance_key(track_id, class_name, corridor_position)
            if detection_ran and track_id != -1:
                mod.track_history[track_id].append({"time": time.time(), "cx": cx, "cy": cy, "area": area, "class_name": class_name})
            risk, reason, debug = mod.score_risk(
                class_name, cx, cy, x1, y1, x2, y2, W, H, track_id, distance_key, detection_ran
            )
            mod.update_object_memory(alert_key, risk, reason)
        mod.cleanup_stale_memory()
        mod.cleanup_stale_face_memory()
        mod.tof_unknown_path_distance()


def new_cycle():
    """New main loop: parse once, reuse tuples, one corridor pass + one snapshot per frame, throttled cleanup."""
    cache = []
    last_cleanup = 0.0
    for detection_ran in (True, False, False):
        if detection_ran:
            detections = mod.extract_detections(results, NAMES)
            cache = detections
        else:
            detections = cache
        snap = mod.get_tof_snapshot()
        overlap = mod.tof_overlap_rectangle(W, H)
        for class_name, conf, x1, y1, x2, y2, track_id in detections:
            cx, cy = mod.center_of_box(x1, y1, x2, y2)
            area = mod.box_area(x1, y1, x2, y2)
            in_cor, corridor_position = mod.corridor_geometry_for_point(cx, y2, W, H)
            alert_key = mod.make_alert_key(track_id, class_name, corridor_position)
            distance_key = mod.make_distance_key(track_id, class_name, corridor_position)
            if detection_ran and track_id != -1:
                mod.track_history[track_id].append({"time": time.time(), "cx": cx, "cy": cy, "area": area, "class_name": class_name})
            risk, reason, debug = mod.score_risk(
                class_name, cx, cy, x1, y1, x2, y2, W, H, track_id, distance_key, detection_ran,
                in_corridor=in_cor, corridor_position=corridor_position, tof_snapshot=snap, tof_overlap=overlap,
            )
            mod.update_object_memory(alert_key, risk, reason)
        now = time.time()
        if now - last_cleanup >= mod.CLEANUP_INTERVAL_SECONDS:
            last_cleanup = now
            mod.cleanup_stale_memory(now)
            mod.cleanup_stale_face_memory(now)
        mod.tof_unknown_path_distance(snap)


def bench(fn, cycles=2000):
    seed()
    fn()  # warm
    seed()
    t0 = time.perf_counter()
    for _ in range(cycles):
        fn()
    dt = time.perf_counter() - t0
    return dt / cycles * 1000.0  # ms per 3-frame cycle


# Count snapshot calls for one cycle of each.
seed(); _snap_calls["n"] = 0; original_cycle(); orig_snaps = _snap_calls["n"]
seed(); _snap_calls["n"] = 0; new_cycle(); new_snaps = _snap_calls["n"]

orig_ms = bench(original_cycle)
new_ms = bench(new_cycle)

print(f"Boxes per frame: 4 | cycle = 1 detection + 2 reuse frames")
print(f"get_tof_snapshot calls / 3-frame cycle : original={orig_snaps}  new={new_snaps}")
print(f"Python hot-path time / cycle   : original={orig_ms:.3f} ms  new={new_ms:.3f} ms")
print(f"Speedup (overhead only)        : {orig_ms / new_ms:.2f}x  (-{100 * (1 - new_ms / orig_ms):.0f}%)")
print("NOTE: excludes YOLO/HOG/TFLite inference (unchanged). Indicative x86 numbers; benchmark on Pi.")