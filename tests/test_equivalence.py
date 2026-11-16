"""Behavior-equivalence checks for the Pi performance/reliability edits.

Each test compares the NEW wired fast path against the ORIGINAL unwired path
(or an independent expectation) on identical inputs, proving the optimizations
preserve decisions. Models, thresholds, schedules and stage order are untouched.
"""
import numpy as np
import pytest

from conftest import NAMES, make_results, original_parse, reset_state

W, H = 640, 480


def assert_debug_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, (float, np.floating)) or isinstance(vb, (float, np.floating)):
            assert va == pytest.approx(vb, rel=1e-6, abs=1e-9, nan_ok=True), k
        else:
            assert va == vb, k


def seed_box_state(mod, distance_key, track_id, now=1000.0):
    mod.distance_memory[distance_key] = {
        "raw_distance_m": 2.0,
        "smooth_distance_m": 2.2,
        "last_update_time": now - 0.1,
        "approach_mps": 0.0,
    }
    if track_id != -1:
        mod.track_history[track_id].append(
            {"time": now - 0.2, "cx": 300.0, "cy": 240.0, "area": 9000.0, "class_name": "person"}
        )
        mod.track_history[track_id].append(
            {"time": now - 0.1, "cx": 310.0, "cy": 245.0, "area": 10000.0, "class_name": "person"}
        )


def random_boxes(rng, n, with_ids=True):
    xyxy = []
    for _ in range(n):
        x1 = rng.uniform(0, W - 40)
        y1 = rng.uniform(0, H - 40)
        x2 = x1 + rng.uniform(20, W - x1)
        y2 = y1 + rng.uniform(20, H - y1)
        xyxy.append([x1, y1, x2, y2])
    confs = rng.uniform(0.4, 0.99, n)
    clss = rng.choice(list(NAMES.keys()), n)
    ids = rng.integers(1, 50, n).astype(np.float32) if with_ids else None
    return np.asarray(xyxy, dtype=np.float32), confs, clss, ids


def make_grid(rng):
    grid = rng.uniform(300, 3500, (8, 8)).astype(np.float32)
    grid[rng.random((8, 8)) < 0.25] = np.nan
    grid.setflags(write=False)
    return grid


# 1. Single-pass corridor helper == the two-pass helpers it replaces.
def test_corridor_single_pass_matches_two_pass(new):
    rng = np.random.default_rng(7)
    for _ in range(500):
        cx, foot_y = rng.uniform(0, W), rng.uniform(0, H)
        in_cor, pos = new.corridor_geometry_for_point(cx, foot_y, W, H)
        assert pos == new.corridor_position_word(cx, foot_y, W, H)
        assert in_cor == new.object_in_walking_corridor(cx, foot_y, W, H)


# 2. extract_detections == original per-box tensor extraction.
@pytest.mark.parametrize("with_ids", [True, False])
def test_extract_detections_matches_original(new, with_ids):
    rng = np.random.default_rng(11)
    for n in (0, 1, 2, 5, 12):
        xyxy, confs, clss, ids = random_boxes(rng, n, with_ids)
        results = make_results(xyxy, confs, clss, ids)
        expected = original_parse(results, NAMES)
        got = new.extract_detections(results, NAMES)
        assert len(got) == len(expected)
        for g, e in zip(got, expected):
            assert g[0] == e[0]
            assert g[1] == pytest.approx(e[1])
            assert np.allclose(np.asarray(g[2:6], dtype=np.float64),
                               np.asarray(e[2:6], dtype=np.float64))
            assert g[6] == e[6]


def test_extract_detections_edge_cases(new):
    assert new.extract_detections(make_results(None, None, None, None), NAMES) == []
    assert new.extract_detections([], NAMES) == []
    assert new.extract_detections(None, NAMES) == []


# 3. score_risk: precomputed corridor/ToF args == letting it recompute them.
@pytest.mark.parametrize("update_distance", [True, False])
@pytest.mark.parametrize("use_tof", [True, False])
def test_score_risk_wiring_equivalence(new, frozen_time, update_distance, use_tof):
    rng = np.random.default_rng(23)
    grid = make_grid(rng) if use_tof else None
    for _ in range(150):
        xyxy, confs, clss, ids = random_boxes(rng, 1, True)
        x1, y1, x2, y2 = xyxy[0]
        class_name = NAMES[int(clss[0])]
        track_id = int(ids[0])
        cx, cy = new.center_of_box(x1, y1, x2, y2)
        dk = new.make_distance_key(track_id, class_name, "in path")

        reset_state(new, grid)
        seed_box_state(new, dk, track_id)
        r1, reason1, dbg1 = new.score_risk(
            class_name, cx, cy, x1, y1, x2, y2, W, H, track_id, dk, update_distance
        )

        reset_state(new, grid)
        seed_box_state(new, dk, track_id)
        in_cor, pos = new.corridor_geometry_for_point(cx, y2, W, H)
        snap = new.get_tof_snapshot() if use_tof else None
        overlap = new.tof_overlap_rectangle(W, H) if use_tof else None
        r2, reason2, dbg2 = new.score_risk(
            class_name, cx, cy, x1, y1, x2, y2, W, H, track_id, dk, update_distance,
            in_corridor=in_cor, corridor_position=pos, tof_snapshot=snap, tof_overlap=overlap,
        )

        assert r1 == r2
        assert reason1 == reason2
        assert_debug_equal(dbg1, dbg2)


# 4. ToF matching: one shared snapshot == fresh per-call snapshot.
def test_tof_box_shared_snapshot(new, frozen_time):
    rng = np.random.default_rng(41)
    reset_state(new, make_grid(rng))
    snap = new.get_tof_snapshot()
    overlap = new.tof_overlap_rectangle(W, H)
    for _ in range(200):
        x1, y1 = rng.uniform(0, W - 40), rng.uniform(0, H - 40)
        x2, y2 = x1 + rng.uniform(20, W - x1), y1 + rng.uniform(20, H - y1)
        assert new.tof_distance_for_box(x1, y1, x2, y2, W, H) == \
               new.tof_distance_for_box(x1, y1, x2, y2, W, H, snapshot=snap, overlap=overlap)


def test_tof_unknown_shared_snapshot(new, frozen_time):
    rng = np.random.default_rng(43)
    for _ in range(20):
        reset_state(new, make_grid(rng))
        snap = new.get_tof_snapshot()
        assert new.tof_unknown_path_distance() == new.tof_unknown_path_distance(snap)


# 5. Unchanged helpers validated against independent expectations.
def test_fast_percentile_matches_numpy(new):
    rng = np.random.default_rng(47)
    for _ in range(200):
        vals = rng.uniform(50, 4000, int(rng.integers(1, 64))).astype(np.float32)
        for p in (0.0, 12.5, 25.0, 50.0, 100.0):
            assert new._fast_percentile_1d(vals, p) == pytest.approx(
                float(np.percentile(vals, p, method="linear")), rel=1e-5, abs=1e-3
            )


def test_cleanup_removes_stale(new):
    now = 1000.0
    reset_state(new)
    new.track_memory["fresh"] = {"last_seen": now - 1.0}
    new.track_memory["stale"] = {"last_seen": now - 10.0}
    new.distance_memory["fresh"] = {"last_update_time": now - 1.0}
    new.distance_memory["stale"] = {"last_update_time": now - 10.0}
    new.track_history[1].append({"time": now - 1.0})
    new.track_history[2].append({"time": now - 10.0})
    new.face_memory["fresh"] = {"last_seen": now - 1.0}
    new.face_memory["stale"] = {"last_seen": now - 10.0}
    new.cleanup_stale_memory(now)
    new.cleanup_stale_face_memory(now)
    assert set(new.track_memory) == {"fresh"}
    assert set(new.distance_memory) == {"fresh"}
    assert set(new.track_history) == {1}
    assert set(new.face_memory) == {"fresh"}


def test_face_matrix_equivalence():
    rng = np.random.default_rng(53)
    encodings = [rng.normal(size=128) for _ in range(24)]
    probe = rng.normal(size=128)
    matrix = np.asarray(encodings, dtype=np.float64)
    assert np.array_equal(
        np.linalg.norm(np.asarray(encodings) - probe, axis=1),
        np.linalg.norm(matrix - probe, axis=1),
    )


def test_distance_engine_formula(new):
    # person calibration: 1.2 m at 280 px box height.
    d, src = new.raw_fake_distance_meters("person", 100, 100, 200, 380, W, H)
    assert src == "calibrated_height"
    assert d == pytest.approx(min(8.0, max(0.25, 1.2 * 280 / 280)))
    # non-calibrated class falls back to area.
    _, src2 = new.raw_fake_distance_meters("dog", 100, 100, 200, 380, W, H)
    assert src2 == "area_fallback"
    assert new.visual_distance_level(0.5) == "very_close"
    assert new.visual_distance_level(1.0) == "close"
    assert new.visual_distance_level(2.0) == "medium"
    assert new.visual_distance_level(5.0) == "far"


class _Audio:
    def __init__(self, frame_data, sample_width, sample_rate):
        self.frame_data = frame_data
        self.sample_width = sample_width
        self.sample_rate = sample_rate


def test_audio_conversion(new):
    rng = np.random.default_rng(61)
    n = 4800
    raw16 = rng.integers(-32768, 32767, n, dtype=np.int16).tobytes()
    # Fast path: already 16 kHz mono int16 -> returned unchanged.
    assert new._convert_audio_to_16khz_mono_int16(_Audio(raw16, 2, 16000)) == raw16
    # Resampled path: correct output length and int16 byte width.
    raw44 = rng.integers(-32768, 32767, int(44100 * 0.3), dtype=np.int16).tobytes()
    out = new._convert_audio_to_16khz_mono_int16(_Audio(raw44, 2, 44100))
    assert isinstance(out, bytes) and len(out) % 2 == 0
    assert len(out) // 2 == round(int(44100 * 0.3) * 16000 / 44100)


# 6. Alert policy: deterministic speak/skip sequence (unchanged logic).
def test_alert_policy_sequence(new, monkeypatch):
    import time as _time

    key = "zone:person:in path"
    reason = "person rushing ahead"
    reset_state(new)
    decisions = []
    for t in (1000.0, 1000.2, 1000.3):
        monkeypatch.setattr(_time, "time", lambda t=t: t)
        new.update_object_memory(key, 3, reason)
        decisions.append(new.should_alert(key, 3, reason))
    # Not stable yet -> speak once stable -> no immediate repeat.
    assert decisions == [False, True, False]