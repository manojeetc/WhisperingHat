"""Association tests for camera detections, ToF depth, and generic warnings.

These are the repo-side checks for the Pi symptom: a chair/box was visible but
the hat still spoke "unknown obstacle". They exercise the same box->ToF cell
mapping and the same generic-warning arbitration used by the main loop.
"""
import numpy as np

from conftest import reset_state


def _grid(default_mm=2000, cells=None, value_mm=600):
    grid = np.full((8, 8), float(default_mm), dtype=np.float32)
    for row_slice, col_slice in cells or []:
        grid[row_slice, col_slice] = float(value_mm)
    return grid


def test_chair_box_claims_close_tof_cells(new, frozen_time):
    reset_state(new, _grid(cells=[(slice(5, 8), slice(2, 6))], value_mm=600))
    snapshot = new.get_tof_snapshot()
    overlap = new.tof_overlap_rectangle(640, 480)
    diagnostics = {}

    match = new.tof_distance_for_box(
        250, 250, 390, 470, 640, 480,
        snapshot=snapshot, overlap=overlap, diagnostics=diagnostics,
    )

    assert match is not None
    assert 0.59 <= match["distance_m"] <= 0.61
    assert diagnostics["reason"] == "matched"
    assert diagnostics["row_range"][0] <= 7
    assert diagnostics["col_range"][0] <= 5


def test_low_object_below_overlap_still_gets_bottom_row(new, frozen_time, monkeypatch):
    monkeypatch.setattr(new, "TOF_BOX_CLAMP_LOW_OBJECTS", True)
    reset_state(new, _grid(cells=[(slice(7, 8), slice(2, 6))], value_mm=500))
    snapshot = new.get_tof_snapshot()
    overlap = new.tof_overlap_rectangle(640, 480)
    diagnostics = {}

    # Entire sampled box is below the calibrated ToF rectangle.
    match = new.tof_distance_for_box(
        250, 395, 390, 478, 640, 480,
        snapshot=snapshot, overlap=overlap, diagnostics=diagnostics,
    )

    assert match is not None
    assert 0.49 <= match["distance_m"] <= 0.51
    assert diagnostics["reason"] == "matched_below_overlap_clamped"


def test_visible_chair_alert_names_the_class_not_generic_obstacle(new, frozen_time):
    reset_state(new, _grid(cells=[(slice(5, 8), slice(2, 6))], value_mm=600))

    risk, reason, debug = new.score_risk(
        class_name="chair",
        cx=320, cy=360, x1=250, y1=250, x2=390, y2=470,
        frame_width=640, frame_height=480,
        track_id=7,
        distance_key="id:7",
        update_distance=True,
        in_corridor=True,
        corridor_position="in path",
    )

    assert debug["distance_source"] == "tof"
    assert debug["tof_match_reason"] == "matched"
    assert risk >= 2
    assert "chair" in reason
    assert "obstacle" not in reason


def test_visible_known_object_suppresses_generic_unknown(new):
    # A chair at ~0.9 m explains a 0.6 m generic ToF reading.
    assert new.unknown_is_separate_from_known(0.6, float("inf"), 0.9) is False
    # A known object at 2.0 m does not explain a distinct 0.2 m obstacle.
    assert new.unknown_is_separate_from_known(0.2, float("inf"), 2.0) is True
    # A close ToF match is sufficient even when visual distance is poor.
    assert new.unknown_is_separate_from_known(0.5, 0.8, 2.0) is False
