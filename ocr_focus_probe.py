"""Stationary Pi-only capture investigation; does not run obstacle detection.

Stop the boot service, center one whole page under even lighting, then run:
    .venv/bin/python -u ocr_focus_probe.py
"""
import json
from pathlib import Path
import tempfile
import time


FOCUS_TIMEOUT_S = 6.0


def log_focus(stage, metadata, **details):
    fields = {key: metadata.get(key) for key in (
        "AfMode", "AfState", "LensPosition", "ExposureTime", "AnalogueGain",
        "SensorTimestamp",
    )}
    print("[focus probe] " + json.dumps(
        dict(stage=stage, **fields, **details), default=str
    ), flush=True)


def focus_request(camera, stage, controls, require_scan=False):
    """Wait for AF on arriving frames; return an owned request and its outcome."""
    deadline = time.monotonic() + FOCUS_TIMEOUT_S
    scanned = not require_scan
    previous = None
    while True:
        request = camera.capture_request()
        try:
            metadata = request.get_metadata()
            state = metadata.get("AfState")
            if (state, metadata.get("LensPosition")) != previous:
                log_focus(stage, metadata)
                previous = (state, metadata.get("LensPosition"))
            scanned |= state == controls.AfStateEnum.Scanning
            outcome = None
            if state is None:
                outcome = "metadata-unavailable"
            elif scanned and state == controls.AfStateEnum.Focused:
                outcome = "focused"
            elif scanned and state == controls.AfStateEnum.Failed:
                outcome = "failed"
            elif time.monotonic() >= deadline:
                outcome = "timeout"
            if outcome:
                return request, outcome
        except BaseException:
            request.release()
            raise
        request.release()


def capture_attempt(source, config, settle, controls):
    """Use the immediate switch path, or refocus after observing the switch."""
    camera = source.picam2
    with source.io_lock:
        if not settle:
            # Same delay=0 switch/capture/switch-back as capture_still(), with
            # metadata returned from the very request that supplied the pixels.
            arrays, metadata = camera.switch_mode_and_capture_arrays(config, ["main"])
            log_focus("immediate-still", metadata)
            return arrays[0]

        preview_config = camera.camera_configuration()
        try:
            camera.switch_mode(config)
            request = camera.capture_request()
            try:
                log_focus("settled-first-still-before-trigger", request.get_metadata())
            finally:
                request.release()
            # Explicit AF cycle, not an arbitrary sleep or assumed focus carryover.
            camera.set_controls({
                "AfMode": controls.AfModeEnum.Auto,
                "AfTrigger": controls.AfTriggerEnum.Start,
            })
            started = time.monotonic()
            request, outcome = focus_request(camera, "still-af", controls, require_scan=True)
            try:
                log_focus("settled-captured", request.get_metadata(),
                          outcome=outcome, wait_s=round(time.monotonic() - started, 3))
                return request.make_array("main")
            finally:
                request.release()
        finally:
            camera.switch_mode(preview_config)
            camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})


def wait_for_preview(source, label):
    """Measure through the first fresh preview after capture, not just API time."""
    deadline = time.monotonic() + FOCUS_TIMEOUT_S
    with source.condition:
        while source.still_capture_started_monotonic is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("preview did not resume after still capture")
            source.condition.wait(timeout=min(0.1, remaining))
        gap = source.last_still_gap_s
    print(f"[focus probe] attempt={label} preview_frame_gap_s={gap}", flush=True)
    return gap


def save_and_read(app, directory, label, frame, mirrored):
    """Keep each original and the exact existing OCR stdin bytes, even on rejection."""
    prefix = f"{directory.name}-{label}"
    original = directory / f"{prefix}-original.png"
    prepared_path = directory / f"{prefix}-tesseract-input.png"
    try:
        if not app.cv2.imwrite(str(original), frame):
            raise OSError(f"could not write {original}")
        print(f"[focus probe] attempt={label} source=still "
              f"original={frame.shape[1]}x{frame.shape[0]} path={original}", flush=True)
    except Exception as error:
        print(f"[focus probe] original write failed: {error}", flush=True)
    prepared = app._prepare_ocr_frame(frame, unmirror=mirrored)
    try:
        text, words, metrics = app._run_tesseract_tsv(prepared, dump_path=prepared_path)
    except Exception as error:
        print(f"[focus probe] attempt={label} OCR failed: {error}", flush=True)
        return
    print("[focus probe] " + json.dumps(dict(
        attempt=label, text=text, words=words, quality_ok=app._ocr_quality_ok(metrics),
        metrics=metrics, original_metrics=app._ocr_frame_metrics(frame),
    )), flush=True)


def main():
    import whisper as app
    from libcamera import controls

    if app.CAMERA3_AF_MODE != "continuous":
        raise RuntimeError("Probe requires the current continuous-AF baseline")
    root = Path(tempfile.gettempdir()).resolve()
    if any((parent / ".git").exists() for parent in (root, *root.parents)):
        raise ValueError("Probe output must be outside Git working trees")
    directory = Path(tempfile.mkdtemp(prefix="whisper-hat-focus-", dir=root))
    # ponytail: capture-only probe; measure under app load before making AF waits default.
    print(f"[focus probe] output={directory}; AfState: 0=Idle 1=Scanning 2=Focused 3=Failed", flush=True)
    print("[focus probe] No obstacle detection is running. Keep hat and page stationary.", flush=True)
    camera_source = app.Picamera2Source()
    source = app.LatestFrameSource(camera_source)
    try:
        config = camera_source.picam2.create_still_configuration(
            main={"size": (app.OCR_STILL_WIDTH, app.OCR_STILL_HEIGHT), "format": "RGB888"},
            transform=camera_source.transform, buffer_count=1,
        )
        log_focus("configuration", {}, initial_af=app.CAMERA3_AF_MODE,
                  preview_controls=camera_source.picam2.camera_configuration().get("controls", {}),
                  still_controls=config.get("controls", {}))
        for label, settle in (("immediate", False), ("settled", True)):
            # Observe preview before changing controls, including after the last switch.
            with camera_source.io_lock:
                log_focus(f"{label}-preview-before", camera_source.picam2.capture_metadata())
                if settle:
                    camera_source.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                    request, outcome = focus_request(camera_source.picam2, "preview-af", controls)
                    try:
                        log_focus("preview-before-settled-switch", request.get_metadata(), outcome=outcome)
                    finally:
                        request.release()
            camera_source.capture_still = lambda: capture_attempt(camera_source, config, settle, controls)
            started = time.monotonic()
            frame = source.capture_still()
            print(f"[focus probe] attempt={label} capture_call_s={time.monotonic() - started:.3f}", flush=True)
            wait_for_preview(source, label)
            with camera_source.io_lock:
                log_focus(f"{label}-preview-after", camera_source.picam2.capture_metadata())
            save_and_read(app, directory, label, frame, source.ocr_mirrored)
    finally:
        source.release()


if __name__ == "__main__":
    main()
