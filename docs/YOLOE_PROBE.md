# Offline obstacle naming comparison

`yoloe_probe.py` compares the same original pixels in three passes:

1. YOLO11n with the live COCO class whitelist.
2. The same YOLO11n without the whitelist, to reveal labels the app discards.
3. Text-prompt YOLOE, defaulting to the officially supported small
   `yoloe-11s-seg.pt` checkpoint and
   `bed,chair,bag of chips,cardboard box,laundry basket`.

This is a standalone experiment. Importing the existing extraction/geometry
helpers does not start the app. It never calls tracking, risk scoring, alert
arbitration, speech, camera, microphone, OCR, or autofocus workers. Naming
results have no connection to hazard warnings. No live files/settings change.

## What currently produces an unknown warning

The inspected baseline is `whisper.py` at `f21faf7`:

- `MODEL_PROFILES["baseline"]` is the tracked `yolo11n.pt`, `imgsz=320`,
  `CONFIDENCE=0.40`, CPU. The probe explicitly selects that profile, regardless
  of live model-profile environment overrides.
- Picamera2 captures 640x480 RGB888 buffers used directly as OpenCV BGR, with
  `CAMERA_HFLIP=False`, `CAMERA_VFLIP=True`. The shared detection image keeps
  that transform. The OCR-only unmirror does not belong in this experiment.
- Ultralytics preserves aspect with stride-aligned LetterBox, converts BGR to
  RGB/BCHW, and scales FP32 pixels by 255. `conf`, `imgsz`, effective `iou`,
  `max_det`, `rect`, and `half` are recorded in each result. In the measured
  version, NMS defaults are `iou=.7`, `max_det=300`, `rect=True`, `half=False`.
- Live `run_yolo()` uses persistent ByteTrack (`bytetrack.yaml`), every third
  processed frame, reusing cached detections between calls. The probe uses
  **predict**, since unrelated saved stills cannot supply tracker history.
  It reproduces detector inputs/filtering, not complete live tracking output.
- `YOLO_CLASSES_TO_KEEP` is
  `[0,1,2,3,5,7,9,11,13,15,16,17,19,24,26,28,56]`.
  **COCO bed is ID 59: the weights know this word, but the live filter removes
  it.** An unfiltered bed box would prove a filtering problem for that frame.
- `extract_detections()` retains the exact model label. `score_risk()` has
  label-specific person/vehicle/animal/static-object branches. Static-object
  naming also depends on distance, lower-frame relevance and corridor/motion
  checks. **`bed` and exact `bag of chips` are absent from
  `KNOWN_OBJECT_CLASSES`/the static set.** `chair`, `cardboard box` and
  `laundry basket` are present. There is no synonym normalization. The generic
  score branch can say "object close"; ToF time-to-collision overrides can
  still insert a raw label. Missing set membership is not a universal ban on
  speaking that label.
- The literal **"unknown obstacle" comes from the ToF fallback**, not a YOLO
  output class. `tof_unknown_path_distance()` needs enabled/calibrated/fresh
  depth and at least two valid cells in central rows 2-7, columns 2-5; it takes
  the median. The main loop smooths it (alpha .35) and applies .12 m risk
  hysteresis. Invalid/unavailable depth does not create this candidate.
- The nearest in-corridor ToF-associated detection can explain that reading
  regardless of known-set membership. A camera-distance fallback explains it
  only for in-corridor exact names in `KNOWN_OBJECT_CLASSES`.
  `unknown_is_separate_from_known()` requires the smoothed unknown distance
  plus .45 m to be nearer than both known-distance alternatives.
- If still unexplained, `unknown_obstacle_risk()` uses .55/1.25/2.5 m
  STOP/caution/notice thresholds. The candidate must win the main loop's
  risk/distance arbitration. `should_alert()`, `alert_text()`, stabilization,
  deduplication and the speech queue decide whether it is actually spoken.

Thus a missed/low-confidence/misclassified detection, a filtered class, a box
outside the walking corridor, failed depth association, or an unusable exact
name can all leave the ToF fallback unexplained. A still alone cannot identify
which path caused a particular historical utterance.

## Windows rerun

From PowerShell (Python 3.12 is available through `uv`):

```powershell
Set-Location C:\Users\manoj\WhisperingHat
uv run --no-project --python 3.12 `
  --with 'opencv-python<5' --with 'ultralytics==8.4.215' `
  --with 'git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40' `
  python .\yoloe_probe.py `
  "$env:USERPROFILE\Downloads\whisper-hat-focus-0bgcherh" `
  "$env:USERPROFILE\Downloads\whisper-hat-ocr-g0m114ny-still-original.png" `
  "$env:USERPROFILE\Downloads\whisper-hat-ocr-k40ie7yb-still-original.png" `
  --checkpoint yoloe-11s-seg.pt `
  --labels 'bed,chair,bag of chips,cardboard box,laundry basket' `
  --cache "$env:TEMP\whisper-yoloe-cache" `
  --output "$env:USERPROFILE\Downloads\whisper-yoloe-results" `
  --warmup 2 --repeats 5
```

Replace the input arguments with a single original image or a folder of saved
detector frames. Folders are searched recursively; `*-tesseract-input.*` files
are skipped and reported. Other processed images cannot be detected by filename:
the caller must supply originals. No flipping, cropping, OCR preprocessing or
manual resizing is applied before either model's own preprocessing.

The first setup downloads public packages/checkpoint/text encoder; subsequent
cached runs perform image inference locally. Images are never sent to a service.
Ultralytics' prompt encoder currently downloads about 572 MB in addition to the
26.5 MB checkpoint; both stay in the private cache. A bare checkpoint filename
can download an official asset; an explicit path must already exist. Use a
text-prompt checkpoint, not a `-pf` variant. The official model list is at
https://docs.ultralytics.com/models/yoloe/.

Each run creates a unique directory beneath `--output`, prints that directory,
and saves `report.json` plus a three-panel comparison per frame. New weights,
settings, encoder and annotations must be outside **any Git checkout**; the CLI
rejects cache/output/checkpoint directories under Git. The pre-existing tracked
baseline weight is read in place. Generated files and private photos are never
added to the repo.

JSON includes original/checkpoint hashes, vocabulary, exact prompts, package
versions, prediction arguments, label/confidence/original-pixel xyxy boxes and
all warmed samples/median milliseconds. Each model gets two untimed warmups
per frame and five timed calls. Times include preprocessing, prediction,
postprocessing (including YOLOE masks) and Results creation, excluding loading,
prompt encoding, annotation and disk I/O. Sequential host timings are subject
to Windows load; they are **not Pi timings or live tracking throughput**.

## Partial app checks

Each candidate reports the actual class filter, exact known-set membership and
walking-corridor geometry, reusing app helpers. Raw YOLO11n candidates are checked
against the live whitelist even though its diagnostic pass disables that
filter. YOLOE's prompt IDs are not COCO IDs: the proposed open-vocabulary path
has no COCO whitelist. Camera-known-distance acceptance means eligibility only;
camera distance and final warning acceptance are not inferred.

Without synchronized sensor data, ToF association, unknown arbitration and
spoken alerts explicitly say **not evaluated**. Optional `--context context.json`
maps absolute image paths (or unique filenames) to objects with:

```json
{
  "frame.png": {
    "frame_size": [640, 480],
    "association_config": {},
    "tof_snapshot": {"grid_mm": [], "fresh": true, "age_seconds": 0.05}
  }
}
```

Replace `association_config` with the exact object from the report, verified
against the capture's configuration, and `grid_mm` with the synchronized 8x8
processed grid (positive millimetres; JSON null for invalid cells). Frame size
must match. Do not reuse preview calibration for a still with a different field
of view. Invalid/mismatched context is rejected. Real `tof_distance_for_box()`
returns accepted/rejected with its reason/cells/distance; this isolated mapping
does not imply a filtered detection would reach the main loop. Full temporal
arbitration and speech remain not evaluated even with a grid.

## Next capture set and decision

Save original **640x480 detection-preview frames**, retaining the shared camera
orientation, from the actual unknown-warning events. Do not substitute OCR
inputs or assume the existing 1920x1440 stills reproduce those events.

- Bed: the actual edge-only view at an unknown warning, several nearby frames,
  and a wider view of the same bed as a positive control. Include approach
  angles/distances relevant to the hat and a bed-free view of the same path.
- Snack bag: bag on a counter, then the identical counter/view with the bag
  removed. The existing carpet/bunk-bed bag shots are smoke tests only.
- Vocabulary checks: chair, cardboard box and laundry basket positives in the
  walking path, plus a view without any of the five target objects. Include
  background clutter so wrong-target boxes are visible.
- Keep timestamps/alert log rows, exact model/settings, detection labels,
  confidences, boxes and track IDs; contemporaneous ToF grid/freshness/age and
  mount calibration; corridor/association diagnostics; smoothed known/unknown
  distances, risk/arbitration and speech decisions. Record sensor distance and
  expected object labels independently. Do not create depth/tracking data from
  a photograph.

Inspect boxes against the foreground object and count misses/wrong labels on
paired positives/negatives. If raw YOLO11n already detects the relevant bed,
fix label filtering/matching first in a separate patch. If YOLOE consistently
recovers useful correct names on that set, benchmark it on the Pi with the same
frames and measure live frame gaps before considering integration. If neither
condition is established, collect better test frames. Never let naming alone
trigger or suppress a hazard warning.

Tests (no weights or model downloads):

```powershell
uv run --no-project --python 3.12 --with numpy --with 'opencv-python<5' --with pytest python -m pytest -q --tb=short
```
