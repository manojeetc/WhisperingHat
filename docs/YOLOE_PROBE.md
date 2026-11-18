# Offline resolution/model naming comparison

The default matrix is YOLO11n at 320, YOLO11n at 640, YOLO11m at 640, and the
existing text-prompt `yoloe-11s-seg.pt` at 640. All receive the same decoded
camera pixels, confidence .40, CPU, unfiltered classes and the existing naming
association rules. Nano at both sizes separates resolution from model gains.
YOLOE keeps its previous configurable five prompts; its vocabulary differs
from COCO. No detector, confidence, hazard, speech or ambiguity policy changes
are deployed by this comparison.

## Capture the missing real frames

No exact alert frames are currently available. The earlier OCR/focus stills
are not substitutes. Prioritize (1) bed edge, (2) the competing laptop/tv scene,
(3) the correctly named chair, and (4) bag on counter outside the corridor.
For the last case, retain an independently triggered central ToF warning so
the off-path bag can be tested as a naming distractor. A scene with no approved
hazard can provide raw detections, but cannot establish a spoken naming result.

Stop the boot service using its actual unit name before a stationary manual
run. From `~/WhisperingHat`:

```bash
WHISPER_HAT_OPEN_VOCAB=0 WHISPER_HAT_NAMING_CAPTURE=1 \
WHISPER_HAT_NAMING_CAPTURE_LIMIT=5 \
WHISPER_HAT_NAMING_CAPTURE_DIR=/tmp/whisper-naming-captures \
WHISPER_HAT_ALERT_TRACE=1 WHISPER_HAT_ASSOCIATION_DEBUG=1 \
WHISPER_HAT_LATENCY_DEBUG=1 .venv/bin/python -u whisper.py
```

Capture is off by default. The limit is 5 pairs per app run, configurable from
1 through 20. A background writer has a two-item queue; full queues drop new
captures. Disk I/O and submission occur after existing speech dispatch. No
extra inference is added. Optional in-memory copies and shape observers record
the inputs to existing inference; debug writing still consumes resources.

Each approved hazard saves a lossless PNG and matching JSON in a unique
`whisper-naming-*` directory outside Git. Generic ToF alerts save the separate
naming frame; legacy alerts such as chair save the detector frame that supplied
the label. Cached detection age is explicit in `detection_frame_number` versus
the decision's `frame_number`. The JSON contains an image hash, ToF grid and age,
association/corridor/distance configuration, raw naming candidates and legacy
candidate diagnostics, selected/spoken text, runtime model/imgsz/resolved size,
and observed BCHW tensor shape. Unavailable tensor shapes are null, not guesses.
Errors are logged without affecting speech. No images are uploaded.

The checkout defaults are raw 640x480 and requested `imgsz=320`; Pi environment
overrides are unverified until a capture is retrieved. The comparison includes
the checkout's configured size and every captured naming size as additional
nano baselines if different from 320/640. `--live-imgsz` adds an explicit size.

Retrieve PNGs and JSONs together from Windows (replace the SSH host):

```powershell
scp -r PI_USER@PI_HOST:/tmp/whisper-naming-captures "$env:USERPROFILE\Downloads"
```

## Compare on Windows

```powershell
Set-Location C:\Users\manoj\WhisperingHat
uv run --no-project --python 3.12 `
  --with 'opencv-python<5' --with 'ultralytics==8.4.215' `
  --with 'git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40' `
  python .\yoloe_probe.py "$env:USERPROFILE\Downloads\whisper-naming-captures" `
  --cache "$env:TEMP\whisper-yoloe-cache" `
  --output "$env:USERPROFILE\Downloads\whisper-naming-comparison" `
  --warmup 2 --repeats 5
```

A single original image is also accepted. Tesseract inputs are excluded.
Capture sidecars load automatically; mismatched hashes/configuration/frame sizes
are rejected. Use the same checkout/calibration/environment as capture.
Images without approved ToF context show raw detections but naming is
**not evaluated**, rather than counted as abstentions.

Public model/encoder packages download during first setup; inference is local
after caching. The new medium weights, existing YOLOE/encoder, photos, settings
and outputs remain outside Git. The tracked nano weights are read in place.

Each run saves a grid of raw detections with current/guarded names and a JSON
report containing boxes, confidence, actual tensor shapes, warmed wall-time
samples and medians, naming eligibility/reasons, counts and artifact paths.
The shared live naming helper is replayed with original hazard approval held
fixed; no sensor/tracker/risk/speech-queue decision is invented or replayed.
Legacy chair captures are a conditional naming-pass comparison, not a claim
that the live app runs that separate pass on chair alerts.

The experimental offline guard keeps the generic warning when the current
winner has a different eligible label with IoU >= .75 and absolute confidence
gap <= .10. These are explicit starting thresholds, not tuned results.
`--ambiguity-iou` and `--ambiguity-score-gap` configure them. Other classes,
disjoint boxes and ineligible/off-path candidates do not force abstention.

## Grade names independently of confidence

Manually inspect each frame and supply acceptable exact hazard names using
`--context labels.json`. It overlays sidecars without replacing sensor data:

```json
{
  "frame-000003.png": {"expected_names": ["bed"]},
  "frame-000042.png": {"expected_names": ["chair"]},
  "frame-000080.png": {"expected_names": []}
}
```

An empty list means generic wording is desired; null/missing means unlabeled.
Use absolute image paths when basenames repeat across sessions. Counts separate
correct names, wrong names, abstentions, unscored names, and not-evaluated frames.
The report also counts wrong names prevented and correct names lost by the
guard. Abstention means no label selected, not automatically a correct decision.
Raw boxes are supplied for visual inspection; no detection mAP is claimed.

## What remains to measure

There is no real-frame winner yet. Synthetic fixtures validate plumbing and
the guard's tradeoff, not recognition accuracy. Windows prediction timings
exclude downloads, model/text initialization, drawing and file I/O; they include
preprocessing and NMS/masks. They are not Pi timings or time to speech.

Only after a model wins on the labeled real frames should it be benchmarked
on the Pi as an isolated naming pass. Keep the live hazard model unchanged.
Measure cold and warmed naming, matched hazard-to-TTS-start latency (existing
`WHISPER_HAT_LATENCY_DEBUG`), and warning-loop frame-gap median/p95/max under
the same scene/load with and without the candidate. Captures record naming
duration, dispatch-after-frame time and preceding loop gap; these do not measure
the next loop gap or acoustic onset. Check actual audible speech separately.
Do not promote a model from offline speed or confidence alone.

Tests require no model downloads:

```powershell
uv run --no-project --python 3.12 --with numpy --with 'opencv-python<5' --with pytest python -m pytest -q --tb=short
```
