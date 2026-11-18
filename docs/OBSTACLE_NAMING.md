# ToF obstacle names from the loaded detector

For capped exact-frame capture and the four-way offline comparison, see
`YOLOE_PROBE.md`. The ambiguity experiment is offline only.

An approved generic ToF warning can use any label from the loaded detector
model. There is no class whitelist or known-name set in the naming pass, and no
special case for bed. Labels come from `model.names` via `extract_detections()`.

The old filters also control tracking, visual risk, scene descriptions and
known-object hazard arbitration, so those uses remain unchanged. Before the
first tracking call, the loaded/fused model is copied into an independent
prediction-only model. Only after `should_alert()` approves a generic ToF
warning does this copy predict with `classes=None`, the existing confidence,
image size and device. It uses original camera pixels before display overlays.
The main detector now defaults to prompted `yoloe-11s-seg.pt`; its first load
may download weights and the text encoder. `WHISPER_HAT_OPEN_VOCAB=0` restores
the YOLO11n main detector. No separate naming worker was added.

Candidates must pass the existing corridor geometry and distance-association
rule (`unknown_is_separate_from_known()` with the existing .45 m tolerance).
Distance uses the existing `tof_distance_for_box()` mapping, with the existing
visual-distance fallback when no cells match. The nearest accepted candidate
supplies the label; confidence breaks distance ties. No match or any naming
initialization/prediction error preserves the exact generic warning.

Only the spoken wording changes. The hazard's original risk, reason, key,
distance, stabilization, deduplication, urgency and repeat/queue metadata stay
unchanged. Names cannot create a warning, suppress one or cause a repeat.
CSV alert rows retain the hazard decision; the opt-in trace shows actual speech.

## Pi verification

Stop the boot service using its actual unit name before a manual run. From
`~/WhisperingHat`:

```bash
WHISPER_HAT_OPEN_VOCAB=0 WHISPER_HAT_ALERT_TRACE=1 \
WHISPER_HAT_ASSOCIATION_DEBUG=1 .venv/bin/python -u whisper.py
```

Both traces are off by default. `[alert trace]` records the original hazard
decision, approval/suppression and actual spoken message. `[obstacle naming]`
records each predicted class, confidence, box, corridor decision, distance
source, ToF match reason and acceptance. Existing association/unknown diagnostics
are retained.

Test a stationary bed-edge view that previously produced an unknown warning,
an unrelated object outside the path, and a real ToF obstacle with no matching
camera box. Also verify no warning from a detection without a ToF hazard.
The independent naming copy adds memory and one prediction per approved generic
warning, not per frame. Its initial cold prediction and added warning latency
must be measured on the Pi before drawing performance conclusions.

## Checkout verification

241 tests passed, including 14 new tests for names absent from the old sets,
off-path/low-confidence/wrong-depth rejection, unchanged visual association,
fallback on no match/errors, no hazard creation, unchanged real alert
deduplication/key/risk, original pixels with overlays, and opt-in trace output.
No hardware or model download is required by these tests:

```powershell
uv run --no-project --python 3.12 --with numpy --with 'opencv-python<5' --with pytest python -m pytest -q --tb=short
```

A Windows smoke check also loaded the existing tracked weights, copied all 80
model-provided labels, and verified independent model/callback/predictor state.
Its sensor grid was a fixture; real bed recognition and association remain Pi
tests. No OCR, autofocus, microphone or startup-service code changed.
