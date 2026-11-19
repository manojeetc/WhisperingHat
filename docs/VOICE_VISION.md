# Request-only offline voice vision

`What do you see?` hands a clean, newly captured frame to Florence-2-base and
speaks its first short sentence. `Read this` uses PP-OCRv5 mobile detection and
English recognition through RapidOCR/ONNX Runtime. Detected quadrilaterals are
perspective-rectified by RapidOCR, then read top-to-bottom and left-to-right.
This is single-column reading order, not a multi-column document layout model.
Words are not paraphrased or corrected. A reading is capped at 600 characters
with an explicit spoken notice when more text remains.

These models load into separate low-priority processes. Inference occurs only
on request, uses one CPU thread per model, and never runs in the microphone or
hazard loop. Caption loading/failure/deadline uses the existing detector-based
description. PP loading/failure/unreliable regions/deadline uses Tesseract with
its language gate plus clipped-page and long-page confidence checks. If both
fail, speech is: `I couldn't read that. Hold the whole page steady and try again.`
PP requires all regions at
least 0.80 confidence and rejects edge-touching text; this cannot guarantee
that a confidently recognized word is correct or that all text was detected.
Detected edge clipping gets a specific request to center the whole page, even
if the fallback recognizes a confident interior fragment.

Picamera2 uses a simultaneous 1920x1440 main stream and 640x480 YUV preview
(the existing dimension settings still apply). The camera owner copies the
next main frame for OCR without a mode switch or additional autofocus cycle.
Preview dimensions, transforms and detection rules are retained. The OCR copy
alone receives the existing unmirror correction. This new camera configuration
needs Pi validation for frame cadence, field of view, color and resource use.
The concurrent path requires the existing latest-frame capture worker; when
that worker is disabled, the prior camera path remains available.

Accepted requests say `Reading.` or `Looking.` immediately through the speech
queue. Each feature accepts one outstanding request; another request gets a
spoken busy response. OCR enqueueing has a 15-second deadline, leaving capacity
for hazards. Detection-off preserves command responses and captioning still runs.
Existing STOP interruption, ToF, YOLOE and boot settings are retained. Separate processes avoid synchronous
inference waits in those loops; CPU/memory contention still needs measurement
on the Pi. No claim of unchanged real-time performance is made from Windows.

## Pi setup and manual launch

These commands assume this build has been pulled into `~/WhisperingHat`.
Before pulling over local edits, back up `whisper.py` and stash its changes;
the committed microphone fix replaces the same local patch, so do not blindly
pop that stash. Restore the Pi's local `SHOW_WINDOW = False` setting after the
pull and keep the stash as a backup. Keep the existing
Pi environment with Picamera2, Vosk, YOLOE and speech dependencies. Python 3.11+
on 64-bit Raspberry Pi OS is required. Torch and ONNX Runtime ARM64 wheels are
available; actual imports, memory use and inference on the Pi remain unverified.

First stop the existing boot service. The repository does not identify its
installed unit name; discover and stop the running WhisperingHat unit:

```bash
systemctl list-units --type=service --state=running --no-legend | grep -i whisper
sudo systemctl stop ACTUAL_UNIT.service
```

One-time dependencies and model download (Internet needed only for setup):

```bash
cd ~/WhisperingHat
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements-voice-vision.txt
sudo apt-get install -y tesseract-ocr
.venv/bin/python voice_vision.py --setup
```

Model weights live outside Git at `~/.cache/whisperinghat/voice-vision`.
Florence source/weights use a pinned Hugging Face revision; PP assets have
SHA-256 checks. Runtime uses local files and offline Hugging Face flags.
`WHISPER_HAT_VISION_MODELS` can select another **local** model directory.

Start headless, stationary for initial camera/reading validation:

```bash
cd ~/WhisperingHat
WHISPER_HAT_HEADLESS=1 WHISPER_HAT_VISION_COMMANDS=1 WHISPER_HAT_LATENCY_DEBUG=1 .venv/bin/python -u whisper.py 2>&1 | tee ~/whisper-voice-vision-review.log
```

Startup confirms each model separately:

```text
[voice vision] caption ready: Florence-2-base (local, request only)
[voice vision] ocr ready: PP-OCRv5 mobile (local, request only)
```

After loading, say each command while holding a single page steady under even
light. Check `[caption]` and `[ocr] PP-OCRv5` output, then the existing
`[latency] command_to_tts_start=...ms` measurements (enabled by
`WHISPER_HAT_LATENCY_DEBUG=1`). Loading or a deadline never counts as a model
success. The default caption deadline is 40 seconds; OCR is 15 seconds before
Tesseract fallback. Override with `WHISPER_HAT_CAPTION_TIMEOUT` and
`WHISPER_HAT_PPOCR_TIMEOUT` (seconds). A timed-out process is terminated and
reaped. The next request starts a replacement; requests during loading get the
fallback, and subsequent ready requests can run normally. Input pipe writes
are included in the deadline. Shutdown prevents restarting a model.
The two-second warm command-to-speech target is **not demonstrated**.

Switch back: stop with Ctrl+C, then run the existing description and Tesseract
paths with their original preview-only camera configuration:

```bash
cd ~/WhisperingHat
WHISPER_HAT_HEADLESS=1 WHISPER_HAT_VISION_COMMANDS=0 WHISPER_HAT_OCR_STILL=0 .venv/bin/python -u whisper.py
```

This switch does not change YOLOE selection or the microphone patch. No boot
service configuration is changed by this upgrade.

## Windows verification

Focused model-free checks (also included in the complete suite):

```powershell
cd C:\Users\manoj\WhisperingHat
uv run --no-project --python 3.12 --with pytest --with numpy --with 'opencv-python<5' python -m pytest -q
```

For real inference, install `requirements-voice-vision.txt` into a separate
environment and run:

```powershell
python voice_vision.py --setup --models "$env:LOCALAPPDATA\WhisperingHat\voice-vision"
python voice_vision.py --kind caption --image 'C:\path\scene.png' --models "$env:LOCALAPPDATA\WhisperingHat\voice-vision" --output "$env:TEMP\hat-caption.json"
python voice_vision.py --kind ocr --image 'C:\path\original.png' --unmirror --models "$env:LOCALAPPDATA\WhisperingHat\voice-vision" --output "$env:TEMP\hat-ocr.json"
```

Use `--unmirror` only for a mirrored original, never an already corrected OCR
input. Smoke JSON contains actual text, quadrilaterals, confidence, acceptance
and cold/warm inference-only timings. Private images and reports stay outside
the checkout. Windows inference timing is neither Pi timing nor speech latency.
