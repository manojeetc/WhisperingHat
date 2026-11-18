# Pi 5 voice, obstacle alerts, and vision notes

This document separates three things that were getting mixed together:

1. **Confirmed from this repository** (code paths, log contents, tests).
2. **Published third-party measurements** (sources cited; not our hardware).
3. **Still unverified on our Pi** (what to run and what to look for).

The working baseline before these changes is commit `40b17d4`
(`Add voice-triggered text reading and scene descriptions`). The default
runtime behavior after these changes is still YOLO11n at `imgsz=320`, the same
`PROCESS_EVERY_N_FRAMES=3`, the same Vosk model, and the same TTS pipeline.
Heavier models and new diagnostics are opt-in.

## 1. Confirmed root causes and code paths

### 1.1 “unknown obstacle” repeats

The message comes from the ToF path, not from YOLO:

- `tof_unknown_path_distance()` in `whisper.py` reads the central 6×4 ToF
  cells (`TOF_UNKNOWN_PATH_ROW_START..END`, `TOF_UNKNOWN_PATH_COL_START..END`).
- It selected the **25th percentile** of valid cells with
  `TOF_MIN_VALID_CELLS = 1`.
- The alert key is always `tof:unknown:in_path`, and
  `should_alert()` spoke again whenever the reason text changed
  (`SPEAK_ON_SAME_RISK_REASON_CHANGE = True`).
- The STOP global gap was `0.5 s`, so a close noisy reading could be spoken
  again almost immediately.

The repository’s own `whisper_hat_tof_log.csv` shows the failure directly:
109 `unknown obstacle` alert rows, with the same key swinging between
`0.05 m` and `2.44 m`, `valid_cells` from 1 to 24, and reason changes such as
`unknown obstacle in path` → `unknown obstacle close in path` →
`unknown obstacle very close in path` within seconds. This is a **distance
estimate instability + reason-change speech policy** problem, not a YOLO label
problem.

What changed:

- The generic path estimate now uses the **median** (`TOF_UNKNOWN_PERCENTILE =
  50`) and requires at least `TOF_UNKNOWN_MIN_VALID_CELLS = 2`.
- The distance is smoothed (`TOF_UNKNOWN_SMOOTH_ALPHA = 0.35`) and banded with
  hysteresis (`TOF_UNKNOWN_HYSTERESIS_M = 0.12`).
- A constant generic key is deduplicated: same-band reason changes do not
  speak. A **materially closer** reading (`TOF_UNKNOWN_CLOSER_BYPASS_M = 0.40`)
  or a new appearance after the path was clear still speaks.
- Risk escalation is still allowed, but rate-limited by
  `TOF_UNKNOWN_ESCALATION_GAP_SECONDS = 1.5`; routine repeats use
  `TOF_UNKNOWN_REPEAT_SECONDS = 30`.
- `[unknown obstacle ...]` debug lines now show `raw`, `smooth`, `valid`,
  `spread`, `nearest_known`, `risk`, `action`, and `reason`. That makes it
  possible to tell a real obstacle from a sensor artifact instead of guessing.

A real obstacle is still announced without a confident label. The change only
stops the same generic hazard from being re-announced every time a noisy
distance crosses a band boundary. `TOF_UNKNOWN_MIN_VALID_CELLS = 2` is a
deliberate tradeoff: a one-cell return is too weak to distinguish a small real
object from the 0.05 m false readings in the log. If the mounted sensor shows
that a real small obstacle can only ever fill one cell, lower it to 1 and rely
on the stabilization/smoothing window, then re-run the four alert tests.

### 1.2 Voice commands not reliably reaching OCR

The recognition grammar already contained `read this`, `read that`, and the
other OCR phrases. The routing layer, however, was exact-match only:

```python
command = _normalize_voice_command(text)
if command in OCR_READ_PHRASES: ...
```

There was no alias handling, no partial/result-stage logging, no record of
whether TTS was active, and no way to tell whether audio was dropped before
Vosk, Vosk returned no text, the command was ignored, the OCR worker was not
running, or the OCR answer was cleared from the speech queue.

A second concrete failure path was found in the speech queue:
`play_alert_sound(risk >= 3)` called `clear_waiting_speech()` and
`interrupt_current_speech()` on **every** STOP, including repeated STOPs.
An OCR answer is queued at risk 2. A repeated STOP could therefore clear the
answer the user just requested, which looks exactly like “read this did not
trigger anything.” Routine repeated alerts no longer clear a direct command
response; a genuinely new/escalated STOP can still interrupt for safety.

What changed:

- `_normalize_voice_command()` removes punctuation/apostrophes.
- `VOICE_DETECTION_ON_ALIASES`, `VOICE_DETECTION_OFF_ALIASES`,
  `OCR_READ_ALIASES`, and `SCENE_DESCRIBE_ALIASES` are included in the grammar
  and mapped to canonical commands. Examples: `read it`, `read this page`,
  `what does this say`, `turn detection off`, `stop detection`,
  `what is ahead`.
- `_match_voice_command()` returns a canonical command or `None`; unknown
  speech is ignored, not fuzzy-matched into a command.
- `VOICE_DEBUG_TIMING` (`WHISPER_HAT_VOICE_DEBUG=1`) logs timestamps for:
  microphone open/actual source rate, ambient calibration, listen wait/timeout,
  captured audio duration, whether TTS overlapped, Vosk partial/final result,
  command match/ignore/debounce, OCR trigger/pickup/frame/Tesseract time, and
  speech enqueue/start/end. No raw audio is saved.
- `VOICE_PLAYBACK_COOLDOWN_SECONDS` (`WHISPER_HAT_VOICE_COOLDOWN`, default
  `0.0`) is an opt-in post-TTS settle window. It is **off by default**; enable
  it only if the test matrix shows TTS overlap causing missed commands.
- `VOICE_MIC_FALLBACK_RATES = (16000, 48000)` is used only if opening the
  configured rate raises. The actual `source.SAMPLE_RATE` and device name are
  logged, so a hardware-rate mismatch is visible.
- `PRINT_LATENCY_DEBUG` (`WHISPER_HAT_LATENCY_DEBUG=1`) logs
  `hazard_to_tts_start` and `command_to_tts_start` in milliseconds.

The **exact** reason a particular “read this” utterance failed on the Pi is not
confirmed by this repository alone. The debug stages above are there to
distinguish the remaining possibilities: mic/sample-rate, Vosk final result,
command routing, OCR worker, or speech-queue dropping.

### 1.3 Listening, detection, OCR, and TTS blocking

The listening loop is its own daemon thread; detection/YOLO is in the main
loop; OCR is another daemon; TTS is a worker plus `espeak-ng`/`pw-play`
subprocesses. They do not share a lock that would stop `recognizer.listen()`.
They do compete for CPU, and TTS playback can physically enter the microphone.
The new `speech_active_event` and `tts_overlap` log field are intended to
separate acoustic overlap from a stopped listening loop.

Do not silently disable commands during frequent alerts. The cooldown is
opt-in and time-limited; if it is enabled, the log shows each wait.

## 2. Published vision benchmarks vs. our device

### 2.1 Same-class accuracy and speed (YOLO11 family)

Official Ultralytics YOLO11 detection table (COCO val, 640 input):

| Model | mAP50-95 | CPU ONNX (ms) | Params (M) | FLOPs (B) |
|---|---:|---:|---:|---:|
| YOLO11n | 39.5 | 56.1 ± 0.8 | 2.6 | 6.5 |
| YOLO11s | 47.0 | 90.0 ± 1.2 | 9.4 | 21.6 |
| YOLO11m | 51.5 | 183.2 ± 2.0 | 20.1 | 68.1 |

Source: <https://docs.ultralytics.com/models/yolo11/> (accessed 2026-11-18).
These are official accuracy/CPU-ONNX figures, not Raspberry Pi numbers.

A peer-reviewed Pi 5 comparison (PyTorch, COCO128, 640, batch 1, all
preprocess/inference/postprocess included) reports:

| Model | Pi 5 FPS, standard | Pi 5 FPS, high power |
|---|---:|---:|
| YOLOv11n | 1.968 | 2.11 |
| YOLOv11s | 1.05 | 1.20 |
| YOLOv11m | 0.39 | 0.438 |

Source: *Benchmarking YOLOv8–YOLOv12 for Real-Time Object Detection on
Single-Board Computers*, Mach. Learn. Knowl. Extr. 2026, 8(7), 204, Table 7,
<https://doi.org/10.3390/make8070204>. This is a published third-party
measurement, not a measurement on our hat. It does not include our tracking,
Vosk, TTS, ToF, camera capture, or 320-pixel input.

Export-format measurements on a Pi 5 from LearnOpenCV (YOLO11n, 640,
COCO8 validation):

| Format | Inference (ms/image) | FPS |
|---|---:|---:|
| PyTorch | 360.09 | 2.78 |
| ONNX | 156.84 | 6.38 |
| OpenVINO | 80.93 | 12.36 |
| MNN | 115.83 | 8.63 |
| NCNN | 292.1 | 3.42 |

Source: <https://learnopencv.com/yolo11-on-raspberry-pi/> (accessed
2026-11-18). Their article reports an 8–10 FPS end-to-end pipeline. The NCNN
number is slower than several other exports in that run, which is why the
same-scene benchmark below matters more than a format reputation.

The current Ultralytics Raspberry Pi guide also has a Pi 5 table, but it now
uses YOLO26, not YOLO11: YOLO26n/s/m are listed at 128.42/352.84/993.78 ms per
image at 640. That is a useful order-of-magnitude sanity check, not a YOLO11
comparison. <https://docs.ultralytics.com/guides/raspberry-pi/>

**Same classes vs. more classes:** YOLO11s/m improve accuracy on the same COCO
classes. They do **not** add vocabulary. If the problem is a chair, sign, or
other object outside the COCO class filter, a larger YOLO11 will not name it.
That is the separate open-vocabulary question below.

### 2.2 Open-vocabulary: YOLOE

Ultralytics YOLOE takes text prompts, visual prompts, or a built-in vocabulary.
Published LVIS text-prompt numbers for the YOLO11-based family:

| Model | mAP50-95 (text prompt) | Params (M) | FLOPs (B) |
|---|---:|---:|---:|
| YOLOE-11s | 27.5 | 10.7 | 22.7 |
| YOLOE-11m | 33.0 | 21.0 | 70.4 |
| YOLOE-26n | 24.7 | 3.9 | 6.1 |

Source: <https://docs.ultralytics.com/models/yoloe/> (accessed 2026-11-18).

Practical constraints from the same official docs:

- The first text-prompt call downloads a CLIP text encoder (YOLOE-11/v8 use
  `mobileclip_blt.ts`) and installs a tokenizer; do this once while online.
- Exports to ONNX/OpenVINO/TFLite/NCNN/RKNN are supported, but prompted
  classes are baked into the exported weights. Changing classes requires
  re-exporting.
- Prompt comparison costs more per forward pass as the prompt list grows
  (the docs report about +19% at 1,203 classes and +89% at 4,585 classes on a
  CPU test). FLOPs tables do not include that cost.
- Zero-shot accuracy is below a closed-set model trained for the same classes,
  and rare classes are the weak spot.

No credible public YOLOE Raspberry Pi 5 benchmark was found. Treat any Pi 5
YOLOE estimate as unverified until our own `tests/bench_pi.py` run. It is a
candidate for an **on-demand** “what’s ahead?” experiment with a small frozen
prompt list, not for the urgent obstacle-alert path.

### 2.3 On-demand scene description

The current `_compose_scene_description()` is cheap: it sorts at most a handful
of existing detections by risk/distance and speaks up to
`SCENE_DESCRIBE_MAX_OBJECTS = 3`. It adds no model, no extra inference, and no
network dependency.

A lightweight offline captioning model (BLIP/ViT-GPT2/Florence-2/Moondream2
class) would add hundreds of MB to multiple GB of weights and is likely to
take seconds per image on a Pi 5 CPU. No Pi 5 measurement for those models was
found in this review. Recommendation: keep the detections-based description as
the default; prototype a captioner only as a separate on-demand experiment and
never use its output as the source of urgent obstacle alerts.

OCR remains a separate path (`read this` → Tesseract → spoken text) and is not
mixed with scene description.

## 3. What to run on the Pi

### 3.1 Baseline and rollback

```bash
cd ~/WhisperingHat
git fetch origin
git switch main
git pull --ff-only
python -m pytest -q
```

To compare with the pre-change baseline without losing the current work:

```bash
git switch -c baseline-40b17d4 40b17d4
# run the same commands, then return:
git switch main
```

The default runtime after these changes is still the baseline model profile.
Heavier models are opt-in:

```bash
# Baseline (default)
python whisper.py

# YOLO11s at 384, opt-in only
WHISPER_HAT_MODEL_PROFILE=accuracy python whisper.py

# YOLO11m at 416, opt-in only (expected to be too slow for real-time)
WHISPER_HAT_MODEL_PROFILE=max_accuracy python whisper.py
```

### 3.2 Voice test matrix

Enable stage logging for the test run only:

```bash
WHISPER_HAT_VOICE_DEBUG=1 WHISPER_HAT_LATENCY_DEBUG=1 \
python whisper.py 2>&1 | tee /tmp/whisper-voice.log
```

Run each phrase five times in each condition:

1. quiet room, speaker muted;
2. quiet room, speaker unmuted;
3. quiet room, speaker muted, while an obstacle alert is playing (use a
   stationary obstacle or the `--tts-phrase` benchmark);
4. normal room with detection and TTS active.

Useful sink controls (PipeWire/PulseAudio):

```bash
pactl set-sink-mute @DEFAULT_SINK@ 1
pactl set-sink-mute @DEFAULT_SINK@ 0
```

Phrases: `read this`, `read that`, `read`, `what does it say`,
`what's that say`, `read it`, `what does this say`, `detection off`,
`detection on`.

Read the log in this order:

| Last stage seen | Meaning / next check |
|---|---|
| `listen_wait` then `listen_timeout` only | Mic not crossing energy threshold; check device/index and gain. |
| `audio_captured seconds≈0` or no capture | Wrong device or capture rate. |
| `vosk_result final=""`, `tts_overlap=True` | Likely self-speech. Retest muted; then try `WHISPER_HAT_VOICE_COOLDOWN=0.35`. |
| `vosk_result final=""`, `tts_overlap=False` | Mic/sample-rate or recognition. Compare `mic_open source=...` with `arecord`. |
| `vosk_result final="read this"` but `command_ignored` | Routing bug; capture the exact final text. |
| `command_routed command=ocr_read` but no `ocr_triggered` | Trigger cooldown or feature disabled. |
| `ocr_triggered worker_alive=False` | OCR worker not started/failed. |
| `ocr_start` / `ocr_tesseract_done` missing | Frame grab or Tesseract failure (`ocr_failed`). |
| `ocr_speech_queued queued=True` but no `tts_start` | Speech queue/worker blocked; look for `speech_dropped` or `speech_queue_cleared`. |
| `tts_start` but no sound | `espeak-ng`/`pw-play`/sink problem, not Vosk. |

Check the actual ALSA device and rate independently:

```bash
arecord -l
pactl list sources short
arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 -d 5 /tmp/whisper-mic-test.wav
aplay /tmp/whisper-mic-test.wav
```

If the mic only works at 48 kHz, set `VOICE_MIC_SAMPLE_RATE = 48000` or rely on
the fallback and confirm `mic_open source=48000` in the log. The resampler
converts to 16 kHz for Vosk.

Do not upgrade the Vosk model until the log shows the failure is recognition
accuracy rather than mic/routing/queue. If Vosk returns the right text, a
larger model cannot fix the routing.

### 3.3 Obstacle-alert test matrix

```bash
WHISPER_HAT_UNKNOWN_DEBUG=1 WHISPER_HAT_LATENCY_DEBUG=1 \
python whisper.py 2>&1 | tee /tmp/whisper-alerts.log
```

Test four cases and record spoken alerts plus `[unknown obstacle]` lines:

1. **Empty path**: no candidate, no alert.
2. **Stationary obstacle** around 1.5 m: one notice after stabilization, then
   deduplication; `action=candidate` should not repeat every band change.
3. **Approaching obstacle**: a materially closer reading (>0.40 m) or risk
   escalation may speak once; the same distance must not chatter.
4. **New obstacle immediately after the first**: remove the first, wait for
   `path_clear`, then present a second. It should speak as a new hazard.

Also record `valid` and `spread`. A high spread with a stable median is usually
a noisy/mis-mounted sensor; a stable high-`valid` close reading is more likely
a real obstacle. Do not lower thresholds just to silence the demo.

### 3.4 Reproducible vision benchmark

Capture a fixed scene set with the production camera settings:

```bash
mkdir -p bench/scenes
for i in $(seq -w 1 20); do
  rpicam-still -n --width 640 --height 480 \
    --autofocus-mode continuous -t 2000 \
    -o "bench/scenes/scene-$i.jpg"
  sleep 1
done
```

Use the same directory, `--imgsz`, and confidence for every candidate. Include
hard lighting, the objects the current model misses, and a labeled validation
set if you want mAP:

```bash
python tests/bench_pi.py \
  --models yolo11n.pt,yolo11s.pt,yolo11m.pt \
  --images bench/scenes \
  --imgsz 320 --conf 0.40 --loops 10 \
  --val-data /path/to/labeled-scenes.yaml \
  --json-out bench/results.json

# OCR and TTS latency on the same frames/device:
python tests/bench_pi.py \
  --models yolo11n.pt --images bench/scenes \
  --ocr --tts-phrase "Caution. Person ahead." \
  --json-out bench/ocr-tts.json

# Combined load: detection + TTS + optional offline voice listening.
# Keep --voice only when the microphone is connected; otherwise use the first
# offline voice test separately.
python tests/bench_pi.py \
  --models yolo11n.pt --images bench/scenes \
  --combined-seconds 60 --voice \
  --app-log /tmp/whisper-voice.log \
  --json-out bench/combined.json
```

The JSON records model size, detection median/p95/FPS, detection counts,
optional mAP, OCR median/p95, TTS wall time, peak RSS, and `vcgencmd`
temperature/`get_throttled` before and after combined load. Run it on battery
or with the official 5 A supply and the active cooler; record the power mode.

For end-to-end hazard/command latencies, the app logs:

```text
[latency] hazard_to_tts_start=...ms
[latency] command_to_tts_start=...ms
```

Collect them from the Pi log and report median and p95 per condition. Pass the
same log to `tests/bench_pi.py --app-log /tmp/whisper-voice.log` to include the
median/p95 in the JSON. The hazard timer starts when the camera frame is read;
the command timer starts when the recognized command is routed to OCR/scene
handling. They do not include the time from the user’s first spoken phoneme to
Vosk’s final result.

## 4. Recommendation based on the current evidence

- **Keep YOLO11n at 320 as the default** until our Pi benchmark shows a
  same-class accuracy gain that justifies the latency cost. The published Pi 5
  and export data show large speed differences between n/s/m; do not assume
  spare CPU once detection, Vosk, TTS, ToF, face, and emotion work overlap.
- **YOLO11s is the first accuracy candidate** if the benchmark shows useful
  same-class gains and p95 hazard-to-spoken latency stays within the chosen
  budget. Start at 384 with the opt-in profile, then try 320/416 on our scenes.
- **YOLO11m is not recommended for real-time Pi 5 CPU** based on the published
  ~0.39–0.44 FPS at 640. It is an accuracy reference, not a wearable default.
- **Do not replace YOLO11n with YOLOE for urgent alerts.** YOLOE adds
  vocabulary, not same-class accuracy, and its zero-shot mAP is lower. Test it
  only as an on-demand “what’s ahead?” experiment with frozen prompts and a
  measured prompt-set cost.
- **Keep scene description detections-based** unless an on-demand captioner
  benchmark shows a useful result within the latency/RAM budget. Never use a
  captioner as the urgent hazard source.
- **Export format is an experiment, not an assumption.** Benchmark PyTorch,
  ONNX, OpenVINO, and NCNN on the same frames. Published Pi 5 exports disagree
  with each other; only our device measurement decides.

## 5. What remains unverified

- Actual microphone recognition rates for the new aliases on the Pi.
- Whether TTS overlap is the dominant cause of missed commands, and whether the
  opt-in cooldown helps without hiding commands.
- Actual sample rate the USB mic runs at (the log now reports it).
- Real ToF noise/spread with the hat mounted, and whether a stationary obstacle
  still produces band changes after smoothing.
- YOLO11n/s/m accuracy and latency on our scenes at 320/384, under combined
  load, with our camera settings.
- Whether NCNN/ONNX/OpenVINO exports improve our end-to-end latency after
  tracking and pre/post-processing are included.
- Peak RAM and thermal throttling on the Pi 5 during a long combined run.
- Any YOLOE Pi 5 latency or scene-captioner Pi 5 latency; no credible public
  figures were found in this review.

Do not fill these gaps with estimates. Run the commands above and attach the
JSON/log files to the commit or issue.
