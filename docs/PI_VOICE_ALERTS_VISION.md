# Pi 5 voice, obstacle alerts, and vision notes

This document separates three things that were getting mixed together:

1. **Confirmed from this repository** (code paths, log contents, tests).
2. **Published third-party measurements** (sources cited; not our hardware).
3. **Still unverified on our Pi** (what to run and what to look for).

The working baseline before the current uncommitted changes is commit `fd76cda`
(`Fix voice command routing and dedupe unknown-obstacle alerts`). The default
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

### 1.1.1 Why a visible chair/box still became “unknown obstacle”

The generic warning was not only a YOLO-vocabulary problem. A known detection
only suppressed the generic warning when its box received a **ToF cell match**
(`distance_source == "tof"`). If the chair/box was below the calibrated ToF
overlap, or outside its cells, the known detection had no ToF match and the
generic path warning could still win. Close/low objects are the common case:
their box can sit below the overlap rectangle even though the camera sees them.

What changed:

- `tof_distance_for_box(..., diagnostics=...)` now records why a match failed:
  `box_outside_overlap_horizontal`, `box_outside_overlap_vertical`,
  `no_valid_cells`, `tof_not_fresh`, or `matched`.
- `TOF_BOX_CLAMP_LOW_OBJECTS` is an opt-in fallback
  (`WHISPER_HAT_TOF_CLAMP_LOW=1`). It clamps a box whose sampled lower part is
  just below the overlap to the bottom ToF row instead of dropping the match,
  and logs `matched_below_overlap_clamped`. It is **off by default**: the
  bottom ToF row may see the floor beyond the object, so enable it only if the
  association log shows the object really is in that row.
- The main loop tracks the nearest known object in the path from **both** ToF
  and camera distance. `unknown_is_separate_from_known()` only keeps the generic
  warning when no known object is close enough to explain the depth reading.
- Static-obstacle speech now names the class (`chair close in path`) instead of
  saying only `obstacle close in path`. The generic wording remains when no
  confident label exists.

Opt-in evidence line (one per class per second):

```bash
WHISPER_HAT_ASSOCIATION_DEBUG=1 python whisper.py
```

It prints the camera class/confidence/box/foot point, whether the foot point is
in the walking corridor, the ToF match distance and reason, sampled rows/cols,
and the visual/ToF source. That separates these cases:

| Association line | Meaning |
|---|---|
| `source=tof reason=matched` | Camera box and depth agree; class name should win. |
| `reason=box_outside_overlap_vertical` | Low/close box was outside the calibrated overlap; check the new clamp. |
| `reason=box_outside_overlap_horizontal` | Object is outside the ToF field of view; the camera may still need a visual distance fallback. |
| `reason=no_valid_cells` | ToF saw the box region but returned no valid ranges; check mounting/status. |
| no association line for the object | YOLO did not detect it above `CONFIDENCE`; this is a model/vocabulary or framing problem. |

The repo-side regression tests simulate a chair box over close ToF cells, a low
box below the overlap, and the generic-warning arbitration. They do not prove
the real camera sees the chair. The Pi run must show either an association line
or a missing detection.

The checked-in `whisper_hat_tof_log.csv` is consistent with both failure modes:
it contains 109 generic alerts but only 11 chair alerts. Several chair alerts
were within five seconds of a generic alert whose distance was different
(for example, chair at 1.44 m and generic at 0.88 m, or chair at 2.40 m and
generic at 1.15 m). That can mean the chair association was dropped, or that a
second closer object was genuinely present. The new association line is needed
to tell those apart; do not assume every nearby chair/generic pair is one
object.

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
trigger anything.” Urgency is now change-based: a new, escalated, or materially
closer STOP interrupts; an unchanged repeat queues, is rechecked for presence,
and can speak between OCR chunks.

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

### 1.4 Mid-session listening dropout and watchdog

The supplied Pi log showed the process and ToF loop still running after the
last voice line, so this is not a whole-process crash. The remaining candidates
are a blocked capture read, a dead listener worker, stale playback suppression,
or a speech queue that keeps dropping. The log did not contain enough
per-stage audio/Vosk/queue evidence to choose between them.

What changed (opt-in diagnostics, baseline behavior otherwise):

- The listener now records a heartbeat at every real stage: `opening`,
  `opened`, `calibrating`, `listening`, `processing`, `error`, `reopening`.
- A watchdog runs every `VOICE_WATCHDOG_INTERVAL_SECONDS` (1 s). It recovers
  **only** when the worker thread is dead or a stage heartbeat is older than
  `VOICE_WATCHDOG_STALE_SECONDS` (12 s). Normal silence is not a failure: the
  1 s `listen()` timeout refreshes the heartbeat.
- Recovery closes/stops the current PyAudio stream so a blocked native read can
  return, then lets the worker reopen the microphone. Retries are rate-limited
  by `VOICE_RECOVERY_MIN_INTERVAL_SECONDS` (30 s) and the skip reason is logged.
- Stale playback suppression is cleared only when `speech_active_event` has
  been set for more than `VOICE_STALE_PLAYBACK_SECONDS` (25 s) **and** the
  playback process is gone or finished. A genuinely playing warning is not
  cleared.
- Session counters separate `vosk_nonempty` (heard), `matched` (command
  grammar), `dispatched` (routed to OCR/scene/detection), and
  `answered_aloud` (a command-origin TTS utterance actually started). Queue
  drops and queue clears are counted separately.
- `print_voice_summary()` always prints those counters at shutdown. With
  `WHISPER_HAT_VOICE_STATUS=1` it also prints a status line every 30 s.
  The watchdog can be disabled for an A/B run with
  `WHISPER_HAT_VOICE_WATCHDOG=0`.
- `detection off` clears detection state and interrupts speech, but it does
  **not** set `voice_control_stop_event`; the listener stays alive so
  `detection on` can be recognized again. The regression test asserts this.

Diagnostic command:

```bash
WHISPER_HAT_VOICE_DEBUG=1 WHISPER_HAT_VOICE_STATUS=1 WHISPER_HAT_LATENCY_DEBUG=1 python whisper.py 2>&1 | tee /tmp/whisper-voice.log
```

The new `playback_suppression_start` / `playback_suppression_clear` events,
`listen_timeout`, `audio_captured`, `vosk_result`, `command_*`, `speech_*`, and
the `[voice status]` line should identify the exact stage that stops. Do not
conclude that TTS feedback is the cause until the log shows
`vosk_result final="" tts_overlap=True`; if it does, compare audible vs.
physically muted speaker while software playback still runs.

### 1.5 OCR garbage, long playback, and urgent STOP

The Pi log confirms the OCR failure mode:

- `ocr_frame frame_shape=(480, 640, 3)` — Tesseract ran on the detection frame.
- `raw_chars=996 chars=600 elapsed_s=6.395` — a long block of low-quality text
  was cleaned and queued at caution priority.
- `duration_s=20.025` — that block hit the 20 s playback timeout.
- `hazard_to_tts_start=6056.9ms` — a risk-3 STOP waited behind the reading.

What changed:

- `_run_tesseract_tsv()` requests Tesseract TSV output so the app gets per-word
  confidence, text boxes, and word counts.
- `_ocr_quality_ok()` rejects results when confidence or language-likeness is
  weak: mean/good-word confidence, alphabetic ratio, vowel-bearing words,
  long consonant runs, and (for longer blocks) a minimum fraction of common
  English function words. A rejected result speaks only:
  `I couldn't read the page. Hold it steady and try again.`
- `_grab_ocr_frame()` can ask the camera for a one-off high-resolution still
  (`OCR_USE_STILL`). It is **opt-in and off by default** because the still-mode
  switch has not yet been measured on the Pi. The detection stream remains at
  `CAMERA_WIDTH x CAMERA_HEIGHT`; only the OCR request switches modes briefly.
  The still is downscaled to `OCR_MAX_WIDTH` before Tesseract. Defaults are
  1920x1440 still, 1024-pixel OCR width, and a 20 s OCR subprocess timeout;
  all are environment-overridable. Enable it for the A/B test with
  `WHISPER_HAT_OCR_STILL=1`.
- `LatestFrameSource` records the gap from the last fresh detection frame to
  the first frame after a still capture and logs
  `[camera] high-res still frame gap=... max=....` Detection and obstacle
  alerts cannot update while no fresh frame exists, so that line is the direct
  measure of the alert blind window. The main loop waits through the still
  instead of treating the pause as a dead camera.
- `ocr_frame` now logs `frame_origin=still|peek|read`, original/OCR frame
  shapes, blur score, brightness, contrast, and text-box placement.
- `ocr_tesseract_done` logs `quality_ok`, word count, mean confidence,
  good-word ratio, alpha/vowel/long-consonant/common-word ratios, text box, and
  elapsed time.
- Readable text is split into up to `OCR_MAX_SPEECH_CHUNKS` chunks of about
  `OCR_SPEECH_CHUNK_CHARS` characters. The first chunk is queued first, so a
  single playback cannot hold the worker for 20 s. `ocr_first_chunk_queued`
  records the time to useful speech.
- STOP urgency is change-based. A new, escalated, or materially closer STOP
  calls `interrupt_current_speech(force=True)` immediately. An unchanged
  repeat of the same STOP queues normally, so a stationary book does not keep
  cutting off its own reading. Later OCR chunks have lower queue priority than
  routine notices/cautions, so a repeat can speak between chunks.
- Queued repeats are rechecked at dequeue time with `hazard_still_present()`.
  If the hazard has not been refreshed within 1.5 s, or its current risk has
  dropped below the queued risk, the repeat is dropped and logged as
  `speech_repeat_dropped reason=hazard_not_present`.

The listener stall log is consistent with a blocked native capture read, not
an ALSA error causing it: the watchdog reported no progress for 12.5 s at
`stage=listening`, closed the stream, and only then did ALSA report
`alsa_snd_pcm_mmap_begin` / `Unanticipated host error`. The stream reopened and
the mic recalibrated. The next run adds `listen_wait phrase_limit_s=4.0
stream_active=...` and `listen_return` so a blocked read can be separated from
a Vosk/CPU stall. Treat the ALSA lines as a consequence of closing a stuck
stream unless a `listen_wait`/`listen_return` pair shows otherwise.

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

### 2.4 Implemented open-vocabulary prototype

The default remains YOLO11n/COCO. A YOLOE candidate is wired behind an explicit
flag:

```bash
# Fixed prompt list, first run may download/build CLIP text embeddings
WHISPER_HAT_OPEN_VOCAB=1 WHISPER_HAT_OPEN_VOCAB_MODEL=yoloe-11s-seg.pt python whisper.py
```

The default prompt list is intentionally small and obstacle-oriented:
`chair`, `stool`, `bench`, `cardboard box`, `box`, `package`, `bag`,
`backpack`, `handbag`, `suitcase`, `trash can`, `bin`, `basket`,
`laundry basket`, `shopping cart`, `cart`, `shoe`, `shoes`, `table`, `desk`,
`door`, `bucket`, `cooler`, `crate`, `traffic cone`, `stroller`, `walker`,
`vacuum cleaner`, `fan`, `heater`, `plant`, `pet bowl`, `cable`, `toy`.

Override with a comma-separated list:

```bash
WHISPER_HAT_OPEN_VOCAB=1 WHISPER_HAT_OPEN_VOCAB_CLASSES='chair,stool,box,cardboard box,laundry basket,trash can,bin,basket,cart,shoe' python whisper.py
```

The opt-in model runs continuously at `WHISPER_HAT_OPEN_VOCAB_IMG_SIZE`
(defaults to `IMG_SIZE`) and uses the same `PROCESS_EVERY_N_FRAMES` schedule.
`load_detection_model()` removes the COCO ID filter for this model, because
YOLOE class IDs are not COCO IDs. `STATIC_OBSTACLE_CLASSES` includes the prompt
names so they get obstacle risk rather than passive-object risk.

For an offline Pi deployment, bake the prompt classes into an export once on a
development machine (with network access) and point at the export. The exact
export support depends on the installed ultralytics version; verify the file
loads before copying it to the Pi:

```bash
python -c "from ultralytics import YOLOE; names=['chair','stool','box','cardboard box','laundry basket','trash can','bin','basket','cart','shoe']; m=YOLOE('yoloe-11s-seg.pt'); m.set_classes(names, m.get_text_pe(names)); m.export(format='onnx', imgsz=320)"
```

Then set `WHISPER_HAT_OPEN_VOCAB_MODEL` to the exported file and, if the export
has baked names, `WHISPER_HAT_OPEN_VOCAB_PROMPT_FREE=1`. Do not assume an export
is faster on the Pi; measure it with the same scenes.

If continuous YOLOE does not meet the warning-latency budget, the safe fallback
is to keep YOLO11n for urgent alerts and run the broad model only for an
explicit “what’s ahead?” request. That on-demand path is not enabled by
default; benchmark continuous YOLOE first and record the latency before adding
it to the alert loop.

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
git switch -c baseline-fd76cda fd76cda
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

### 3.4 Controlled camera/ToF association test

Run with the association log and place each object at roughly 0.5 m, 1.0 m,
and 1.5 m in the walking path:

```bash
WHISPER_HAT_ASSOCIATION_DEBUG=1 WHISPER_HAT_UNKNOWN_DEBUG=1 WHISPER_HAT_LATENCY_DEBUG=1 python whisper.py 2>&1 | tee /tmp/whisper-association.log
```

Objects, in this order: chair, cardboard box, laundry basket, trash can, low
stool. For each distance, record:

1. whether an `[association]` line appears at all (camera detection);
2. its `class`, `conf`, `foot`, `in_path`, `tof`, `reason`, and `source`;
3. the spoken message and whether it named the object or stayed generic;
4. whether the object is visible in the preview at the bottom of the frame.

If the camera sees the object but the association line is missing, the failure
is YOLO/vocabulary or framing. If the association line exists but says
`box_outside_overlap_*`, tune the mount/overlap or use the visual distance
fallback. Do not tune the model until these are separated.

### 3.5 Reproducible vision benchmark

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
set if you want mAP. A label file maps each image filename to one expected
name or a list of acceptable aliases:

```json
{
  "scene-01.jpg": {"expected": ["chair"], "tof_visible": true},
  "scene-02.jpg": {"expected": ["cardboard box", "box"], "tof_visible": true},
  "scene-03.jpg": {"expected": ["laundry basket", "basket"], "tof_visible": true},
  "scene-04.jpg": {"expected": ["trash can", "bin"], "tof_visible": false}
}
```

`tof_visible` records whether the ToF path warning fired for that scene (or
whether the obstacle was present in the ToF region). Without it, a no-name
result is reported as `generic_or_missed`; with it, the benchmark can separate
`generic` (ToF saw it, model did not name it) from `missed` (neither the model
nor ToF flagged it).

`--labels` adds separate `correct`, `wrong`, `generic`, `missed`, and
`generic_or_missed` counts. The per-image detail records exactly which expected
aliases and detected names were present, so the outcomes can be audited rather
than inferred from an aggregate.

```bash
python tests/bench_pi.py \
  --models yolo11n.pt,yolo11s.pt,yolo11m.pt \
  --images bench/scenes \
  --imgsz 320 --conf 0.40 --loops 10 \
  --val-data /path/to/labeled-scenes.yaml \
  --labels bench/labels.json \
  --json-out bench/results.json

# Open-vocabulary candidate on the same scenes. Continuous YOLOE first:
python tests/bench_pi.py \
  --open-vocab --models yoloe-11s-seg.pt \
  --images bench/scenes --imgsz 320 --conf 0.40 --loops 5 \
  --labels bench/labels.json \
  --json-out bench/yoloe.json

# Prompt-free/exported model with baked class names:
python tests/bench_pi.py \
  --open-vocab --open-vocab-prompt-free \
  --models /path/to/yoloe-export.onnx \
  --images bench/scenes --imgsz 320 --loops 5 \
  --labels bench/labels.json \
  --json-out bench/yoloe-export.json

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
optional mAP, OCR median/p95 and quality acceptance counts, TTS wall time,
peak RSS, and `vcgencmd` temperature/`get_throttled` before and after combined
load. Run it on battery or with the official 5 A supply and the active cooler;
record the power mode.

For end-to-end hazard/command latencies, the app logs:

```text
[latency] hazard_to_tts_start=...ms message='...'
[latency] command_to_tts_start=...ms message='...'
```

The app log parser now reports `hazard_named` and `hazard_generic` separately
by looking for `unknown obstacle` in that message, in addition to the combined
`hazard` sample.

Collect them from the Pi log and report median and p95 per condition. Pass the
same log to `tests/bench_pi.py --app-log /tmp/whisper-voice.log` to include the
median/p95 in the JSON. The hazard timer starts when the camera frame is read;
the command timer starts when the recognized command is routed to OCR/scene
handling. They do not include the time from the user’s first spoken phoneme to
Vosk’s final result.

### 3.6 Fifteen-minute persistence and vocabulary run

Run the production model first, then repeat only the obstacle cases with YOLOE.
Keep voice listening and TTS active for every run so the test does not hide a
voice regression.

```bash
WHISPER_HAT_VOICE_DEBUG=1 WHISPER_HAT_VOICE_STATUS=1 WHISPER_HAT_LATENCY_DEBUG=1 WHISPER_HAT_ASSOCIATION_DEBUG=1 WHISPER_HAT_UNKNOWN_DEBUG=1 python whisper.py 2>&1 | tee /tmp/whisper-15min.log
```

At the start, midway (about 7 minutes), and near the end (about 14 minutes),
say each of these twice: `read this`, `read that`, `detection off`,
`detection on`. Between them, trigger repeated alerts and interrupt one spoken
response with a new STOP. The expected result is that `detection on` is still
recognized after `detection off`, the listener summary shows a recent
`last_audio_age_s`, and the watchdog logs a recovery only if the stream/worker
actually failed.

At shutdown, copy the final line:

```text
[voice summary] audio_frames=... vosk_nonempty=... matched=... dispatched=... answered_aloud=... ...
```

Report these as separate measurements, not one “success” number:

- **recognized**: `vosk_nonempty`, plus how many were the expected phrase;
- **dispatched**: `matched` / `dispatched` (OCR, scene, or detection command);
- **answered aloud**: `answered_aloud` (TTS actually started);
- **latency**: median and p95 of `command_to_tts_start` for successful
  attempts, from `WHISPER_HAT_LATENCY_DEBUG=1` logs;
- **warning latency**: median and p95 of `hazard_to_tts_start` for correct
  obstacle names vs generic warnings;
- **recoveries/drops**: `recoveries`, `speech_drops`, `speech_cleared`.

For the obstacle/vocabulary matrix, use the five objects from 3.4 at three
distances. For each candidate record `correct name`, `wrong name`, `generic`
(no named detection; ToF warning only), or `missed` (no warning at all), plus
whether the object was visible in the camera frame. Run YOLO11n, YOLO11s, and
one YOLOE candidate on the same frames and conditions. Do not call the dropout
fixed, or the vocabulary upgrade successful, until these Pi logs exist.

### 3.7 OCR quality and interruption test

#### Next Pi test: inspect the actual images first

The Pi run at `863a6ea` did not read the book successfully. Both high-resolution
attempts captured 1920x1440 and sent 1024x768 to Tesseract: one returned no words,
the other 24 words at mean confidence 44. Both were correctly rejected. The
observed detection-frame gaps were about 0.26-0.29 s. Without the images, focus,
framing, orientation, resizing, and preprocessing remain undiagnosed.

Stop the WhisperingHat boot service **before** a manual run so it releases the
camera and microphone. The service unit name is not recorded in this repository;
find it with `systemctl list-unit-files --type=service | grep -i whisper`, then
run `sudo systemctl stop ACTUAL_UNIT.service` (use the actual unit name). If the
boot launcher has another name, stop that launcher instead. Stay **stationary**
throughout this high-resolution test because still capture pauses fresh obstacle
updates.

```bash
cd ~/WhisperingHat
WHISPER_HAT_OCR_STILL=1 WHISPER_HAT_OCR_DUMP_ONCE=1 \
  WHISPER_HAT_OCR_QUALITY=1 WHISPER_HAT_OCR_MAX_WIDTH=1024 \
  WHISPER_HAT_OCR_STILL_WIDTH=1920 WHISPER_HAT_OCR_STILL_HEIGHT=1440 \
  WHISPER_HAT_VOICE_DEBUG=1 WHISPER_HAT_LATENCY_DEBUG=1 \
  .venv/bin/python -u whisper.py 2>&1 | tee /tmp/whisper-ocr-dump.log
```

Say **"read this"** with the real printed page in view. The first request with
an image writes one matched PNG pair, including when OCR returns no words or
the quality gate rejects it. Subsequent requests in that app run write nothing;
restart to collect another pair. The option is off by default and does not
enable high-resolution capture by itself.

`[ocr dump]` logs the capture source, dimensions, and absolute paths even without
voice debugging. The private output directory is
`/tmp/whisper-hat-ocr-<id>/` on a standard Pi setup (the system temporary directory
is used; trust the logged paths if `TMPDIR` differs). Both filenames share that
attempt ID:

- `whisper-hat-ocr-<id>-still-original.png`: capture output saved before OCR
  resizing or other OCR transforms.
- `whisper-hat-ocr-<id>-still-tesseract-input.png`: the exact encoded PNG bytes
  piped to Tesseract, with no independent reconstruction.

If still capture falls back, the pair uses `preview-peek` or `preview-read`
instead of `still`. Files stay local, outside Git; the dump refuses a temporary
directory inside a Git working tree. Write failures are logged and OCR continues
without retrying the dump. A failed write can leave a partial pair.

After Ctrl+C, retrieve the two images from Windows PowerShell. Replace
`PI_USER`, `PI_HOST`, and the directory below with the exact logged directory:

```powershell
scp "PI_USER@PI_HOST:/tmp/whisper-hat-ocr-<id>/*.png" "$env:USERPROFILE\Downloads\"
```

Inspect both images against the actual printed page. If the original is sharp
and upright but the Tesseract input loses legibility, compare the current
pipeline against **crop the page before resizing**, using the **same saved
original** and the same OCR settings. Judge actual recognized text against the
page, not confidence alone. If the original is blurry, sideways, or clipped,
report that evidence first; cropping or loosening the gate is not an established
fix. Keep the quality gate enabled. No diagnosis or crop change is justified
until these images are available.

Track the microphone stall independently: the watchdog fired while TTS was
inactive, so playback overlap alone cannot explain it. Preserve the preceding
`listen_wait`, watchdog, and next `listen_return` (or its absence), including TTS
state, for a separate investigation. This image-dump patch does not change the
recognizer or microphone recovery.

#### Mirror finding from saved pair `g0m114ny`

Both saved PNGs are horizontally mirrored. The 1024x768 input matches an
`INTER_AREA` resize of the 1920x1440 original pixel for pixel, so OCR resizing
did not introduce the reflection. `Picamera2Source` configures preview and
still capture with the same `Transform(hflip=False, vflip=True)`. That single
axis reflection reverses handedness; with this camera mounting, the resulting
text is upright but horizontally mirrored.

OCR now receives the capture's reflection metadata through `LatestFrameSource`.
`_prepare_ocr_frame` applies `cv2.flip(frame, 1)` once to the OCR copy when the
camera transform has an odd number of axis reflections, before the existing
resize. Sources without this reflection are unchanged. The original dump stays
untouched; the input dump still contains the exact Tesseract bytes. Camera
transforms, shared detection pixels, left/right obstacle cues, and the quality
gate are unchanged. `ocr_frame ocr_unmirrored=True` records the correction.

Local Windows comparison of this saved input used native Tesseract 5.5.4 via
tesserocr, English `tessdata_fast`, PSM 3, and the same 1024x768 pixels except
for the horizontal flip. **Both as-is and flipped returned zero nonblank words**;
neither recovered the visible heading "The Call of the Wild" or the phrase
"the sweep of the current". A whitespace-only TSV token with confidence 95 is
not a recognized word. A known-text sanity check passed using the same engine.
The Pi engine version/traineddata were not available for matching, so this is
a Windows result, not a Pi accuracy claim.

The photographed book occupies the lower portion of a cluttered scene, is
clipped by the bottom edge, and has strong sunlight/shadow bands across its
text. The next Pi run should use the command above, **one complete page
centered and filling most of the frame under even lighting**, while stationary
with the boot service stopped. Check the untouched original against the
corrected input and compare recognized words against the page. Do this before
tuning resolution, cropping, segmentation, or quality thresholds. The mirror
correction is established; successful full-page reading still needs a Pi test.

#### Broader OCR and interruption matrix (after image inspection)

Use a real book page, not a synthetic screenshot. Run with the detection stream
unchanged and test both preview-frame OCR and the high-resolution still:

```bash
# A: preview frame only (A/B baseline)
WHISPER_HAT_OCR_STILL=0 WHISPER_HAT_VOICE_DEBUG=1 WHISPER_HAT_LATENCY_DEBUG=1 \
  .venv/bin/python -u whisper.py 2>&1 | tee /tmp/whisper-ocr-preview.log

# B: one high-resolution still per "read this"
WHISPER_HAT_OCR_STILL=1 WHISPER_HAT_VOICE_DEBUG=1 WHISPER_HAT_LATENCY_DEBUG=1 \
  .venv/bin/python -u whisper.py 2>&1 | tee /tmp/whisper-ocr-still.log
```

Keep the **same upright, well-lit page at the same distance** for the first A/B
comparison so resolution is the only variable. Fill most of the frame with
text, hold the page flat, and keep the camera level. Record the printed text
first, then compare what the hat actually reads against it; `quality_ok=True`
alone is not a pass. Then repeat at about 0.5 m, 0.8 m, and 1.2 m. Try rotated
pages separately (90 and 180 degrees) after the upright comparison, so
orientation does not confound the resolution result. Do not change
preprocessing thresholds until focus/framing is ruled out. Keep the quality
gate enabled while collecting and comparing the images.

Read these fields from each `read this` attempt:

| Field | What it tells you |
|---|---|
| `ocr_frame frame_origin=still/peek` | Whether the high-resolution still path worked. |
| `original_shape` / `ocr_shape` | Capture resolution and the actual Tesseract input. |
| `blur_score` / `brightness` / `contrast` | Focus and exposure before blaming Tesseract. |
| `ocr_tesseract_done quality_ok` | Whether the garbage gate accepted the page. |
| `mean_conf`, `common_word_ratio`, `text_box` | Why the gate accepted/rejected, and whether text fills the frame. |
| `ocr_first_chunk_queued elapsed_s` | Time from command to first speakable chunk being queued. |
| `command_to_tts_start` | Time from command to the first OCR audio actually starting. |
| `ocr_quality_rejected` | Rejected garbage; the spoken fallback should follow. |
| `[camera] high-res still frame gap` | Longest gap between fresh detection frames and the alert blind window during a still. |
| `speech_repeat_dropped` | Queued repeat skipped because the hazard was stale or no longer at that risk. |

Then verify interruption in two separate cases:

1. **Stationary book:** while its reading is playing, leave the same book/
   obstacle in place. An unchanged repeat should queue, not call
   `speech_interrupt force=True`, and should not repeatedly cut off the
   reading. It may speak between chunks.
2. **New or closer obstacle:** add a separate obstacle, or move the existing
   one materially closer. Record `speech_interrupt force=True`,
   `hazard_to_tts_start`, and whether the STOP is heard before the remaining
   OCR chunks. It should interrupt promptly; a risk-2 caution should not kill
   the reading.

Finally, without restarting, say `read that` and then `detection on`. Confirm
`vosk_nonempty`, `command_routed`, and `ocr_triggered` increment after the
reading, and check `[voice summary]` for `recoveries`/`audio_frames`. If the
watchdog fired, include the preceding `listen_wait` line and the next
`listen_return` or absence of it.

Report separate numbers for preview vs. still and for each distance: the
printed text, what the hat actually read, accepted pages, rejected pages,
wrong/gibberish speech, OCR time, time to first useful speech, the longest
`high-res still frame gap`, and whether a new/closer STOP interrupted within
1 s. State separately whether an unchanged stationary repeat stayed queued.

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
- **Do not replace YOLO11n with YOLOE for urgent alerts yet.** YOLOE is now
  wired as an opt-in continuous candidate with a frozen obstacle prompt list,
  but it adds vocabulary, not same-class accuracy, and its zero-shot mAP is
  lower. Promote it only if the same-scene Pi run shows more correct names
  without worsening median/p95 time-to-warning or the voice counters. If
  continuous inference is too slow, keep YOLO11n for the first warning and run
  the broad model only for an explicit “what’s ahead?” request.
- **Keep scene description detections-based** unless an on-demand captioner
  benchmark shows a useful result within the latency/RAM budget. Never use a
  captioner as the urgent hazard source.
- **Export format is an experiment, not an assumption.** Benchmark PyTorch,
  ONNX, OpenVINO, and NCNN on the same frames. Published Pi 5 exports disagree
  with each other; only our device measurement decides.

## 5. What remains unverified

- Actual OCR acceptance/rejection rates, high-resolution still latency, and
  time to first useful speech on a real book page. The new gate is repo-side
  only until the Pi A/B run in 3.7.
- Whether a new/escalated/materially closer STOP interrupts promptly with OCR
  chunks active, while an unchanged stationary repeat does not keep cutting off
  the reading.
- Actual microphone recognition rates for the new aliases on the Pi.
- Whether TTS overlap is the dominant cause of missed commands, and whether the
  opt-in cooldown helps without hiding commands.
- Whether the new watchdog actually recovers the mid-session dropout. The
  repo-side tests cover dead-worker/stale-heartbeat/stale-playback logic only.
- Actual sample rate the USB mic runs at (the log now reports it).
- Real ToF noise/spread with the hat mounted, and whether a stationary obstacle
  still produces band changes after smoothing.
- Whether close/low objects are visible in the real camera frame and whether
  `matched_below_overlap_clamped` fires at the distances where the user saw
  “unknown obstacle.”
- YOLO11n/s/m accuracy and latency on our scenes at 320/384, under combined
  load, with our camera settings.
- YOLOE-11s correct/wrong/generic/missed rates and median/p95 warning latency
  on the Pi, including RAM and thermal behavior under combined voice/TTS load.
- Whether NCNN/ONNX/OpenVINO exports improve our end-to-end latency after
  tracking and pre/post-processing are included.
- Peak RAM and thermal throttling on the Pi 5 during a long combined run.
- Any scene-captioner Pi 5 latency; no credible public figures were found in
  this review.

Do not fill these gaps with estimates. Run the commands above and attach the
JSON/log files to the commit or issue.
