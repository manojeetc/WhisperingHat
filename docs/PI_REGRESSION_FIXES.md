# Pi regression fixes after 3b03ea1

PortAudio capture now lives in one supervised subprocess. Only that process
opens, reads, stops and closes the native stream. The listener reads a bounded
eight-chunk pipe queue and checks recovery/shutdown every 50 ms. No PCM for three
seconds (including a native read stall) causes teardown. Zero-valued silence is
valid progress. Teardown terminates, waits, kills if needed, and reaps the old
owner before opening another microphone. An unconfirmed teardown prevents reopen.
The watchdog requests recovery; it never touches a native stream from another
thread. This fixes the demonstrated ownership defect, without establishing the
cause of the earlier allocator crash.

Capture continues throughout TTS and vision inference. Speaker playback and a
short 250 ms tail are silenced only in the recognizer's PCM copy. The configured
post-playback cooldown can extend that tail. Recognition resumes automatically
after normal completion or interruption. Commands spoken over the hat's speaker
are not supported without acoustic echo cancellation.

Read/describe aliases still use offline Vosk. Accepted requests receive a short
acknowledgment, each feature has at most one pending/running request, and failures
clear that slot. Inference and input writes have deadlines. Timed-out model owners
are reaped and can restart on a later request. OCR still unmirrors once, uses
PP-OCRv5 first and Tesseract as fallback, and adds clipped-page/uncertain-page
rejection before speaking. See [VOICE_VISION.md](VOICE_VISION.md) for limits.

Approach/TTC speech now requires at least four distinct fresh ToF samples over
at least 0.6 seconds, a consistent decreasing trend and at least 0.2 m net change
on the same tracked object. Source changes and sampling gaps reset the trend;
untracked boxes and visual size estimates cannot claim approach. Distance-based
close/STOP rules still run independently. Unchanged generic notice/caution
reminders are deduplicated; first warnings, material closing, escalations and
urgent STOP reminders remain enabled.
Unchanged stationary-object notice/caution repeats and direction/band wording
jitter are also suppressed. A first escalation, materially closer distance,
reliable closing trend or STOP still speaks.

The default prompted YOLOE vocabulary adds `book` and `pillow`. Startup prints
the effective loaded labels. Neither name is added to the static hazard classes;
names can describe/match an object without changing the ToF warning decision.

## Focused regression checks

```powershell
cd C:\Users\manoj\WhisperingHat
uv run --no-project --python 3.12 --with pytest --with numpy --with 'opencv-python<5' python -m pytest -q tests/test_voice_capture.py tests/test_voice_watchdog.py tests/test_voice_commands.py tests/test_voice_vision.py tests/test_pi_regressions.py tests/test_alert_suppression.py tests/test_scene_description.py tests/test_optimizations.py
```

Tests use real subprocess pipes with a deliberately stuck capture owner, plus
mocked recognizer/model boundaries. They check bounded teardown, no early reopen,
replacement decoding/dispatch, shutdown during recovery, silent PCM, echo-tail
resumption, request repetition after failures, model restart, clipped-page
rejection, stationary visual jitter, sustained depth approach and urgent warnings.
Real microphone/camera/ToF behavior still needs the stationary Pi smoke test.
