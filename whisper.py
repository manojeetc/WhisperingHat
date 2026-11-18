import os

# Keep CPU thread usage reasonable on Raspberry Pi.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import csv
import json
import re
import sys
import cv2
import time
import math
import subprocess
import shutil
import tempfile
import threading
import queue
import pickle
import numpy as np
from pathlib import Path
from contextlib import contextmanager

# Optional USB microphone voice control.
# Microphone capture uses SpeechRecognition/PyAudio, while speech-to-text uses
# Vosk entirely on-device (no Internet connection or cloud API required).
try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from vosk import Model as VoskModel, KaldiRecognizer
except ImportError:
    VoskModel = None
    KaldiRecognizer = None
from collections import defaultdict, deque

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from ultralytics import YOLO, YOLOE
except ImportError:
    from ultralytics import YOLO
    YOLOE = None

try:
    import face_recognition
except ImportError:
    face_recognition = None

# SparkFun Qwiic VL53L5CX driver. The program remains camera-only if missing.
try:
    import qwiic_vl53l5cx
except ImportError:
    qwiic_vl53l5cx = None

# LiteRT/TFLite interpreter fallback chain.
# On Raspberry Pi OS, use whichever package is available.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            Interpreter = None


# ============================================================
# WHISPERING HAT - PI 5 + CAMERA MODULE 3 WIDE + VL53L5CX TOF
# ============================================================
#
# What this version changes from your V4:
#   1. Adds Raspberry Pi Camera Module support through Picamera2.
#   2. Keeps USB webcam / video-file support through OpenCV.
#   3. Adds the walking corridor back as a trapezoid, not just a flat center box.
#   4. Makes objects inside the walking path more important than side objects.
#   5. Uses tracking so each detected object gets separate alert and distance memory.
#   6. Keeps calibrated fake distance and smoothing.
#   7. Adds optional reference-height distance, but leaves it OFF by default.
#   8. Keeps face identity and emotion inference on independent schedules.
#
# Important:
#   A single camera still cannot know true distance reliably.
#   For a wearable, use sonar / ToF later as the real distance source.
#
# ============================================================


# ============================================================
# SOURCE SETTINGS - RASPBERRY PI 5 + RASPBERRY PI OS TRIXIE
#                  + OFFICIAL PI CAMERA MODULE 3
# ============================================================

# For Pi 5 + Camera Module 3, use Picamera2/libcamera directly.
# Do not use the old PiCamera package or the legacy camera stack.
CAMERA_BACKEND = "picamera2"

# Camera Module 3 on Pi 5 is usually camera 0.
# If you attach two cameras, change this to 1 for the second camera.
PICAMERA2_CAMERA_INDEX = 0

# Used only if you change CAMERA_BACKEND to "opencv" for a USB webcam/video file.
OPENCV_SOURCE = 0

# Good starter resolution for Pi 5 CPU YOLO.
# 640x480 gives a stable 4:3 image and keeps CPU load reasonable.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 20

# Camera Module 3 autofocus.
# Options: "continuous", "auto", "manual", "off"
# For a wearable, continuous is usually best.
CAMERA3_AF_MODE = "continuous"

# Used only when CAMERA3_AF_MODE = "manual".
# LensPosition is in diopters: 0.0 is infinity, larger values focus closer.
# Try 1.0 to 3.0 for hallway/walking distance, 6.0+ for close objects.
CAMERA3_MANUAL_LENS_POSITION = 2.0

# Flip only if your camera is mounted upside down or mirrored.
CAMERA_HFLIP = False
CAMERA_VFLIP = True

# In SSH/headless mode, leave this False.
# On the Pi desktop or through VNC, change it to True to see the preview window.
SHOW_WINDOW = True

# Pi 5 performance settings.
# The baseline profile stays conservative. Heavier models are opt-in only.
PROCESS_EVERY_N_FRAMES = 3

# Capture continuously in a background thread and always process the newest frame.
# This prevents YOLO from building up a queue of stale camera frames.
USE_LATEST_FRAME_THREAD = True
LATEST_FRAME_WAIT_SECONDS = 1.5

CONFIDENCE = 0.40
DEVICE = "cpu"

# Detection model profiles. The default is the existing YOLO11n baseline;
# YOLO11s/YOLO11m stay behind an explicit environment flag so a Pi can always
# be rolled back to the model that was benchmarked as the working baseline.
#
#   WHISPER_HAT_MODEL_PROFILE=baseline       yolo11n.pt @ 320 (default)
#   WHISPER_HAT_MODEL_PROFILE=accuracy       yolo11s.pt @ 384 (opt-in)
#   WHISPER_HAT_MODEL_PROFILE=max_accuracy   yolo11m.pt @ 416 (opt-in)
#
# A specific checkpoint / size can still be overridden for benchmarking with
# WHISPER_HAT_MODEL and WHISPER_HAT_IMG_SIZE.
MODEL_PROFILES = {
    "baseline": {"model": "yolo11n.pt", "img_size": 320},
    "accuracy": {"model": "yolo11s.pt", "img_size": 384},
    "max_accuracy": {"model": "yolo11m.pt", "img_size": 416},
}
MODEL_PROFILE = os.environ.get(
    "WHISPER_HAT_MODEL_PROFILE", "baseline"
).strip().lower()
if MODEL_PROFILE not in MODEL_PROFILES:
    print(
        f"[model] Unknown WHISPER_HAT_MODEL_PROFILE={MODEL_PROFILE!r}; "
        "using the baseline YOLO11n profile."
    )
    MODEL_PROFILE = "baseline"
MODEL_NAME = os.environ.get(
    "WHISPER_HAT_MODEL", MODEL_PROFILES[MODEL_PROFILE]["model"]
)
IMG_SIZE = int(
    os.environ.get(
        "WHISPER_HAT_IMG_SIZE", MODEL_PROFILES[MODEL_PROFILE]["img_size"]
    )
)

# Opt-in open-vocabulary detector for everyday obstacles the COCO baseline
# cannot name. The baseline remains untouched unless this is explicitly on.
#
#   WHISPER_HAT_OPEN_VOCAB=1
#   WHISPER_HAT_OPEN_VOCAB_MODEL=yoloe-11s-seg.pt
#   WHISPER_HAT_OPEN_VOCAB_PROMPT_FREE=0
#
# For a Pi deployment, export the model once with these classes baked in and
# point OPEN_VOCAB_MODEL at the exported file; otherwise the first run needs
# the CLIP text encoder (network access) to build the text embeddings.
OPEN_VOCAB_ENABLED = os.environ.get(
    "WHISPER_HAT_OPEN_VOCAB", "0"
).strip().lower() in {"1", "true", "yes", "on"}
OPEN_VOCAB_MODEL = os.environ.get(
    "WHISPER_HAT_OPEN_VOCAB_MODEL", "yoloe-11s-seg.pt"
).strip()
OPEN_VOCAB_PROMPT_FREE = os.environ.get(
    "WHISPER_HAT_OPEN_VOCAB_PROMPT_FREE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
OPEN_VOCAB_IMG_SIZE = int(
    os.environ.get("WHISPER_HAT_OPEN_VOCAB_IMG_SIZE", str(IMG_SIZE))
)
OPEN_VOCAB_CLASSES = tuple(
    name.strip()
    for name in os.environ.get(
        "WHISPER_HAT_OPEN_VOCAB_CLASSES",
        "chair,stool,bench,cardboard box,box,package,bag,backpack,handbag,"
        "suitcase,trash can,bin,basket,laundry basket,shopping cart,cart,"
        "shoe,shoes,table,desk,door,bucket,cooler,crate,traffic cone,"
        "stroller,walker,vacuum cleaner,fan,heater,plant,pet bowl,cable,toy",
    ).split(",")
    if name.strip()
)

# Tracking keeps separate memory for different people and objects.
# It costs some extra CPU, but prevents objects in the same zone from sharing state.
USE_TRACKING = True
TRACKER_NAME = "bytetrack.yaml"

cv2.setNumThreads(1)

# Load the fallback Haar face detector once instead of rebuilding it repeatedly.
# Some Raspberry Pi OpenCV builds do not provide cv2.data, so try several
# common locations safely.
def find_haar_cascade_path():
    candidates = []

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )

    candidates.extend([
        Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
        Path("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
        Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


HAAR_CASCADE_PATH = find_haar_cascade_path()

if HAAR_CASCADE_PATH is not None:
    HAAR_FACE_CASCADE = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    print(f"Loaded Haar face cascade: {HAAR_CASCADE_PATH}")
else:
    HAAR_FACE_CASCADE = cv2.CascadeClassifier()
    print("Haar face cascade not found. Haar fallback face detection is disabled.")


# ============================================================
# WALKING CORRIDOR SETTINGS
# ============================================================
#
# The corridor is a trapezoid/funnel because the path ahead appears narrower
# near the horizon and wider near the bottom of the image.
#
# Use the object's bottom-center point, not its box center, because the bottom
# of the object is closer to where it touches the ground.

USE_WALKING_CORRIDOR = True

# Top of the corridor as a fraction of image height.
# 0.30 means the corridor starts 30% down from the top.
CORRIDOR_TOP_Y = 0.32

# Width near the top/horizon.
CORRIDOR_TOP_LEFT = 0.43
CORRIDOR_TOP_RIGHT = 0.57

# Width near your feet / bottom of frame.
CORRIDOR_BOTTOM_LEFT = 0.18
CORRIDOR_BOTTOM_RIGHT = 0.82

# If an object is very low in the frame, it may still matter even near the side.
VERY_LOW_OBJECT_Y = 0.78

DRAW_CORRIDOR = True


# ============================================================
# DISTANCE SETTINGS
# ============================================================
#
# Best practical method for now:
#   calibrated_height
#
# Why not reference sizes by default?
#   Because YOLO box height depends on whether the whole object is visible,
#   camera angle, crop, lens distortion, pose, and object type.
#
# Calibration is easier:
#   1. Put a person/object at a known distance, like 1.2 m.
#   2. Read the on-screen h=___px label.
#   3. Update known_box_height_px below.

DISTANCE_MODE = "calibrated_height"
# Options:
#   "calibrated_height" = recommended
#   "reference_height"  = optional rough formula; needs focal length calibration
#   "area_fallback"     = roughest fallback

DISTANCE_CALIBRATION = {
    "person": {
        "known_distance_m": 1.2,
        "known_box_height_px": 280,
    },
    "chair": {
        "known_distance_m": 1.2,
        "known_box_height_px": 220,
    },
    "bench": {
        "known_distance_m": 1.2,
        "known_box_height_px": 220,
    },
    "backpack": {
        "known_distance_m": 1.0,
        "known_box_height_px": 180,
    },
    "handbag": {
        "known_distance_m": 1.0,
        "known_box_height_px": 160,
    },
    "suitcase": {
        "known_distance_m": 1.0,
        "known_box_height_px": 220,
    },
}

# Optional rough reference heights in meters.
# Keep this OFF unless you calibrate FOCAL_LENGTH_PX for your exact camera mode.
REFERENCE_HEIGHT_M = {
    "person": 1.68,      # rough U.S. adult average across male/female averages
    "chair": 0.90,      # very rough full chair height
    "bench": 0.80,      # very rough full bench height
    "backpack": 0.45,   # very variable
    "handbag": 0.35,    # very variable
    "suitcase": 0.65,   # carry-on-ish, very variable
    "dog": 0.45,        # very variable
    "cat": 0.25,        # very variable
}

# Focal length in pixels for reference-height distance.
# Better: set this by calibration using a full visible person/object.
# Example formula:
#   focal_px = box_height_px_at_known_distance * known_distance_m / real_height_m
FOCAL_LENGTH_PX = None

MIN_FAKE_DISTANCE_M = 0.25
MAX_FAKE_DISTANCE_M = 8.0

# Lower alpha = smoother but slower to react.
# Higher alpha = faster but jumpier.
DISTANCE_SMOOTH_ALPHA = 0.25

VERY_CLOSE_M = 0.65
CLOSE_M = 1.25
MEDIUM_M = 2.50

# For people, normal talking distance should not become endless STOP.
PERSON_STOP_DISTANCE_M = 0.55

APPROACHING_MPS = 0.25
FAST_APPROACHING_MPS = 0.70

AREA_GROWTH_APPROACHING = 0.22
AREA_GROWTH_FAST = 0.45


# ============================================================
# VL53L5CX QWIIC TOF SETTINGS
# ============================================================
#
# The ToF board can be connected and tested immediately. However, camera/ToF
# fusion is intentionally disabled until both boards are rigidly mounted.
# Once mounted together and pointing in the same direction, change:
#     TOF_MOUNT_CALIBRATED = True
# Then tune the overlap offsets while viewing the preview.

ENABLE_TOF = True
TOF_REQUIRED = False
TOF_MOUNT_CALIBRATED = True

# 8x8 gives 64 depth zones. The VL53L5CX supports at most 15 Hz in 8x8 mode.
TOF_RESOLUTION = 64
TOF_FREQUENCY_HZ = 12
TOF_RECONNECT_SECONDS = 3.0
TOF_STALE_SECONDS = 0.75
TOF_PRINT_SUMMARY = True
TOF_PRINT_INTERVAL_SECONDS = 1.0

# Camera Module 3 Wide is about 102 degrees horizontal and 67 degrees vertical.
# The VL53L5CX sensing area is about 45 x 45 degrees. With parallel optical
# axes, this is roughly the central 33.5% of image width and 62.5% of height.
# These are starting values only; actual Picamera2 sensor mode/cropping matters.
TOF_OVERLAP_WIDTH_FRACTION = 0.335
TOF_OVERLAP_HEIGHT_FRACTION = 0.625
TOF_OVERLAP_CENTER_X_OFFSET = 0.0
TOF_OVERLAP_CENTER_Y_OFFSET = 0.0

# Change these if the displayed depth grid is mirrored/upside down after mount.
# SparkFun's example reverses columns when printing, so X flip is a useful
# starting point, but the correct value depends on board orientation.
TOF_FLIP_X = True
TOF_FLIP_Y = False

# Use only plausible, recent measurements. Status 5 and 9 are commonly treated
# as usable VL53L5CX ranges. If the Python driver does not expose status, the
# distance is still range-checked.
TOF_VALID_STATUS_CODES = {5, 9}
TOF_MIN_DISTANCE_MM = 50
TOF_MAX_DISTANCE_MM = 4000
TOF_MIN_VALID_CELLS = 1
TOF_DISTANCE_PERCENTILE = 25.0

# Sample the middle/lower part of a YOLO box. This reduces background depth
# leaking into narrow object boxes while still catching a person's torso.
TOF_BOX_X_INSET_FRACTION = 0.18
TOF_BOX_Y_START_FRACTION = 0.22
TOF_CELL_PADDING = 0

# A close/low object can sit just below the calibrated ToF overlap rectangle
# while still being visible to the camera. Clamp only the vertical sample to
# the bottom ToF row so the camera detection can still claim that depth cell.
TOF_BOX_CLAMP_LOW_OBJECTS = os.environ.get(
    "WHISPER_HAT_TOF_CLAMP_LOW", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Controlled-test logging: one line per class per interval showing the camera
# box, corridor decision, ToF cells, and why a match was or was not made.
TOF_ASSOCIATION_DEBUG = os.environ.get(
    "WHISPER_HAT_ASSOCIATION_DEBUG", "0"
).strip().lower() in {"1", "true", "yes", "on"}
TOF_ASSOCIATION_DEBUG_INTERVAL_SECONDS = 1.0

# Unknown-obstacle warnings are powerful but should only be enabled after the
# ToF is rigidly mounted and its central path region is verified.
TOF_UNKNOWN_OBSTACLE_ENABLED = True
TOF_UNKNOWN_PATH_COL_START = 2
TOF_UNKNOWN_PATH_COL_END = 5
TOF_UNKNOWN_PATH_ROW_START = 2
TOF_UNKNOWN_PATH_ROW_END = 7
TOF_UNKNOWN_NOTICE_M = 2.5
TOF_UNKNOWN_CAUTION_M = 1.25
TOF_UNKNOWN_STOP_M = 0.55
TOF_UNKNOWN_MATCH_TOLERANCE_M = 0.45

# Robustness for the generic "something is in the path" warning. A single
# spurious close cell used to dominate the 25th percentile, producing 0.05 m
# then 2.4 m for the same stationary obstacle. Use the median and require a
# couple of valid cells, then smooth/hysteresis the band transitions.
TOF_UNKNOWN_PERCENTILE = 50.0
TOF_UNKNOWN_MIN_VALID_CELLS = 2
TOF_UNKNOWN_SMOOTH_ALPHA = 0.35
TOF_UNKNOWN_HYSTERESIS_M = 0.12
TOF_UNKNOWN_CLOSER_BYPASS_M = 0.40

# Deduplication for the generic warning. It has one stable alert key, so a
# band change is not a new hazard. A materially closer reading or a new
# appearance after the path was clear still speaks immediately.
TOF_UNKNOWN_MIN_SPEECH_GAP_SECONDS = 5.0
TOF_UNKNOWN_ESCALATION_GAP_SECONDS = 1.5
TOF_UNKNOWN_REPEAT_SECONDS = 30.0
TOF_UNKNOWN_SPEAK_ON_REASON_CHANGE = False
TOF_UNKNOWN_DEBUG = os.environ.get("WHISPER_HAT_UNKNOWN_DEBUG", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
TOF_UNKNOWN_DEBUG_INTERVAL_SECONDS = 2.0

# Time-to-collision thresholds used only with real ToF distance.
TOF_TTC_CAUTION_SECONDS = 2.0
TOF_TTC_STOP_SECONDS = 0.9

# ============================================================
# LOGGING SETTINGS
# ============================================================

LOG_CSV = True
LOG_FILE = str(SCRIPT_DIR / "whisper_hat_tof_log.csv")
LOG_ALL_DETECTIONS = False


# ============================================================
# SOUND SETTINGS
# ============================================================

# Options:
#   "off"      = no sound
#   "bell"     = terminal bell, may or may not make sound over SSH
#   "espeak"   = speak alert text if espeak is installed
#   "auto"     = Windows beep; on Linux terminal bell
ALERT_SOUND_MODE = "espeak"

# USB/PipeWire speech settings.
SPEECH_VOICE = "en-us"
SPEECH_RATE = 145
SPEAK_STARTUP_MESSAGE = True

# PipeWire normally lives in /run/user/<uid>. Explicitly passing this to
# pw-play fixes silent playback when Python is started from SSH, VNC, or an IDE.
AUDIO_RUNTIME_DIR = (
    f"/run/user/{os.getuid()}"
    if hasattr(os, "getuid")
    else os.environ.get("XDG_RUNTIME_DIR", "")
)
PRINT_SPEECH_DEBUG = True

# USB microphone voice-control settings.
# Exact command phrases:
#   "detection off" = pause YOLO, face/emotion, and ToF hazard alerts
#   "detection on"  = resume detection
ENABLE_VOICE_CONTROL = True
VOICE_MIC_DEVICE_INDEX = 0      # Tested USB Microphone: Audio (hw:2,0)
VOICE_MIC_SAMPLE_RATE = 44100   # Hardware capture rate that passed your mic test.
VOICE_RECOGNITION_SAMPLE_RATE = 16000  # Vosk receives mono 16-bit PCM at 16 kHz.
VOICE_LISTEN_TIMEOUT = 1.0
VOICE_PHRASE_TIME_LIMIT = 4.0
VOICE_AMBIENT_CALIBRATION_SECONDS = 2.0
# Download/extract a small English Vosk model into this folder once. After that,
# recognition is fully offline. Example folder: vosk-model-small-en-us-0.15
VOICE_VOSK_MODEL_PATH = str(SCRIPT_DIR / "vosk-model-small-en-us-0.15")
VOICE_COMMAND_COOLDOWN_SECONDS = 1.0
VOICE_CONFIRM_COMMANDS = True
VOICE_PRINT_DEBUG = True

# Opt-in diagnostics. These print timestamps and never save raw audio.
# Turn this on for one test run when a command is missed; leave it off normally.
VOICE_DEBUG_TIMING = os.environ.get("WHISPER_HAT_VOICE_DEBUG", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

# If a USB microphone rejects the configured capture rate, try these rates in
# order before giving up. The first working rate is logged when debug is on.
VOICE_MIC_FALLBACK_RATES = (16000, 48000)

# Post-TTS guard. 0.0 keeps the baseline behavior (listening continues while
# speech is queued/played). Set this to a small value such as 0.35 only if the
# test matrix shows the speaker is triggering false/ignored recognitions.
VOICE_PLAYBACK_COOLDOWN_SECONDS = float(
    os.environ.get("WHISPER_HAT_VOICE_COOLDOWN", "0") or 0.0
)

# Listener watchdog. This recovers only when the audio stream stops making
# progress or the worker thread dies, never just because the user is silent.
# The normal listen timeout is 1 s and the phrase limit is 4 s; 12 s without a
# heartbeat means a blocked stream read or a stuck native call.
VOICE_WATCHDOG_ENABLED = os.environ.get(
    "WHISPER_HAT_VOICE_WATCHDOG", "1"
).strip().lower() in {"1", "true", "yes", "on"}
VOICE_WATCHDOG_INTERVAL_SECONDS = 1.0
VOICE_WATCHDOG_STALE_SECONDS = 12.0
VOICE_RECOVERY_MIN_INTERVAL_SECONDS = 30.0
VOICE_STALE_PLAYBACK_SECONDS = 25.0

# Low-volume status line, disabled by default. It also appears when
# VOICE_DEBUG_TIMING is on. Counts are cumulative for the session.
VOICE_STATUS_LOG = os.environ.get(
    "WHISPER_HAT_VOICE_STATUS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
VOICE_STATUS_LOG_SECONDS = 30.0

# Command aliases. The grammar includes only enabled aliases; normalization
# removes punctuation and apostrophes, so "what's" and "whats" are equivalent.
VOICE_DETECTION_ON_ALIASES = (
    "turn detection on",
    "start detection",
)
VOICE_DETECTION_OFF_ALIASES = (
    "turn detection off",
    "stop detection",
)
OCR_READ_ALIASES = (
    "read it",
    "read this page",
    "read the text",
    "read that text",
    "what does this say",
    "what does that say",
    "read aloud",
)
SCENE_DESCRIBE_ALIASES = (
    "what is there",
    "what can you see",
    "describe the scene",
    "what is ahead",
)

# Requires: sudo apt install -y tesseract-ocr
ENABLE_OCR_READING = True
OCR_BINARY = "tesseract"
OCR_LANGUAGE = "eng"
OCR_PSM = 3                      # 3 = fully automatic page segmentation
OCR_TIMEOUT_SECONDS = 20
OCR_MAX_SPEECH_CHARS = 600       # cap spoken text at a whole-word boundary
OCR_TRIGGER_COOLDOWN_SECONDS = 2.0

# OCR quality gate. Tesseract confidence alone is not enough: the Pi log
# returned 996 characters of gibberish with many plausible-looking tokens.
# Accept only when confidence and language-likeness agree; otherwise speak one
# short retry instruction instead of reading garbage.
OCR_QUALITY_CHECK = os.environ.get(
    "WHISPER_HAT_OCR_QUALITY", "1"
).strip().lower() in {"1", "true", "yes", "on"}
OCR_QUALITY_GOOD_CONF = 60.0
OCR_QUALITY_MIN_MEAN_CONF = 55.0
OCR_QUALITY_MIN_GOOD_WORD_RATIO = 0.50
OCR_QUALITY_MIN_ALPHA_RATIO = 0.55
OCR_QUALITY_MIN_VOWEL_WORD_RATIO = 0.55
OCR_QUALITY_MAX_LONG_CONSONANT_RATIO = 0.25
OCR_QUALITY_MIN_COMMON_WORD_RATIO = 0.12
OCR_QUALITY_COMMON_WORD_MIN_COUNT = 8
OCR_QUALITY_MIN_WORDS_FOR_CONFIDENCE = 2
OCR_QUALITY_MIN_SINGLE_WORD_CONF = 80.0

# Speak the first chunk quickly instead of one 600-character block. Three
# chunks of about 180 characters keep any single pw-play call below the 20 s
# timeout and let a genuinely urgent STOP jump the queue sooner.
OCR_SPEECH_CHUNK_CHARS = 180
OCR_MAX_SPEECH_CHUNKS = 3

# Capture a still only for "read this". The detection stream stays at
# CAMERA_WIDTH x CAMERA_HEIGHT. The still is downscaled to OCR_MAX_WIDTH before
# Tesseract so higher detail does not multiply OCR time without bound.
OCR_USE_STILL = os.environ.get(
    "WHISPER_HAT_OCR_STILL", "0"
).strip().lower() in {"1", "true", "yes", "on"}
OCR_STILL_WIDTH = int(os.environ.get("WHISPER_HAT_OCR_STILL_WIDTH", "1920"))
OCR_STILL_HEIGHT = int(os.environ.get("WHISPER_HAT_OCR_STILL_HEIGHT", "1440"))
OCR_MAX_WIDTH = int(os.environ.get("WHISPER_HAT_OCR_MAX_WIDTH", "1024"))
OCR_STILL_CAPTURE_WAIT_SECONDS = 8.0
OCR_READ_PHRASES = (
    "read this",
    "read that",
    "read",
    "read text",
    "what does it say",
    "what's that say",
)

ENABLE_SCENE_DESCRIBE = True
SCENE_DESCRIBE_MAX_OBJECTS = 3
SCENE_DESCRIBE_COOLDOWN_SECONDS = 2.0
SCENE_DESCRIBE_PHRASES = (
    "what's there",
    "what do you see",
    "describe",
    "what's around me",
    "what's ahead",
)


# ============================================================
# FACE RECOGNITION SETTINGS
# ============================================================

# Face recognition runs locally on the Pi.
# Put consented reference photos in:
#   known_faces/Alex/front.jpg
#   known_faces/Alex/left.jpg
#   known_faces/Mom/1.jpg
ENABLE_FACE_RECOGNITION = True

KNOWN_FACES_DIR = str(SCRIPT_DIR / "known_faces")
FACE_CACHE_FILE = str(SCRIPT_DIR / "known_faces_cache.pkl")

# Set True after adding/changing known face photos, run once, then set back to False.
REBUILD_FACE_CACHE = False

# On Raspberry Pi CPU, keep this slow/occasional.
# Your YOLO already runs every PROCESS_EVERY_N_FRAMES, and this runs even less often.
FACE_RECOGNITION_EVERY_N_FRAMES = 9

# Lower = stricter. 0.50 is a good safety-minded starting point.
FACE_MATCH_TOLERANCE = 0.50

# Use HOG on Raspberry Pi CPU. CNN is much heavier.
FACE_DETECTOR_MODEL = "hog"

# 1 is fastest. Higher can improve accuracy but slows the Pi.
FACE_NUM_JITTERS = 1

# Do not try face recognition on tiny/far-away person boxes.
FACE_MIN_PERSON_BOX_HEIGHT = 70

# Padding around the YOLO person box before searching for a face.
FACE_CROP_PAD_FRAC = 0.08


# ============================================================
# EMOTION RECOGNITION SETTINGS
# ============================================================

# Uses the bundled FER TFLite model. The interpreter reads its actual
# input shape automatically. The included model uses 64x64 grayscale faces.
ENABLE_EMOTION_RECOGNITION = True
EMOTION_MODEL_FILE = str(SCRIPT_DIR / "emotion_model.tflite")

# The included FER model was trained with pixel values scaled to [-1, 1].
# Other common choices are "zero_to_one" and "raw".
EMOTION_INPUT_NORMALIZATION = "minus_one_to_one"

# Standard FER2013 output order. Change this only if your model uses another order.
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# Emotion inference is relatively cheap, but do not run it on every camera frame.
EMOTION_EVERY_N_FRAMES = 6
EMOTION_MIN_CONFIDENCE = 0.25

# Useful when SHOW_WINDOW=False or the program is launched through SSH.
PRINT_EMOTION_TO_TERMINAL = True
EMOTION_PRINT_COOLDOWN_SECONDS = 1.5

# Speak stable facial-expression estimates as low-priority information.
# Safety alerts always receive a higher speech-queue priority.
SPEAK_DETECTED_EMOTIONS = True
EMOTION_SPEAK_MIN_CONFIDENCE = 0.45
EMOTION_SPEAK_STABLE_READINGS = 2
EMOTION_SPEAK_COOLDOWN_SECONDS = 15.0
EMOTION_SPEAK_CHANGE_GAP_SECONDS = 4.0
EMOTION_SPEAK_NEUTRAL = False
EMOTION_SPEAK_MAX_DISTANCE_M = 3.0

# Smooth several predictions so the label does not flicker.
EMOTION_HISTORY_LENGTH = 5


# ============================================================
# YOLO CLASS FILTER
# ============================================================
# COCO IDs:
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck,
# 9 traffic light, 11 stop sign, 13 bench, 15 cat, 16 dog,
# 17 horse, 19 cow, 24 backpack, 26 handbag, 28 suitcase, 56 chair

YOLO_CLASSES_TO_KEEP = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
    11,  # stop sign
    13,  # bench
    15,  # cat
    16,  # dog
    17,  # horse
    19,  # cow
    24,  # backpack
    26,  # handbag
    28,  # suitcase
    56,  # chair
]

# The COCO ID filter is replaced by None for a broad-vocabulary model whose
# class IDs are not COCO IDs. load_detection_model() sets these.
DETECTION_CLASSES = YOLO_CLASSES_TO_KEEP
DETECTION_IMG_SIZE = IMG_SIZE


# ============================================================
# ALERT / MEMORY SETTINGS
# ============================================================

# Speak risk 1 notices as well as risk 2 and 3 warnings.
# Set this back to 2 later if ordinary notices are too talkative.
MIN_SPOKEN_RISK = 1

# Short stabilization delays prevent one-frame false detections while remaining responsive.
STABLE_SECONDS_FOR_NOTICE = 0.30
STABLE_SECONDS_FOR_CAUTION = 0.25
STABLE_SECONDS_FOR_STOP = 0.10

# Balanced alert-history behavior:
# Speak immediately when a hazard first appears, speak again if the risk rises,
# and give occasional reminders while the hazard remains present.
REPEAT_ALERTS_WHILE_STILL_PRESENT = True
SPEAK_ON_SAME_RISK_REASON_CHANGE = True

# Reminder timing for an unchanged ongoing hazard.
# Notices are infrequent, cautions are more frequent, and STOP repeats fastest.
NOTICE_REPEAT_SECONDS = 20.0
CAUTION_REPEAT_SECONDS = 8.0
STOP_REPEAT_SECONDS = 4.0

GLOBAL_NOTICE_GAP_SECONDS = 2.0
GLOBAL_CAUTION_GAP_SECONDS = 1.2
GLOBAL_STOP_GAP_SECONDS = 0.5

# A class/zone must disappear this long before its alert history is forgotten.
ALERT_REARM_AFTER_ABSENT_SECONDS = 4.0

# A queued routine repeat is rechecked at dequeue time. If the hazard has not
# been refreshed recently, or has dropped below the queued risk, skip it.
HAZARD_REPEAT_PRESENT_MAX_AGE_SECONDS = 1.5

# Distance and tracking histories can expire sooner without forgetting that an
# alert was already spoken.
TRACK_STALE_SECONDS = 3.0

LOWER_FRAME_IMPORTANCE = 0.55

# Print the current best risk once per second while testing.
PRINT_ALERT_DEBUG = True
ALERT_DEBUG_INTERVAL_SECONDS = 1.0

# Opt-in end-to-end timing. Logs command/hazard origin to the moment TTS starts.
# Leave off for normal runs; turn on during the Pi measurement session.
PRINT_LATENCY_DEBUG = os.environ.get("WHISPER_HAT_LATENCY_DEBUG", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

# Alert memory uses a stable class-and-zone key. Distance/motion memory still
# uses tracking IDs, so a temporary ByteTrack ID change does not silence speech.
USE_TRACK_ID_FOR_ALERT_KEY = False


# ============================================================
# OBJECT CLASSES BY NAME
# ============================================================

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
PERSON_CLASSES = {"person"}
ANIMAL_CLASSES = {"dog", "cat", "horse", "cow"}
STATIC_OBSTACLE_CLASSES = {
    "bench", "chair", "stool", "box", "cardboard box", "package", "crate",
    "bag", "backpack", "handbag", "suitcase", "trash can", "trashcan", "bin",
    "garbage can", "basket", "laundry basket", "shopping cart", "cart",
    "shoe", "shoes", "table", "desk", "door", "bucket", "cooler",
    "traffic cone", "stroller", "walker", "vacuum cleaner", "fan", "heater",
    "plant", "pet bowl", "cable", "toy", "luggage",
}
PASSIVE_SIGN_CLASSES = {"stop sign", "traffic light"}
KNOWN_OBJECT_CLASSES = (
    STATIC_OBSTACLE_CLASSES | PERSON_CLASSES | VEHICLE_CLASSES | ANIMAL_CLASSES
)


# ============================================================
# STATE
# ============================================================

track_history = defaultdict(lambda: deque(maxlen=8))
track_memory = {}
distance_memory = {}
last_global_alert_time = 0
last_alert_debug_time = 0.0
last_cleanup_time = 0.0
CLEANUP_INTERVAL_SECONDS = 0.5

_corridor_overlay_cache = {}
_log_file_handle = None
_log_writer = None
_speech_program_path = None
_audio_player_path = None
_speech_paths_checked = False
_speech_env_cache = None
_speech_missing_warned = False

# A single worker speaks queued messages. A priority queue lets STOP and
# Caution messages move ahead of informational emotion announcements.
speech_queue = queue.PriorityQueue(maxsize=8)
speech_worker_thread = None
speech_sequence = 0
speech_sequence_lock = threading.Lock()

known_face_names = []
known_face_encodings = []
known_face_encodings_matrix = None
face_memory = {}
emotion_recognizer = None
emotion_history = defaultdict(lambda: deque(maxlen=EMOTION_HISTORY_LENGTH))
last_emotion_print = {}
emotion_speech_memory = {}
state_lock = threading.RLock()


# ToF worker/state. Access to the latest grid is protected by a lock.
tof_worker = None
tof_state_lock = threading.Lock()
tof_state = {
    "connected": False,
    "initialized": False,
    "grid_mm": None,
    "status_grid": None,
    "last_update": 0.0,
    "last_error": None,
    "frame_counter": 0,
}
last_tof_summary_time = 0.0

# Current audio process is tracked so an urgent STOP can interrupt speech.
current_audio_process = None
current_audio_lock = threading.Lock()
current_audio_protected = False

# TTS activity is shared with the voice worker so debug logs can distinguish
# acoustic self-speech from a stopped/dropped recognition path.
speech_active_event = threading.Event()
speech_active_message = None
speech_last_start_time = 0.0
speech_last_end_time = 0.0

# Set by should_alert() for the current decision; the main loop uses it to
# decide whether a new/escalated hazard should interrupt a command response.
last_alert_decision = {}

# Voice-control state. Detection starts ON and can be toggled while the
# camera/program keeps running. threading.Event is safe to share across threads.
detection_enabled_event = threading.Event()
detection_enabled_event.set()
voice_control_stop_event = threading.Event()
voice_control_thread = None
voice_control_lifecycle_lock = threading.Lock()
voice_watchdog_thread = None
voice_listener_source = None
voice_listener_microphone = None
voice_listener_source_lock = threading.Lock()
voice_listener_heartbeat = 0.0
voice_listener_stage = "stopped"
voice_last_audio_monotonic = 0.0
voice_last_processed_monotonic = 0.0
voice_audio_frames = 0
voice_audio_bytes = 0
voice_audio_timeouts = 0
voice_nonempty_results = 0
voice_commands_matched = 0
voice_commands_dispatched = 0
voice_direct_responses_started = 0
voice_ocr_triggers = 0
voice_scene_triggers = 0
voice_recovery_count = 0
voice_last_recovery_monotonic = 0.0
voice_last_status_log = 0.0
speech_drop_count = 0
speech_cleared_count = 0
last_voice_command_time = 0.0
scene_describe_event = threading.Event()
last_scene_describe_time = 0.0
scene_last_trigger_monotonic = 0.0

# Voice-activated OCR reader state. A separate daemon waits for a trigger event,
# grabs one clean frame, runs Tesseract, and speaks the result. The subprocess is
# tracked so shutdown can kill a read that is still running.
ocr_trigger_event = threading.Event()
ocr_stop_event = threading.Event()
ocr_worker_thread = None
ocr_camera_source = None
ocr_frame_origin = "none"
last_ocr_trigger_time = 0.0
ocr_last_trigger_monotonic = 0.0
current_ocr_process = None
current_ocr_process_lock = threading.Lock()
_ocr_available_cache = None
_ocr_missing_warned = False

# Generic "unknown obstacle" temporal state. Kept separate from the raw ToF
# snapshot so a stationary obstacle does not jump between risk bands.
unknown_obstacle_state = {
    "distance_m": None,
    "last_update": 0.0,
    "risk": 0,
}
last_unknown_debug_time = 0.0

# Debug timing epoch. _voice_debug() is opt-in via VOICE_DEBUG_TIMING.
_DEBUG_EPOCH = time.monotonic()


# ============================================================
# BASIC HELPERS
# ============================================================

def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _voice_debug(stage, **fields):
    """Opt-in, timestamped stage log. No raw audio is written anywhere."""
    if not VOICE_DEBUG_TIMING:
        return
    elapsed = time.monotonic() - _DEBUG_EPOCH
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(
        f"[voice debug {now_str()}] +{elapsed:.3f}s {stage}{suffix}",
        flush=True,
    )


def _set_speech_active(active, message=None):
    global speech_active_message, speech_last_start_time, speech_last_end_time
    if active:
        speech_last_start_time = time.time()
        speech_active_message = message
        speech_active_event.set()
        _voice_debug("playback_suppression_start", message=repr(message))
    else:
        speech_last_end_time = time.time()
        speech_active_message = None
        speech_active_event.clear()
        _voice_debug("playback_suppression_clear")


def _voice_heartbeat(stage):
    """Record real listener progress for the watchdog/status line."""
    global voice_listener_heartbeat, voice_listener_stage
    voice_listener_heartbeat = time.monotonic()
    voice_listener_stage = stage
    _log_voice_status(voice_listener_heartbeat)


def _set_voice_listener_source(source, microphone=None):
    global voice_listener_source, voice_listener_microphone
    with voice_listener_source_lock:
        voice_listener_source = source
        voice_listener_microphone = microphone


def print_voice_summary():
    """Always print the session counters once, even when periodic logs are off."""
    status = _voice_status_snapshot()
    print(
        "[voice summary] "
        f"audio_frames={status['audio_frames']} "
        f"timeouts={status['timeouts']} "
        f"vosk_nonempty={status['vosk_nonempty']} "
        f"matched={status['matched']} "
        f"dispatched={status['dispatched']} "
        f"answered_aloud={status['answered']} "
        f"ocr_triggers={status['ocr_triggers']} "
        f"scene_triggers={status['scene_triggers']} "
        f"recoveries={status['recoveries']} "
        f"last_audio_age_s={status['last_audio_age_s']} "
        f"last_processed_age_s={status['last_processed_age_s']} "
        f"speech_queue={status['speech_queue']} "
        f"speech_drops={status['speech_drops']} "
        f"speech_cleared={status['speech_cleared']} "
        f"listener={status['thread']} stage={status['stage']}",
        flush=True,
    )


def _voice_status_snapshot():
    now = time.monotonic()
    thread = voice_control_thread
    heartbeat_age = (
        None if voice_listener_heartbeat <= 0.0
        else round(now - voice_listener_heartbeat, 2)
    )
    return {
        "thread": "alive" if thread is not None and thread.is_alive() else "stopped",
        "stage": voice_listener_stage,
        "heartbeat_age_s": heartbeat_age,
        "last_audio_age_s": (
            None if voice_last_audio_monotonic <= 0.0
            else round(now - voice_last_audio_monotonic, 2)
        ),
        "last_processed_age_s": (
            None if voice_last_processed_monotonic <= 0.0
            else round(now - voice_last_processed_monotonic, 2)
        ),
        "audio_frames": voice_audio_frames,
        "audio_bytes": voice_audio_bytes,
        "timeouts": voice_audio_timeouts,
        "vosk_nonempty": voice_nonempty_results,
        "matched": voice_commands_matched,
        "dispatched": voice_commands_dispatched,
        "answered": voice_direct_responses_started,
        "ocr_triggers": voice_ocr_triggers,
        "scene_triggers": voice_scene_triggers,
        "speech_queue": speech_queue.qsize(),
        "speech_drops": speech_drop_count,
        "speech_cleared": speech_cleared_count,
        "tts_active": speech_active_event.is_set(),
        "recoveries": voice_recovery_count,
    }


def _log_voice_status(now=None, force=False):
    """Low-volume session counters; opt-in via debug or status flag."""
    global voice_last_status_log
    if not (VOICE_DEBUG_TIMING or VOICE_STATUS_LOG):
        return
    now = time.monotonic() if now is None else now
    if not force and now - voice_last_status_log < VOICE_STATUS_LOG_SECONDS:
        return
    voice_last_status_log = now
    status = _voice_status_snapshot()
    fields = " ".join(f"{key}={value}" for key, value in status.items())
    print(f"[voice status {now_str()}] {fields}", flush=True)


def _report_latency(origin_perf, label, message=None):
    """Opt-in end-to-end timing from a detected hazard/command to TTS start."""
    if not PRINT_LATENCY_DEBUG or origin_perf is None:
        return
    elapsed_ms = (time.perf_counter() - origin_perf) * 1000.0
    suffix = "" if message is None else f" message={message!r}"
    print(
        f"[latency] {label}_to_tts_start={elapsed_ms:.1f}ms{suffix}",
        flush=True,
    )


def clamp(value, low, high):
    return max(low, min(high, value))


def center_of_box(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_width(x1, x2):
    return max(1, x2 - x1)


def box_height(y1, y2):
    return max(1, y2 - y1)


def box_area(x1, y1, x2, y2):
    return max(1, box_width(x1, x2) * box_height(y1, y2))


def direction_word(cx, frame_width):
    if cx < frame_width * 0.33:
        return "left"
    elif cx > frame_width * 0.66:
        return "right"
    return "ahead"


# ============================================================
# WALKING CORRIDOR HELPERS
# ============================================================

def corridor_bounds_at_y(y, frame_width, frame_height):
    """
    Returns corridor left/right x at a given y.
    The corridor widens from top to bottom.
    """
    top_y_px = CORRIDOR_TOP_Y * frame_height
    bottom_y_px = frame_height

    if y <= top_y_px:
        t = 0.0
    else:
        t = (y - top_y_px) / max(1.0, bottom_y_px - top_y_px)
        t = clamp(t, 0.0, 1.0)

    left_frac = CORRIDOR_TOP_LEFT + t * (CORRIDOR_BOTTOM_LEFT - CORRIDOR_TOP_LEFT)
    right_frac = CORRIDOR_TOP_RIGHT + t * (CORRIDOR_BOTTOM_RIGHT - CORRIDOR_TOP_RIGHT)

    return left_frac * frame_width, right_frac * frame_width


def object_in_walking_corridor(cx, foot_y, frame_width, frame_height):
    if not USE_WALKING_CORRIDOR:
        return True

    left_x, right_x = corridor_bounds_at_y(foot_y, frame_width, frame_height)
    return left_x <= cx <= right_x and foot_y >= CORRIDOR_TOP_Y * frame_height


def corridor_position_word(cx, foot_y, frame_width, frame_height):
    if not USE_WALKING_CORRIDOR:
        return direction_word(cx, frame_width)

    left_x, right_x = corridor_bounds_at_y(foot_y, frame_width, frame_height)

    if left_x <= cx <= right_x and foot_y >= CORRIDOR_TOP_Y * frame_height:
        return "in path"
    if cx < left_x:
        return "left side"
    return "right side"


def corridor_geometry_for_point(cx, foot_y, frame_width, frame_height):
    """Compute both in_corridor and corridor_position in a single pass."""
    if not USE_WALKING_CORRIDOR:
        return True, direction_word(cx, frame_width)

    left_x, right_x = corridor_bounds_at_y(foot_y, frame_width, frame_height)
    if left_x <= cx <= right_x and foot_y >= CORRIDOR_TOP_Y * frame_height:
        return True, "in path"
    if cx < left_x:
        return False, "left side"
    return False, "right side"


def draw_walking_corridor(frame):
    if not (USE_WALKING_CORRIDOR and DRAW_CORRIDOR):
        return

    frame_height, frame_width = frame.shape[:2]
    cache_key = (
        frame_height,
        frame_width,
        CORRIDOR_TOP_Y,
        CORRIDOR_TOP_LEFT,
        CORRIDOR_TOP_RIGHT,
        CORRIDOR_BOTTOM_LEFT,
        CORRIDOR_BOTTOM_RIGHT,
    )
    cached = _corridor_overlay_cache.get("entry")
    if cached is None or cached[0] != cache_key:
        top_y = clamp(int(CORRIDOR_TOP_Y * frame_height), 0, max(0, frame_height - 1))
        bottom_y = max(0, frame_height - 1)

        top_left = (clamp(int(CORRIDOR_TOP_LEFT * frame_width), 0, max(0, frame_width - 1)), top_y)
        top_right = (clamp(int(CORRIDOR_TOP_RIGHT * frame_width), 0, max(0, frame_width - 1)), top_y)
        bottom_left = (clamp(int(CORRIDOR_BOTTOM_LEFT * frame_width), 0, max(0, frame_width - 1)), bottom_y)
        bottom_right = (clamp(int(CORRIDOR_BOTTOM_RIGHT * frame_width), 0, max(0, frame_width - 1)), bottom_y)

        pts = np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)
        roi_x0 = int(np.min(pts[:, 0]))
        roi_x1 = min(frame_width, int(np.max(pts[:, 0])) + 1)
        roi_y0 = int(np.min(pts[:, 1]))
        roi_y1 = min(frame_height, int(np.max(pts[:, 1])) + 1)

        pts_rel = pts - np.array([roi_x0, roi_y0], dtype=np.int32)
        buf = np.empty((max(1, roi_y1 - roi_y0), max(1, roi_x1 - roi_x0), 3), dtype=np.uint8)
        cached = (
            cache_key,
            top_y,
            top_left,
            top_right,
            bottom_left,
            bottom_right,
            roi_x0,
            roi_x1,
            roi_y0,
            roi_y1,
            pts_rel,
            buf,
        )
        _corridor_overlay_cache["entry"] = cached

    (
        _,
        top_y,
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        roi_x0,
        roi_x1,
        roi_y0,
        roi_y1,
        pts_rel,
        overlay_buf,
    ) = cached

    # Subtle translucent overlay over only the corridor bounding box.
    if roi_x1 > roi_x0 and roi_y1 > roi_y0:
        roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]
        np.copyto(overlay_buf, roi)
        cv2.fillPoly(overlay_buf, [pts_rel], (40, 40, 40))
        cv2.addWeighted(overlay_buf, 0.18, roi, 0.82, 0, roi)

    cv2.line(frame, bottom_left, top_left, (255, 255, 255), 2)
    cv2.line(frame, bottom_right, top_right, (255, 255, 255), 2)
    cv2.line(frame, top_left, top_right, (255, 255, 255), 1)

    cv2.putText(
        frame,
        "walking corridor",
        (top_left[0], max(20, top_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )


def make_alert_key(track_id, class_name, corridor_position):
    # Zone-based keys are intentionally stable for speech timing.
    if USE_TRACK_ID_FOR_ALERT_KEY and track_id != -1:
        return f"id:{track_id}"
    return f"zone:{class_name}:{corridor_position}"


def make_distance_key(track_id, class_name, corridor_position):
    # Distance and motion estimates should stay separate for individual objects.
    if track_id != -1:
        return f"id:{track_id}"
    return f"zone:{class_name}:{corridor_position}"


# ============================================================
# VL53L5CX TOF WORKER + CAMERA FUSION HELPERS
# ============================================================

class ToFDepthWorker:
    """Continuously reads the newest 8x8 VL53L5CX depth frame."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.sensor = None

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self._run,
            name="vl53l5cx-depth-worker",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        sensor = self.sensor
        if sensor is not None:
            try:
                sensor.stop_ranging()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=3.0)

    def _set_state(self, **updates):
        with tof_state_lock:
            tof_state.update(updates)

    def _initialize_sensor(self):
        if qwiic_vl53l5cx is None:
            raise RuntimeError(
                "SparkFun ToF package missing. Install with: "
                "python -m pip install sparkfun-qwiic-vl53l5cx"
            )

        sensor = qwiic_vl53l5cx.QwiicVL53L5CX()
        if not sensor.is_connected():
            raise RuntimeError("VL53L5CX not found on I2C. Check Qwiic wiring and I2C.")

        print("[ToF] Initializing VL53L5CX; firmware upload can take several seconds...")
        if not sensor.begin():
            raise RuntimeError("VL53L5CX begin() failed")

        resolution = getattr(sensor, "kResolution8x8", TOF_RESOLUTION)
        status_ok = getattr(sensor, "kStatusOK", 0)

        if sensor.set_resolution(resolution) != status_ok:
            raise RuntimeError("VL53L5CX could not enter 8x8 mode")

        max_hz = 15 if int(TOF_RESOLUTION) == 64 else 60
        frequency = int(clamp(TOF_FREQUENCY_HZ, 1, max_hz))
        if sensor.set_ranging_frequency_hz(frequency) != status_ok:
            print(f"[ToF warning] Could not set frequency to {frequency} Hz; using sensor default.")

        target_closest = getattr(sensor, "kTargetOrderClosest", None)
        if target_closest is not None:
            try:
                sensor.set_target_order(target_closest)
            except Exception as error:
                print(f"[ToF warning] Could not set closest-target mode: {error}")

        if sensor.start_ranging() != status_ok:
            raise RuntimeError("VL53L5CX start_ranging() failed")

        self.sensor = sensor
        self._set_state(
            connected=True,
            initialized=True,
            last_error=None,
        )
        print(f"[ToF] Running 8x8 depth at requested {frequency} Hz.")

    @staticmethod
    def _reshape_optional(values, dtype=None):
        if values is None:
            return None
        array = np.asarray(values, dtype=dtype).reshape(-1)
        if array.size < 64:
            return None
        return array[:64].reshape(8, 8)

    def _store_measurement(self, measurement):
        distances = self._reshape_optional(getattr(measurement, "distance_mm", None), dtype=np.float32)
        if distances is None:
            raise RuntimeError("ToF result did not contain 64 distance values")

        status_values = getattr(measurement, "status", None)
        if status_values is None:
            status_values = getattr(measurement, "target_status", None)
        status = self._reshape_optional(status_values)
        targets = self._reshape_optional(getattr(measurement, "nb_target_detected", None))

        distances = distances.copy()
        valid = np.isfinite(distances)
        valid &= distances >= TOF_MIN_DISTANCE_MM
        valid &= distances <= TOF_MAX_DISTANCE_MM

        if status is not None:
            if len(TOF_VALID_STATUS_CODES) == 2:
                code_a, code_b = tuple(TOF_VALID_STATUS_CODES)
                valid &= (status == code_a) | (status == code_b)
            else:
                valid &= np.isin(status, tuple(TOF_VALID_STATUS_CODES))
        if targets is not None:
            valid &= targets > 0

        distances[~valid] = np.nan

        if TOF_FLIP_X:
            distances = distances[:, ::-1]
            if status is not None:
                status = status[:, ::-1]
        if TOF_FLIP_Y:
            distances = distances[::-1, :]
            if status is not None:
                status = status[::-1, :]

        distances = np.ascontiguousarray(distances)
        if status is not None:
            status = np.ascontiguousarray(status)
        distances.setflags(write=False)
        if status is not None:
            status.setflags(write=False)

        with tof_state_lock:
            tof_state["connected"] = True
            tof_state["initialized"] = True
            tof_state["grid_mm"] = distances
            tof_state["status_grid"] = status
            tof_state["last_update"] = time.time()
            tof_state["last_error"] = None
            tof_state["frame_counter"] += 1

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self._initialize_sensor()
                while not self.stop_event.is_set():
                    if self.sensor.check_data_ready():
                        measurement = self.sensor.get_ranging_data()
                        self._store_measurement(measurement)
                    else:
                        self.stop_event.wait(0.008)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                print(f"[ToF error] {message}")
                self._set_state(
                    connected=False,
                    initialized=False,
                    last_error=message,
                )
                self.sensor = None
                # Missing software cannot recover by retrying. Wiring/I2C faults
                # may recover after reconnection, so those continue retrying.
                if TOF_REQUIRED or qwiic_vl53l5cx is None:
                    return
                self.stop_event.wait(TOF_RECONNECT_SECONDS)


def start_tof_worker():
    global tof_worker
    if not ENABLE_TOF:
        print("[ToF] Disabled in settings.")
        return
    if tof_worker is None:
        tof_worker = ToFDepthWorker()
    tof_worker.start()


def stop_tof_worker():
    if tof_worker is not None:
        tof_worker.stop()


def _fast_percentile_1d(valid, percentile):
    """Exact linear-interpolation percentile for tiny 1D float arrays without numpy dispatch overhead."""
    n = valid.size
    if n == 1:
        return float(valid[0])
    sorted_vals = np.sort(valid)
    idx = (percentile * 0.01) * (n - 1)
    lo = int(idx)
    if lo >= n - 1:
        return float(sorted_vals[-1])
    frac = idx - lo
    if frac == 0.0:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo]))


def get_tof_snapshot(copy_grid=False):
    """Return the latest immutable grid (or a copy if requested) plus age/health information."""
    with tof_state_lock:
        grid = tof_state["grid_mm"]
        snapshot = {
            "connected": bool(tof_state["connected"]),
            "initialized": bool(tof_state["initialized"]),
            "grid_mm": None if grid is None else (grid.copy() if copy_grid else grid),
            "last_update": float(tof_state["last_update"]),
            "last_error": tof_state["last_error"],
            "frame_counter": int(tof_state["frame_counter"]),
        }
    snapshot["age_seconds"] = (
        float("inf")
        if snapshot["last_update"] <= 0
        else max(0.0, time.time() - snapshot["last_update"])
    )
    snapshot["fresh"] = (
        snapshot["grid_mm"] is not None
        and snapshot["age_seconds"] <= TOF_STALE_SECONDS
    )
    return snapshot


def maybe_print_tof_summary(snapshot=None):
    global last_tof_summary_time
    if not (ENABLE_TOF and TOF_PRINT_SUMMARY):
        return
    now = time.time()
    if now - last_tof_summary_time < TOF_PRINT_INTERVAL_SECONDS:
        return
    last_tof_summary_time = now

    if snapshot is None:
        snapshot = get_tof_snapshot()
    grid = snapshot["grid_mm"]
    if snapshot["fresh"] and grid is not None:
        valid = grid[np.isfinite(grid)]
        if valid.size > 0:
            print(
                "[ToF] "
                f"closest={np.min(valid) / 1000.0:.2f}m "
                f"median={_fast_percentile_1d(valid, 50.0) / 1000.0:.2f}m "
                f"valid={valid.size}/64 "
                f"fusion={'ON' if TOF_MOUNT_CALIBRATED else 'LOCKED until mounted'}"
            )
            return
    if snapshot["last_error"]:
        print(f"[ToF] unavailable: {snapshot['last_error']}")
    else:
        print("[ToF] waiting for first depth frame...")


def tof_overlap_rectangle(frame_width, frame_height):
    width = TOF_OVERLAP_WIDTH_FRACTION * frame_width
    height = TOF_OVERLAP_HEIGHT_FRACTION * frame_height
    center_x = (0.5 + TOF_OVERLAP_CENTER_X_OFFSET) * frame_width
    center_y = (0.5 + TOF_OVERLAP_CENTER_Y_OFFSET) * frame_height

    left = clamp(center_x - width / 2, 0, frame_width - 1)
    right = clamp(center_x + width / 2, 1, frame_width)
    top = clamp(center_y - height / 2, 0, frame_height - 1)
    bottom = clamp(center_y + height / 2, 1, frame_height)
    return float(left), float(top), float(right), float(bottom)


def _pixel_to_tof_cell(x, y, overlap):
    left, top, right, bottom = overlap
    col = int((x - left) / max(1.0, right - left) * 8)
    row = int((y - top) / max(1.0, bottom - top) * 8)
    return int(clamp(col, 0, 7)), int(clamp(row, 0, 7))


def tof_distance_for_box(x1, y1, x2, y2, frame_width, frame_height, snapshot=None, overlap=None,
                         diagnostics=None):
    """Match a YOLO box to overlapping ToF cells.

    Returns None when the mount has not been calibrated, the box is outside the
    overlap, or no fresh valid cells are available. ``diagnostics`` is filled
    with the reason and sampled rectangle for controlled association tests.
    """
    def _diag(reason, **extra):
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(reason=reason, **extra)

    if not (ENABLE_TOF and TOF_MOUNT_CALIBRATED):
        _diag("tof_disabled_or_uncalibrated")
        return None

    if snapshot is None:
        snapshot = get_tof_snapshot()
    if not snapshot["fresh"]:
        _diag("tof_not_fresh", age_seconds=snapshot.get("age_seconds"))
        return None

    if overlap is None:
        overlap = tof_overlap_rectangle(frame_width, frame_height)
    left, top, right, bottom = overlap

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    sample_left = x1 + box_w * TOF_BOX_X_INSET_FRACTION
    sample_right = x2 - box_w * TOF_BOX_X_INSET_FRACTION
    sample_top = y1 + box_h * TOF_BOX_Y_START_FRACTION
    sample_bottom = y2

    inter_left = max(sample_left, left)
    inter_right = min(sample_right, right)
    inter_top = max(sample_top, top)
    inter_bottom = min(sample_bottom, bottom)

    reason = "matched"
    if inter_right <= inter_left:
        _diag(
            "box_outside_overlap_horizontal",
            sample_rect=(sample_left, sample_top, sample_right, sample_bottom),
            overlap_rect=overlap,
        )
        return None
    if inter_bottom <= inter_top:
        if (
            TOF_BOX_CLAMP_LOW_OBJECTS
            and sample_top >= bottom
            and sample_bottom > bottom
        ):
            # The visible object is below the calibrated depth rectangle.
            # Use the nearest (bottom) ToF row instead of dropping the match.
            inter_top = max(top, bottom - 1.0)
            inter_bottom = bottom
            reason = "matched_below_overlap_clamped"
        else:
            _diag(
                "box_outside_overlap_vertical",
                sample_rect=(sample_left, sample_top, sample_right, sample_bottom),
                overlap_rect=overlap,
            )
            return None

    col0, row0 = _pixel_to_tof_cell(inter_left, inter_top, overlap)
    col1, row1 = _pixel_to_tof_cell(inter_right - 1e-3, inter_bottom - 1e-3, overlap)

    col0 = int(clamp(col0 - TOF_CELL_PADDING, 0, 7))
    col1 = int(clamp(col1 + TOF_CELL_PADDING, 0, 7))
    row0 = int(clamp(row0 - TOF_CELL_PADDING, 0, 7))
    row1 = int(clamp(row1 + TOF_CELL_PADDING, 0, 7))

    cells = snapshot["grid_mm"][row0:row1 + 1, col0:col1 + 1]
    valid = cells[np.isfinite(cells)]
    if valid.size < TOF_MIN_VALID_CELLS:
        _diag(
            "no_valid_cells",
            row_range=(row0, row1),
            col_range=(col0, col1),
            valid_cells=int(valid.size),
            sample_rect=(sample_left, sample_top, sample_right, sample_bottom),
            overlap_rect=overlap,
        )
        return None

    distance_mm = _fast_percentile_1d(valid, TOF_DISTANCE_PERCENTILE)
    _diag(
        reason,
        row_range=(row0, row1),
        col_range=(col0, col1),
        valid_cells=int(valid.size),
        sample_rect=(sample_left, sample_top, sample_right, sample_bottom),
        overlap_rect=overlap,
    )
    return {
        "distance_m": distance_mm / 1000.0,
        "valid_cells": int(valid.size),
        "row_range": (row0, row1),
        "col_range": (col0, col1),
        "age_seconds": snapshot["age_seconds"],
        "overlap_rect": overlap,
    }


def tof_unknown_path_distance(snapshot=None):
    if not (
        ENABLE_TOF
        and TOF_MOUNT_CALIBRATED
        and TOF_UNKNOWN_OBSTACLE_ENABLED
    ):
        return None
    if snapshot is None:
        snapshot = get_tof_snapshot()
    if not snapshot["fresh"]:
        return None
    grid = snapshot["grid_mm"]
    cells = grid[
        TOF_UNKNOWN_PATH_ROW_START:TOF_UNKNOWN_PATH_ROW_END + 1,
        TOF_UNKNOWN_PATH_COL_START:TOF_UNKNOWN_PATH_COL_END + 1,
    ]
    valid = cells[np.isfinite(cells)]
    if valid.size < TOF_UNKNOWN_MIN_VALID_CELLS:
        return None

    # Median, not a low percentile: one spurious near-range cell must not turn
    # a 2 m empty corridor into a 0.05 m obstacle. The spread is logged so a
    # genuinely noisy sensor can be identified rather than hidden.
    distance_m = _fast_percentile_1d(valid, TOF_UNKNOWN_PERCENTILE) / 1000.0
    spread_m = (
        _fast_percentile_1d(valid, 90.0) - _fast_percentile_1d(valid, 10.0)
    ) / 1000.0
    return {
        "distance_m": distance_m,
        "spread_m": spread_m,
        "valid_cells": int(valid.size),
        "age_seconds": snapshot["age_seconds"],
    }


def unknown_is_separate_from_known(
    unknown_distance_m, nearest_tof_m, nearest_object_m,
):
    """True only when no known object can explain the generic depth reading."""
    nearest_known_m = min(nearest_tof_m, nearest_object_m)
    return (
        unknown_distance_m + TOF_UNKNOWN_MATCH_TOLERANCE_M
        < nearest_known_m
    )


def reset_unknown_obstacle_state():
    global unknown_obstacle_state
    unknown_obstacle_state = {
        "distance_m": None,
        "last_update": 0.0,
        "risk": 0,
    }


def smooth_unknown_path_distance(raw_distance_m, now=None):
    """Temporally stabilize the generic path distance and remember its band."""
    now = time.time() if now is None else now
    state = unknown_obstacle_state
    previous = state["distance_m"]
    if previous is None or now - state["last_update"] > TOF_STALE_SECONDS:
        smoothed = raw_distance_m
    else:
        alpha = clamp(TOF_UNKNOWN_SMOOTH_ALPHA, 0.01, 1.0)
        smoothed = alpha * raw_distance_m + (1.0 - alpha) * previous
    state["distance_m"] = smoothed
    state["last_update"] = now
    return smoothed


def note_unknown_path_absent(now=None):
    now = time.time() if now is None else now
    if (
        unknown_obstacle_state["distance_m"] is not None
        and now - unknown_obstacle_state["last_update"] > TOF_STALE_SECONDS
    ):
        reset_unknown_obstacle_state()


def unknown_obstacle_risk(distance_m, previous_risk=0):
    """Map distance to risk with hysteresis so band edges do not chatter."""
    hysteresis = max(0.0, TOF_UNKNOWN_HYSTERESIS_M)
    stop_limit = TOF_UNKNOWN_STOP_M + (hysteresis if previous_risk >= 3 else 0.0)
    caution_limit = TOF_UNKNOWN_CAUTION_M + (hysteresis if previous_risk >= 2 else 0.0)
    notice_limit = TOF_UNKNOWN_NOTICE_M + (hysteresis if previous_risk >= 1 else 0.0)

    if distance_m <= stop_limit:
        risk = 3
        reason = "unknown obstacle very close in path"
    elif distance_m <= caution_limit:
        risk = 2
        reason = "unknown obstacle close in path"
    elif distance_m <= notice_limit:
        risk = 1
        reason = "unknown obstacle in path"
    else:
        risk = 0
        reason = "unknown obstacle far in path"
    return risk, reason


def print_unknown_obstacle_debug(now, raw_m, smooth_m, valid_cells, spread_m,
                                 nearest_known_m, risk, reason, action):
    global last_unknown_debug_time
    if not TOF_UNKNOWN_DEBUG:
        return
    if now - last_unknown_debug_time < TOF_UNKNOWN_DEBUG_INTERVAL_SECONDS:
        return
    last_unknown_debug_time = now
    nearest = (
        "none" if not math.isfinite(nearest_known_m)
        else f"{nearest_known_m:.2f}m"
    )
    print(
        f"[unknown obstacle {now_str()}] "
        f"raw={raw_m:.2f}m smooth={smooth_m:.2f}m "
        f"valid={valid_cells} spread={spread_m:.2f}m "
        f"nearest_known={nearest} risk={risk} action={action} "
        f"reason={reason}"
    )


last_association_debug_time = {}


def maybe_print_association_debug(
    class_name, conf, x1, y1, x2, y2, track_id, risk, reason, debug,
):
    """Log camera detection, ToF match, and association reason (opt-in).

    This is the evidence needed to distinguish a missed camera detection from
    a detection that the ToF association dropped. It is rate-limited per class
    so a controlled chair/box/basket test stays readable.
    """
    if not TOF_ASSOCIATION_DEBUG:
        return
    now = time.time()
    key = str(class_name)
    if now - last_association_debug_time.get(key, 0.0) < TOF_ASSOCIATION_DEBUG_INTERVAL_SECONDS:
        return
    last_association_debug_time[key] = now

    def _rect(value):
        if not value:
            return "none"
        return "(" + ",".join(f"{float(part):.0f}" for part in value) + ")"

    match_distance = debug.get("tof_match_distance_m")
    match_text = "none" if match_distance is None else f"{match_distance:.2f}m"
    print(
        f"[association {now_str()}] class={class_name} conf={conf:.2f} "
        f"track={track_id} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
        f"foot=({(x1 + x2) / 2.0:.0f},{y2:.0f}) "
        f"in_path={debug.get('in_corridor')} pos={debug.get('corridor_position')} "
        f"tof={match_text} reason={debug.get('tof_match_reason')} "
        f"cells={debug.get('tof_valid_cells')} "
        f"rows={debug.get('tof_row_range')} cols={debug.get('tof_col_range')} "
        f"sample={_rect(debug.get('tof_sample_rect'))} "
        f"overlap={_rect(debug.get('tof_overlap_rect'))} "
        f"visual={debug.get('fake_distance_m'):.2f}m "
        f"source={debug.get('distance_source')} "
        f"risk={risk} alert={reason}",
        flush=True,
    )


def make_unknown_tof_debug(distance_m, valid_cells):
    return {
        "direction": "ahead",
        "corridor_position": "in path",
        "in_corridor": True,
        "lower_half": True,
        "distance_level": visual_distance_level(distance_m),
        "raw_distance_m": distance_m,
        "fake_distance_m": distance_m,
        "visual_distance_m": None,
        "distance_source": "tof_unknown",
        "approach_mps": 0.0,
        "approaching": False,
        "fast_approaching": False,
        "area_growth": 0.0,
        "speed_px": 0.0,
        "box_h_px": 0.0,
        "ttc_seconds": float("inf"),
        "tof_valid_cells": valid_cells,
        "tof_age_seconds": 0.0,
        "tof_row_range": None,
        "tof_col_range": None,
    }

# ============================================================
# DISTANCE ENGINE
# ============================================================

def visual_distance_level(distance_m):
    if distance_m < VERY_CLOSE_M:
        return "very_close"
    elif distance_m < CLOSE_M:
        return "close"
    elif distance_m < MEDIUM_M:
        return "medium"
    return "far"


def fallback_area_distance_meters(x1, y1, x2, y2, frame_width, frame_height):
    area = box_area(x1, y1, x2, y2)
    ratio = area / max(1, frame_width * frame_height)

    if ratio > 0.45:
        return 0.4
    elif ratio > 0.32:
        return 0.8
    elif ratio > 0.22:
        return 1.2
    elif ratio > 0.08:
        return 2.0
    elif ratio > 0.03:
        return 3.5
    return 5.0


def reference_height_distance_meters(class_name, x1, y1, x2, y2):
    h = box_height(y1, y2)

    if FOCAL_LENGTH_PX is None:
        return None

    if class_name not in REFERENCE_HEIGHT_M:
        return None

    return FOCAL_LENGTH_PX * REFERENCE_HEIGHT_M[class_name] / h


def calibrated_height_distance_meters(class_name, x1, y1, x2, y2):
    h = box_height(y1, y2)

    if class_name not in DISTANCE_CALIBRATION:
        return None

    cal = DISTANCE_CALIBRATION[class_name]
    distance_m = cal["known_distance_m"] * cal["known_box_height_px"] / h
    return distance_m


def raw_fake_distance_meters(class_name, x1, y1, x2, y2, frame_width, frame_height):
    if DISTANCE_MODE == "reference_height":
        d = reference_height_distance_meters(class_name, x1, y1, x2, y2)
        if d is not None:
            return clamp(d, MIN_FAKE_DISTANCE_M, MAX_FAKE_DISTANCE_M), "reference_height"

    if DISTANCE_MODE == "calibrated_height":
        d = calibrated_height_distance_meters(class_name, x1, y1, x2, y2)
        if d is not None:
            return clamp(d, MIN_FAKE_DISTANCE_M, MAX_FAKE_DISTANCE_M), "calibrated_height"

    d = fallback_area_distance_meters(x1, y1, x2, y2, frame_width, frame_height)
    return clamp(d, MIN_FAKE_DISTANCE_M, MAX_FAKE_DISTANCE_M), "area_fallback"


def update_distance_memory(distance_key, raw_distance_m, update=True):
    now = time.time()

    with state_lock:
        if distance_key not in distance_memory:
            distance_memory[distance_key] = {
                "raw_distance_m": raw_distance_m,
                "smooth_distance_m": raw_distance_m,
                "last_update_time": now,
                "approach_mps": 0.0,
            }
            return raw_distance_m, 0.0

        mem = distance_memory[distance_key]

        if not update:
            return mem["smooth_distance_m"], mem.get("approach_mps", 0.0)

        last_smooth = mem["smooth_distance_m"]
        last_time = mem["last_update_time"]
        dt = max(0.001, now - last_time)

        smooth = DISTANCE_SMOOTH_ALPHA * raw_distance_m + (1.0 - DISTANCE_SMOOTH_ALPHA) * last_smooth

        # Positive means distance is shrinking, so object is approaching.
        approach_mps = (last_smooth - smooth) / dt

        mem["raw_distance_m"] = raw_distance_m
        mem["smooth_distance_m"] = smooth
        mem["last_update_time"] = now
        mem["approach_mps"] = approach_mps

        return smooth, approach_mps


def estimate_motion(track_id):
    if track_id == -1:
        return 0, 0, 0, 0

    with state_lock:
        hist = track_history.get(track_id)
        if not hist or len(hist) < 2:
            return 0, 0, 0, 0

        old = hist[0]
        new = hist[-1]

    dx = new["cx"] - old["cx"]
    dy = new["cy"] - old["cy"]

    old_area = max(1, old["area"])
    new_area = max(1, new["area"])

    area_growth = (new_area - old_area) / old_area
    speed_px = math.hypot(dx, dy)

    return dx, dy, area_growth, speed_px


# ============================================================
# CSV LOGGING
# ============================================================

CSV_FIELDS = [
    "time",
    "event",
    "class_name",
    "confidence",
    "risk",
    "reason",
    "direction",
    "corridor_position",
    "in_corridor",
    "distance_level",
    "distance_source",
    "raw_distance_m",
    "fake_distance_m",
    "visual_distance_m",
    "tof_valid_cells",
    "tof_age_seconds",
    "ttc_seconds",
    "approach_mps",
    "area_growth",
    "speed_px",
    "box_h_px",
    "face_name",
    "face_distance_score",
    "emotion",
    "emotion_confidence",
    "track_id",
]


def ensure_log_file():
    global _log_file_handle, _log_writer
    if not LOG_CSV:
        return

    if _log_file_handle is not None and not _log_file_handle.closed:
        return

    path = Path(LOG_FILE)
    needs_header = not path.exists() or path.stat().st_size == 0

    _log_file_handle = open(LOG_FILE, "a", newline="", encoding="utf-8", buffering=1)
    _log_writer = csv.DictWriter(_log_file_handle, fieldnames=CSV_FIELDS)
    if needs_header:
        _log_writer.writeheader()
        _log_file_handle.flush()


def close_log_file():
    global _log_file_handle, _log_writer
    handle = _log_file_handle
    _log_file_handle = None
    _log_writer = None
    if handle is not None and not handle.closed:
        try:
            handle.flush()
            handle.close()
        except OSError:
            pass


def log_row(event, class_name, confidence, risk, reason, debug, track_id):
    if not LOG_CSV:
        return

    if _log_writer is None:
        ensure_log_file()
    if _log_writer is None:
        return

    _log_writer.writerow({
        "time": now_str(),
        "event": event,
        "class_name": class_name,
        "confidence": round(confidence, 3),
        "risk": risk,
        "reason": reason,
        "direction": debug["direction"],
        "corridor_position": debug["corridor_position"],
        "in_corridor": debug["in_corridor"],
        "distance_level": debug["distance_level"],
        "distance_source": debug["distance_source"],
        "raw_distance_m": round(debug["raw_distance_m"], 2),
        "fake_distance_m": round(debug["fake_distance_m"], 2),
        "visual_distance_m": (
            "" if debug.get("visual_distance_m") is None
            else round(debug.get("visual_distance_m"), 2)
        ),
        "tof_valid_cells": debug.get("tof_valid_cells", 0),
        "tof_age_seconds": (
            "" if debug.get("tof_age_seconds") is None
            else round(debug.get("tof_age_seconds"), 3)
        ),
        "ttc_seconds": (
            "" if not math.isfinite(debug.get("ttc_seconds", float("inf")))
            else round(debug.get("ttc_seconds"), 2)
        ),
        "approach_mps": round(debug["approach_mps"], 2),
        "area_growth": round(debug["area_growth"], 3),
        "speed_px": round(debug["speed_px"], 2),
        "box_h_px": round(debug["box_h_px"], 1),
        "face_name": debug.get("face_name", ""),
        "face_distance_score": (
            "" if debug.get("face_distance_score") is None
            else round(debug.get("face_distance_score"), 3)
        ),
        "emotion": debug.get("emotion", ""),
        "emotion_confidence": (
            "" if debug.get("emotion_confidence") is None
            else round(debug.get("emotion_confidence"), 3)
        ),
        "track_id": track_id,
    })
    if event == "alert" and _log_file_handle is not None:
        _log_file_handle.flush()


# ============================================================
# RISK ENGINE
# ============================================================

def apply_corridor_filter(class_name, risk, reason, distance_level, distance_m, approaching, fast_approaching, lower_half, in_corridor, corridor_position):
    """
    Makes the walking corridor matter.
    Objects in the path keep their normal risk.
    Side objects are downgraded unless they are very close or moving in.
    """
    if not USE_WALKING_CORRIDOR:
        return risk, reason

    if in_corridor:
        return risk, reason.replace("ahead", "in path") if "ahead" in reason else f"{reason} in path"

    # Static obstacles on the side usually should not trigger walking warnings.
    if class_name in STATIC_OBSTACLE_CLASSES:
        if lower_half and distance_level == "very_close":
            return min(risk, 1), f"side obstacle very close {corridor_position}"
        return 0, f"side {class_name} ignored"

    # People on the side are useful notice, but not usually STOP.
    if class_name in PERSON_CLASSES:
        if distance_m < PERSON_STOP_DISTANCE_M and fast_approaching:
            return 2, f"person moving in from {corridor_position}"
        if distance_level == "very_close":
            return min(risk, 2), f"person very close {corridor_position}"
        if distance_level == "close" and approaching:
            return min(risk, 2), f"person approaching from {corridor_position}"
        if distance_level in {"close", "medium"}:
            return min(risk, 1), f"person {corridor_position}"
        return 0, f"person far {corridor_position}"

    # Vehicles on the side can still matter if close or approaching.
    if class_name in VEHICLE_CLASSES:
        if distance_level in {"very_close", "close"} or fast_approaching:
            return min(max(risk, 2), 3), f"{class_name} close {corridor_position}"
        if approaching and distance_level == "medium":
            return min(max(risk, 2), 2), f"{class_name} approaching {corridor_position}"
        return min(risk, 1), f"{class_name} {corridor_position}"

    # Animals on side: warn only if close / approaching.
    if class_name in ANIMAL_CLASSES:
        if distance_level == "very_close" or (distance_level == "close" and approaching):
            return min(max(risk, 2), 2), f"{class_name} close {corridor_position}"
        if distance_level in {"close", "medium"}:
            return min(risk, 1), f"{class_name} {corridor_position}"
        return 0, f"{class_name} far {corridor_position}"

    # Signs and other objects outside path are passive.
    return min(risk, 1), f"{class_name} {corridor_position}"


def score_risk(
    class_name,
    cx,
    cy,
    x1,
    y1,
    x2,
    y2,
    frame_width,
    frame_height,
    track_id,
    distance_key,
    update_distance,
    in_corridor=None,
    corridor_position=None,
    tof_snapshot=None,
    tof_overlap=None,
):
    direction = direction_word(cx, frame_width)
    foot_y = y2
    lower_half = cy > frame_height * LOWER_FRAME_IMPORTANCE
    very_low_object = foot_y > frame_height * VERY_LOW_OBJECT_Y

    if in_corridor is None or corridor_position is None:
        in_corridor, corridor_position = corridor_geometry_for_point(
            cx, foot_y, frame_width, frame_height
        )

    _, _, area_growth, speed_px = estimate_motion(track_id)

    visual_distance_m, visual_distance_source = raw_fake_distance_meters(
        class_name=class_name,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    tof_match_diagnostics = {}
    tof_match = tof_distance_for_box(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        frame_width=frame_width,
        frame_height=frame_height,
        snapshot=tof_snapshot,
        overlap=tof_overlap,
        diagnostics=tof_match_diagnostics,
    )

    if tof_match is not None:
        raw_distance_m = tof_match["distance_m"]
        distance_source = "tof"
    else:
        raw_distance_m = visual_distance_m
        distance_source = visual_distance_source

    distance_m, approach_mps = update_distance_memory(
        distance_key=distance_key,
        raw_distance_m=raw_distance_m,
        update=update_distance,
    )

    distance_level = visual_distance_level(distance_m)

    approaching_by_distance = approach_mps > APPROACHING_MPS
    fast_approaching_by_distance = approach_mps > FAST_APPROACHING_MPS
    approaching_by_size = area_growth > AREA_GROWTH_APPROACHING
    fast_approaching_by_size = area_growth > AREA_GROWTH_FAST

    approaching = approaching_by_distance or approaching_by_size
    fast_approaching = fast_approaching_by_distance or fast_approaching_by_size

    if approach_mps > 0.10:
        ttc_seconds = distance_m / approach_mps
    else:
        ttc_seconds = float("inf")

    risk = 0
    reason = "passive"

    # ----------------------------
    # PERSON
    # ----------------------------
    if class_name in PERSON_CLASSES:
        if distance_m < PERSON_STOP_DISTANCE_M and fast_approaching:
            risk = 3
            reason = f"person rushing {direction}"
        elif distance_m < PERSON_STOP_DISTANCE_M:
            risk = 2
            reason = f"person very close {direction}"
        elif distance_level == "very_close" and approaching:
            risk = 2
            reason = f"person approaching close {direction}"
        elif distance_level == "very_close":
            risk = 2
            reason = f"person very close {direction}"
        elif distance_level == "close":
            risk = 2
            reason = f"person close {direction}"
        elif approaching and distance_level == "medium":
            risk = 2
            reason = f"person approaching {direction}"
        elif distance_level == "medium":
            risk = 1
            reason = f"person {direction}"
        else:
            risk = 0
            reason = f"person far {direction}"

    # ----------------------------
    # VEHICLES
    # ----------------------------
    elif class_name in VEHICLE_CLASSES:
        if distance_level == "very_close":
            risk = 3
            reason = f"{class_name} very close {direction}"
        elif fast_approaching and distance_level in {"close", "medium"}:
            risk = 3
            reason = f"{class_name} approaching fast {direction}"
        elif distance_level == "close":
            risk = 3
            reason = f"{class_name} close {direction}"
        elif approaching and distance_level == "medium":
            risk = 3
            reason = f"{class_name} approaching {direction}"
        elif distance_level == "medium":
            risk = 2
            reason = f"{class_name} {direction}"
        else:
            risk = 1
            reason = f"{class_name} nearby {direction}"

    # ----------------------------
    # ANIMALS
    # ----------------------------
    elif class_name in ANIMAL_CLASSES:
        if distance_level == "very_close" and approaching:
            risk = 3
            reason = f"{class_name} very close and approaching {direction}"
        elif distance_level == "very_close":
            risk = 2
            reason = f"{class_name} very close {direction}"
        elif distance_level == "close":
            risk = 2
            reason = f"{class_name} close {direction}"
        elif approaching and distance_level == "medium":
            risk = 2
            reason = f"{class_name} approaching {direction}"
        elif distance_level == "medium":
            risk = 1
            reason = f"{class_name} {direction}"
        else:
            risk = 0
            reason = f"{class_name} far {direction}"

    # ----------------------------
    # STATIC OBSTACLES
    # ----------------------------
    elif class_name in STATIC_OBSTACLE_CLASSES:
        path_relevant = lower_half or very_low_object
        if path_relevant and distance_level == "very_close":
            risk = 3
            reason = f"{class_name} very close {direction}"
        elif path_relevant and distance_level == "close":
            risk = 2
            reason = f"{class_name} close {direction}"
        elif path_relevant and approaching and distance_level == "medium":
            risk = 2
            reason = f"{class_name} getting closer {direction}"
        elif path_relevant and distance_level == "medium":
            risk = 1
            reason = f"{class_name} {direction}"
        elif distance_level == "very_close":
            risk = 1
            reason = f"{class_name} very close but not low {direction}"
        else:
            risk = 0
            reason = f"passive {class_name} {direction}"

    # ----------------------------
    # SIGNS / TRAFFIC LIGHTS
    # ----------------------------
    elif class_name in PASSIVE_SIGN_CLASSES:
        if distance_level == "very_close":
            risk = 1
            reason = f"{class_name} very close {direction}"
        else:
            risk = 0
            reason = f"passive {class_name} {direction}"

    # ----------------------------
    # OTHER OBJECTS
    # ----------------------------
    else:
        if lower_half and distance_level == "very_close":
            risk = 1
            reason = f"object close {direction}"
        else:
            risk = 0
            reason = f"passive {class_name} {direction}"

    risk, reason = apply_corridor_filter(
        class_name=class_name,
        risk=risk,
        reason=reason,
        distance_level=distance_level,
        distance_m=distance_m,
        approaching=approaching,
        fast_approaching=fast_approaching,
        lower_half=lower_half,
        in_corridor=in_corridor,
        corridor_position=corridor_position,
    )

    # Real depth enables time-to-collision decisions. This also catches the
    # wearer's own motion toward a stationary obstacle, which is useful for
    # collision avoidance.
    if distance_source == "tof" and in_corridor:
        if ttc_seconds <= TOF_TTC_STOP_SECONDS and distance_m <= CLOSE_M:
            risk = max(risk, 3)
            reason = f"{class_name} collision risk in path"
        elif ttc_seconds <= TOF_TTC_CAUTION_SECONDS and distance_m <= MEDIUM_M:
            risk = max(risk, 2)
            reason = f"{class_name} getting closer in path"

    debug = {
        "direction": direction,
        "corridor_position": corridor_position,
        "in_corridor": in_corridor,
        "lower_half": lower_half,
        "distance_level": distance_level,
        "raw_distance_m": raw_distance_m,
        "fake_distance_m": distance_m,
        "visual_distance_m": visual_distance_m,
        "distance_source": distance_source,
        "approach_mps": approach_mps,
        "ttc_seconds": ttc_seconds,
        "tof_valid_cells": 0 if tof_match is None else tof_match["valid_cells"],
        "tof_age_seconds": None if tof_match is None else tof_match["age_seconds"],
        "tof_row_range": None if tof_match is None else tof_match["row_range"],
        "tof_col_range": None if tof_match is None else tof_match["col_range"],
        "tof_match_reason": tof_match_diagnostics.get("reason"),
        "tof_sample_rect": tof_match_diagnostics.get("sample_rect"),
        "tof_overlap_rect": tof_match_diagnostics.get("overlap_rect"),
        "tof_match_distance_m": (
            None if tof_match is None else tof_match["distance_m"]
        ),
        "approaching": approaching,
        "fast_approaching": fast_approaching,
        "area_growth": area_growth,
        "speed_px": speed_px,
        "box_h_px": box_height(y1, y2),
    }

    return risk, reason, debug


# ============================================================
# OBJECT MEMORY + ALERT POLICY
# ============================================================

def update_object_memory(alert_key, risk, reason, distance_m=None):
    """Update hazard state without forgetting previously spoken alerts.

    The alert is rearmed only when cleanup_stale_memory() confirms that the
    object/class-zone has actually been absent for several seconds.
    """
    now = time.time()

    with state_lock:
        if alert_key not in track_memory:
            track_memory[alert_key] = {
                "first_seen": now,
                "last_seen": now,
                "last_risk": risk,
                "last_reason": reason,
                "last_distance_m": distance_m,
                "last_alert_time": 0.0,
                "last_alert_risk": 0,
                "last_alert_reason": None,
                "last_alert_distance_m": None,
                "spoken_count": 0,
            }
            return track_memory[alert_key]

        mem = track_memory[alert_key]
        old_risk = mem.get("last_risk", 0)

        # Restart stabilization only when entering a spoken level for the first
        # time, or when the danger level escalates.
        if (
            risk >= MIN_SPOKEN_RISK
            and (old_risk < MIN_SPOKEN_RISK or risk > old_risk)
        ):
            mem["first_seen"] = now

        mem["last_seen"] = now
        mem["last_risk"] = risk
        mem["last_reason"] = reason
        if distance_m is not None:
            mem["last_distance_m"] = distance_m

        return mem


def cleanup_stale_memory(now=None):
    if now is None:
        now = time.time()

    with state_lock:
        # Forget spoken-alert history only after the hazard has truly disappeared.
        stale_alert_keys = [
            key
            for key, mem in track_memory.items()
            if now - mem["last_seen"] > ALERT_REARM_AFTER_ABSENT_SECONDS
        ]
        for key in stale_alert_keys:
            del track_memory[key]

        stale_distance_keys = [
            key
            for key, mem in distance_memory.items()
            if now - mem["last_update_time"] > TRACK_STALE_SECONDS
        ]
        for key in stale_distance_keys:
            del distance_memory[key]

        stale_history_ids = [
            track_id
            for track_id, history in track_history.items()
            if not history or now - history[-1]["time"] > TRACK_STALE_SECONDS
        ]
        for track_id in stale_history_ids:
            del track_history[track_id]


def alert_text(risk, reason):
    if risk == 3:
        return f"STOP. {reason}."
    elif risk == 2:
        return f"Caution. {reason}."
    elif risk == 1:
        return f"Notice. {reason}."
    return ""


def should_alert(alert_key, risk, reason, distance_m=None):
    """Speak first alerts, escalations, meaningful changes, and timed reminders.

    Generic ToF "unknown obstacle" warnings use one stable key. For that key a
    distance-band change is not a new hazard, so it is deduplicated unless the
    risk escalates, the obstacle appears after the path was clear, or it becomes
    materially closer. This keeps a real warning but stops band-flap chatter.
    """
    global last_global_alert_time, last_alert_decision

    is_unknown = alert_key.startswith("tof:unknown:")
    last_alert_decision = {
        "alert_key": alert_key,
        "risk": risk,
        "reason": reason,
        "distance_m": distance_m,
        "spoken": False,
        "new": False,
        "escalated": False,
        "reason_changed": False,
        "materially_closer": False,
        "repeat": False,
        "suppressed_by": None,
    }

    if risk < MIN_SPOKEN_RISK:
        last_alert_decision["suppressed_by"] = "below_spoken_risk"
        return False

    now = time.time()
    with state_lock:
        mem = track_memory.get(alert_key)
        if mem is None:
            last_alert_decision["suppressed_by"] = "no_memory"
            return False

        stable_seconds = now - mem["first_seen"]

        if risk >= 3:
            stable_required = STABLE_SECONDS_FOR_STOP
            global_gap = GLOBAL_STOP_GAP_SECONDS
            repeat_seconds = STOP_REPEAT_SECONDS
        elif risk == 2:
            stable_required = STABLE_SECONDS_FOR_CAUTION
            global_gap = GLOBAL_CAUTION_GAP_SECONDS
            repeat_seconds = CAUTION_REPEAT_SECONDS
        else:
            stable_required = STABLE_SECONDS_FOR_NOTICE
            global_gap = GLOBAL_NOTICE_GAP_SECONDS
            repeat_seconds = NOTICE_REPEAT_SECONDS

        if is_unknown:
            global_gap = max(global_gap, TOF_UNKNOWN_MIN_SPEECH_GAP_SECONDS)
            repeat_seconds = TOF_UNKNOWN_REPEAT_SECONDS

        if stable_seconds < stable_required:
            last_alert_decision["suppressed_by"] = "stabilizing"
            return False

        never_spoken = mem["last_alert_time"] == 0
        risk_escalated = risk > mem["last_alert_risk"]
        reason_changed = reason != mem.get("last_alert_reason")
        previous_distance = mem.get("last_alert_distance_m")
        materially_closer = (
            is_unknown
            and previous_distance is not None
            and distance_m is not None
            and previous_distance - distance_m >= TOF_UNKNOWN_CLOSER_BYPASS_M
        )
        repeat_due = (
            REPEAT_ALERTS_WHILE_STILL_PRESENT
            and mem["last_alert_time"] > 0
            and now - mem["last_alert_time"] >= repeat_seconds
        )
        global_gap_passed = now - last_global_alert_time >= global_gap

        last_alert_decision.update(
            new=never_spoken,
            escalated=risk_escalated,
            reason_changed=reason_changed,
            materially_closer=materially_closer,
            repeat=repeat_due,
        )

        should_speak = never_spoken or risk_escalated or repeat_due
        if materially_closer:
            should_speak = True
        if SPEAK_ON_SAME_RISK_REASON_CHANGE and reason_changed:
            # Generic "something in path" band names are not distinct hazards.
            if not is_unknown or TOF_UNKNOWN_SPEAK_ON_REASON_CHANGE:
                should_speak = True

        if not global_gap_passed:
            # A materially closer/new hazard or a genuine new object may bypass
            # the global gap. Unknown band flapping may not; a sustained risk
            # escalation gets a shorter, still rate-limited bypass.
            bypass_gap = materially_closer or (never_spoken and risk >= 2)
            if is_unknown:
                bypass_gap = bypass_gap or (
                    risk_escalated
                    and now - last_global_alert_time
                    >= TOF_UNKNOWN_ESCALATION_GAP_SECONDS
                )
            else:
                bypass_gap = bypass_gap or risk_escalated
            if not bypass_gap:
                last_alert_decision["suppressed_by"] = "global_gap"
                return False

        if not should_speak:
            last_alert_decision["suppressed_by"] = "deduplicated"
            return False

        mem["last_alert_time"] = now
        mem["last_alert_risk"] = risk
        mem["last_alert_reason"] = reason
        if distance_m is not None:
            mem["last_alert_distance_m"] = distance_m
        mem["spoken_count"] = mem.get("spoken_count", 0) + 1
        last_global_alert_time = now
        last_alert_decision["spoken"] = True
        return True



def print_alert_debug(risk, reason, debug):
    global last_alert_debug_time

    if not PRINT_ALERT_DEBUG or debug is None:
        return

    now = time.time()
    if now - last_alert_debug_time < ALERT_DEBUG_INTERVAL_SECONDS:
        return

    last_alert_debug_time = now
    print(
        "[alert debug] "
        f"risk={risk} "
        f"distance={debug['fake_distance_m']:.2f}m "
        f"source={debug.get('distance_source')} "
        f"ttc={debug.get('ttc_seconds', float('inf')):.2f}s "
        f"position={debug['corridor_position']} "
        f"reason={reason}"
    )


def _speech_environment():
    """Return an environment that can connect to the user's PipeWire session."""
    global _speech_env_cache
    if _speech_env_cache is None:
        env = os.environ.copy()
        if AUDIO_RUNTIME_DIR:
            env["XDG_RUNTIME_DIR"] = AUDIO_RUNTIME_DIR
            env.setdefault(
                "DBUS_SESSION_BUS_ADDRESS",
                f"unix:path={AUDIO_RUNTIME_DIR}/bus",
            )
        _speech_env_cache = env
    return _speech_env_cache


def _get_speech_binaries():
    global _speech_program_path, _audio_player_path, _speech_paths_checked
    if not _speech_paths_checked:
        _speech_program_path = shutil.which("espeak-ng") or shutil.which("espeak")
        _audio_player_path = shutil.which("pw-play")
        _speech_paths_checked = True
    return _speech_program_path, _audio_player_path


def _preferred_speech_temp_dir():
    """Prefer RAM-backed tmpfs directories on Raspberry Pi to avoid SD card writes."""
    if AUDIO_RUNTIME_DIR and os.path.isdir(AUDIO_RUNTIME_DIR) and os.access(AUDIO_RUNTIME_DIR, os.W_OK):
        return AUDIO_RUNTIME_DIR
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return None


def _speak_message_now(message, origin_perf=None, origin_label="speech"):
    """Run one TTS utterance with shared activity/timing instrumentation."""
    _set_speech_active(True, message)
    _report_latency(origin_perf, origin_label, message=message)
    _voice_debug("tts_start", message=repr(message), protected=current_audio_protected)
    try:
        _speak_message_now_inner(
            message, count_command=(origin_label == "command"),
        )
    finally:
        _voice_debug(
            "tts_end",
            message=repr(message),
            duration_s=round(time.time() - speech_last_start_time, 3),
        )
        _set_speech_active(False)


def _speak_message_now_inner(message, count_command=False):
    """Stream synthesized WAV audio directly into pw-play; fall back to tmpfs WAV if needed."""
    global current_audio_process, _speech_missing_warned, voice_direct_responses_started

    speech_program, audio_player = _get_speech_binaries()

    if speech_program is None or audio_player is None:
        if not _speech_missing_warned:
            missing = "espeak-ng/espeak" if speech_program is None else "pw-play"
            print(f"[speech error] {missing} was not found; speech is disabled.")
            _speech_missing_warned = True
        return

    if count_command:
        voice_direct_responses_started += 1

    wav_path = None
    synth_proc = None
    try:
        if PRINT_SPEECH_DEBUG:
            print(f"[speech start] {message}")

        synth_cmd = [
            speech_program,
            "-v", SPEECH_VOICE,
            "-s", str(SPEECH_RATE),
            "--stdout",
            message,
        ]
        synth_proc = subprocess.Popen(
            synth_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # ponytail: direct pipe avoids SD writes; falls back to tmpfs file only if pw-play rejects '-'
        process = subprocess.Popen(
            [audio_player, "--target=auto", "-"],
            stdin=synth_proc.stdout,
            env=_speech_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if synth_proc.stdout is not None:
            synth_proc.stdout.close()

        with current_audio_lock:
            current_audio_process = process

        try:
            _stdout, stderr = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            print("[speech error] speech playback timed out.")
            return
        finally:
            if synth_proc.poll() is None:
                synth_proc.terminate()
            try:
                _, synth_err_bytes = synth_proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                synth_proc.kill()
                _, synth_err_bytes = synth_proc.communicate()

        if synth_proc.returncode not in (0, -13, -15):
            synth_err = (synth_err_bytes or b"").decode("utf-8", errors="replace").strip()
            print(f"[speech error] synthesis failed: {synth_err or synth_proc.returncode}")
            return

        # Fallback if an older pw-play version does not accept '-' stdin streaming.
        if process.returncode not in (0, -15) and "-" in (stderr or ""):
            with current_audio_lock:
                current_audio_process = None
            with tempfile.NamedTemporaryFile(
                prefix="whisper_hat_",
                suffix=".wav",
                dir=_preferred_speech_temp_dir(),
                delete=False,
            ) as temp_file:
                wav_path = temp_file.name

            speech_result = subprocess.run(
                [
                    speech_program,
                    "-v", SPEECH_VOICE,
                    "-s", str(SPEECH_RATE),
                    "-w", wav_path,
                    message,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
            if speech_result.returncode != 0:
                error_text = speech_result.stderr.strip() or "unknown eSpeak error"
                print(f"[speech error] WAV generation failed: {error_text}")
                return

            process = subprocess.Popen(
                [audio_player, "--target=auto", wav_path],
                env=_speech_environment(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            with current_audio_lock:
                current_audio_process = process
            try:
                _stdout, stderr = process.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                print("[speech error] speech playback timed out.")
                return

        # A STOP interruption terminates pw-play, so a nonzero code is expected.
        if process.returncode not in (0, -15):
            error_text = (stderr or "").strip() or f"pw-play code {process.returncode}"
            print(f"[speech warning] {error_text}")

        if PRINT_SPEECH_DEBUG:
            print("[speech complete]")

    except subprocess.TimeoutExpired:
        print("[speech error] speech generation timed out.")
    except Exception as error:
        print(f"[speech error] {type(error).__name__}: {error}")
    finally:
        if synth_proc is not None and synth_proc.poll() is None:
            try:
                synth_proc.kill()
                synth_proc.wait(timeout=1)
            except Exception:
                pass
        with current_audio_lock:
            current_audio_process = None
        if wav_path:
            try:
                os.remove(wav_path)
            except OSError:
                pass


def hazard_still_present(alert_key, risk, distance_m=None, now=None,
                         max_age=None):
    """Return True only when a queued repeat still has a fresh hazard."""
    if not alert_key:
        return True
    now = time.time() if now is None else now
    max_age = (
        HAZARD_REPEAT_PRESENT_MAX_AGE_SECONDS
        if max_age is None else max_age
    )
    with state_lock:
        mem = track_memory.get(alert_key)
        if mem is None:
            return False
        if now - mem.get("last_seen", 0.0) > max_age:
            return False
        if risk is not None and int(risk) > int(mem.get("last_risk", 0)):
            return False
    return True


def _speech_worker():
    """Speak one queued message at a time, highest-risk messages first."""
    global current_audio_protected

    while True:
        item = speech_queue.get()

        try:
            _priority, _sequence, message, protected, *origin = item
            if message is None:
                return

            origin_perf = origin[0] if len(origin) > 0 else None
            origin_label = origin[1] if len(origin) > 1 else "speech"
            alert_key = origin[2] if len(origin) > 2 else None
            alert_risk = origin[3] if len(origin) > 3 else None
            alert_distance_m = origin[4] if len(origin) > 4 else None
            alert_repeat = bool(origin[5]) if len(origin) > 5 else False

            if alert_repeat and not hazard_still_present(
                alert_key, alert_risk, alert_distance_m,
            ):
                _voice_debug(
                    "speech_repeat_dropped",
                    reason="hazard_not_present",
                    alert_key=alert_key,
                    risk=alert_risk,
                    queue=speech_queue.qsize(),
                    message=repr(message),
                )
                continue

            current_audio_protected = bool(protected)
            _voice_debug(
                "tts_dequeue",
                priority=_priority,
                protected=current_audio_protected,
                queue=speech_queue.qsize(),
                message=repr(message),
            )
            _speak_message_now(message, origin_perf, origin_label)
        except Exception as error:
            # Keep the worker alive so one bad message cannot silence every
            # future alert during a long run.
            print(f"[speech error] worker: {type(error).__name__}: {error}")
        finally:
            current_audio_protected = False
            speech_queue.task_done()


def start_speech_worker():
    global speech_worker_thread

    if (
        speech_worker_thread is not None
        and speech_worker_thread.is_alive()
    ):
        return

    speech_worker_thread = threading.Thread(
        target=_speech_worker,
        name="whisper-hat-speech-worker",
        daemon=True,
    )
    speech_worker_thread.start()

    if PRINT_SPEECH_DEBUG:
        print(
            "[speech worker started] "
            f"runtime={AUDIO_RUNTIME_DIR}"
        )


def stop_speech_worker():
    """Allow the current queued message to finish when the program closes."""
    global speech_sequence

    if speech_worker_thread is None:
        return

    try:
        with speech_sequence_lock:
            speech_sequence += 1
            sequence = speech_sequence
        speech_queue.put_nowait((999, sequence, None, False, None, None))
    except queue.Full:
        pass

    speech_worker_thread.join(timeout=3.0)


def clear_waiting_speech(keep_protected=False):
    """Drop messages that have not begun playing.

    Direct answers to a user command (OCR/scene description) are protected
    from routine repeated hazard alerts. A genuinely new/escalated STOP may
    still take priority, but the answer is not silently discarded by band flap.
    """
    kept = []
    dropped = 0
    while True:
        try:
            item = speech_queue.get_nowait()
        except queue.Empty:
            break
        speech_queue.task_done()
        if keep_protected and len(item) >= 4 and item[3]:
            kept.append(item)
        else:
            dropped += 1

    for item in kept:
        try:
            speech_queue.put_nowait(item)
        except queue.Full:
            pass

    if dropped or kept:
        global speech_cleared_count
        speech_cleared_count += dropped
        _voice_debug(
            "speech_queue_cleared",
            dropped=dropped,
            kept_protected=len(kept),
            tts_active=speech_active_event.is_set(),
        )


def interrupt_current_speech(force=False):
    """Immediately stop active pw-play so a new/urgent warning can take over."""
    if current_audio_protected and not force:
        _voice_debug("speech_interrupt_skipped", reason="direct_response_playing")
        return
    with current_audio_lock:
        process = current_audio_process
        if process is not None and process.poll() is None:
            _voice_debug("speech_interrupt", force=force)
            try:
                process.terminate()
            except Exception:
                pass


def play_alert_sound(risk, message, protected=False, urgent=None,
                     origin_perf=None, origin_label="speech",
                     priority=None, alert_key=None, alert_risk=None,
                     alert_distance_m=None, alert_repeat=False):
    """Queue speech without blocking camera and YOLO processing.

    Risk 3 receives the highest priority. Emotion announcements use risk 0,
    so they wait behind STOP, Caution, and Notice messages already queued.

    ``urgent`` defaults to True for STOP. Callers pass False for an unchanged
    repeated STOP so it cannot keep cutting off an OCR reading.
    """
    global speech_sequence

    if ALERT_SOUND_MODE == "off" or not message:
        return False

    if ALERT_SOUND_MODE != "espeak":
        print("", end="", flush=True)
        return True

    start_speech_worker()

    if urgent is None:
        urgent = risk >= 3

    _voice_debug(
        "speech_enqueue",
        risk=risk,
        protected=bool(protected),
        urgent=bool(urgent),
        tts_active=speech_active_event.is_set(),
        queue=speech_queue.qsize(),
        message=repr(message),
    )

    # A new or escalated STOP must not wait behind a face name, emotion, OCR
    # answer, notice, or caution. Routine repeats must not interrupt answers.
    if urgent:
        clear_waiting_speech(keep_protected=True)
        interrupt_current_speech(force=True)

    try:
        with speech_sequence_lock:
            speech_sequence += 1
            sequence = speech_sequence

        queue_priority = -int(risk) if priority is None else int(priority)
        speech_queue.put_nowait((
            queue_priority,
            sequence,
            message,
            bool(protected),
            origin_perf if origin_perf is not None else time.perf_counter(),
            origin_label,
            alert_key,
            risk if alert_risk is None else alert_risk,
            alert_distance_m,
            bool(alert_repeat),
        ))
        return True
    except queue.Full:
        global speech_drop_count
        speech_drop_count += 1
        print("[speech warning] queue full, dropping this message.")
        _voice_debug("speech_dropped", reason="queue_full", message=repr(message))
        return False


def _play_direct_response(message, risk=2, origin_perf=None, origin_label="command",
                          priority=None):
    """Queue an answer to a user command with routine-repeat protection."""
    try:
        return play_alert_sound(
            risk,
            message,
            protected=True,
            origin_perf=origin_perf,
            origin_label=origin_label,
            priority=priority,
        )
    except TypeError:
        # Compatibility with test/double wrappers that still accept (risk, msg).
        return play_alert_sound(risk, message)


def safety_alert_is_urgent(risk, decision=None):
    """Only a changed STOP interrupts an active answer.

    A new, escalated, or materially closer hazard interrupts immediately. An
    unchanged repeat of the same STOP queues normally, so a stationary book
    does not repeatedly cut off its own reading.
    """
    decision = decision or {}
    return bool(
        int(risk) >= 3
        and (
            decision.get("new")
            or decision.get("escalated")
            or decision.get("materially_closer")
        )
    )


def _play_safety_alert(risk, message, urgent, origin_perf=None,
                       alert_key=None, alert_distance_m=None, repeat=False):
    """Queue a hazard alert; only a changed STOP is urgent."""
    try:
        return play_alert_sound(
            risk,
            message,
            urgent=urgent,
            origin_perf=origin_perf,
            origin_label="hazard",
            alert_key=alert_key,
            alert_risk=risk,
            alert_distance_m=alert_distance_m,
            alert_repeat=repeat,
        )
    except TypeError:
        # Compatibility with test/double wrappers that still accept (risk, msg).
        return play_alert_sound(risk, message)

# ============================================================
# USB MICROPHONE VOICE CONTROL
# ============================================================

@contextmanager
def suppress_audio_stderr():
    """Temporarily silence native ALSA/JACK probing messages.

    PyAudio can print many harmless lines such as "Unknown PCM surround51"
    directly to file descriptor 2.  We use this only while constructing/opening
    the microphone, so normal Python/camera/ToF errors remain visible.
    """
    try:
        sys.stderr.flush()
    except Exception:
        pass

    stderr_fd = 2
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


@contextmanager
def open_microphone_quietly(microphone):
    """Open/close a SpeechRecognition microphone without ALSA/JACK spam."""
    source = None
    with suppress_audio_stderr():
        source = microphone.__enter__()
    try:
        yield source
    finally:
        with suppress_audio_stderr():
            microphone.__exit__(None, None, None)


def detection_is_enabled():
    return detection_enabled_event.is_set()


def set_detection_enabled(enabled, announce=True):
    """Turn all hazard/object detection processing on or off."""
    with state_lock:
        changed = detection_is_enabled() != bool(enabled)

        if enabled:
            detection_enabled_event.set()
        else:
            detection_enabled_event.clear()
            # Serialize OFF cancellation with on-demand scene speech enqueueing.
            clear_waiting_speech()
            interrupt_current_speech(force=True)
            track_memory.clear()
            distance_memory.clear()
            track_history.clear()
            face_memory.clear()
            emotion_history.clear()
            last_emotion_print.clear()
            emotion_speech_memory.clear()

    if changed:
        state = "ON" if enabled else "OFF"
        print(f"[voice control] Detection {state}.")
        if announce and VOICE_CONFIRM_COMMANDS:
            play_alert_sound(2 if enabled else 1, f"Detection {state.lower()}.")

    return changed


def _select_voice_microphone_index():
    """Return the configured USB microphone index without scanning all ALSA devices."""
    if sr is None:
        return None

    if VOICE_MIC_DEVICE_INDEX is not None:
        index = int(VOICE_MIC_DEVICE_INDEX)
        print(f"[voice control] Using configured USB microphone device {index}.", flush=True)
        return index

    print("[voice control] No microphone index configured; using default input.")
    return None


def _voice_microphone_name(index):
    """Best-effort human-readable device name for diagnostics."""
    if sr is None:
        return "unknown"
    try:
        with suppress_audio_stderr():
            names = sr.Microphone.list_microphone_names()
        if not names:
            return "unknown"
        if index is None:
            return f"default ({len(names)} input devices)"
        if 0 <= int(index) < len(names):
            return names[int(index)]
        return f"index {index} out of range ({len(names)} devices)"
    except Exception as error:
        return f"unavailable ({type(error).__name__})"


def _open_voice_source(mic_index):
    """Open the mic, trying fallback capture rates before giving up.

    A USB mic can open successfully at a requested rate that the hardware does
    not actually run at, which makes Vosk decode shifted speech. Logging the
    source rate makes that visible. The fallback list only applies when opening
    raises, so it never changes a working baseline configuration.
    """
    rates = []
    for rate in (VOICE_MIC_SAMPLE_RATE, *VOICE_MIC_FALLBACK_RATES):
        rate = int(rate)
        if rate > 0 and rate not in rates:
            rates.append(rate)

    errors = []
    for rate in rates:
        microphone = None
        try:
            with suppress_audio_stderr():
                microphone = sr.Microphone(
                    device_index=mic_index,
                    sample_rate=rate,
                )
                source = microphone.__enter__()
            name = _voice_microphone_name(mic_index)
            print(
                f"[voice control] Microphone open: index={mic_index} "
                f"name={name!r} requested={rate}Hz "
                f"source={source.SAMPLE_RATE}Hz "
                f"chunk={source.CHUNK} width={source.SAMPLE_WIDTH}",
                flush=True,
            )
            _voice_debug(
                "mic_open",
                index=mic_index,
                rate=rate,
                source_rate=source.SAMPLE_RATE,
                chunk=source.CHUNK,
                sample_width=source.SAMPLE_WIDTH,
            )
            return source, microphone, rate
        except Exception as error:
            errors.append(f"{rate}Hz: {type(error).__name__}: {error}")
            if microphone is not None:
                try:
                    with suppress_audio_stderr():
                        microphone.__exit__(None, None, None)
                except Exception:
                    pass
            _voice_debug("mic_open_failed", index=mic_index, rate=rate, error=error)

    raise RuntimeError("; ".join(errors) or "no capture rate available")


def _normalize_voice_command(text):
    # Remove punctuation/apostrophes so "what's" and "whats" match the same
    # canonical command, and so Vosk's punctuation variants do not get dropped.
    cleaned = str(text or "").lower().replace("'", "").replace("’", "")
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    return " ".join(cleaned.split())


def _antialias_lowpass(samples, rate, target_rate, num_taps=64):
    """Windowed-sinc FIR low-pass at the output Nyquist (numpy-only, no scipy).

    Downsampling without band-limiting folds everything above target_rate/2 back
    into the speech band as aliasing, which hurts Vosk. This removes it first.
    Unity DC gain keeps in-band level unchanged; mode='same' preserves length and
    alignment. Chunks shorter than the kernel are returned untouched so a tiny
    buffer never grows to the kernel length.
    """
    if samples.size < num_taps:
        return samples
    fc = (target_rate / 2.0) / rate
    center = (num_taps - 1) / 2.0
    n = np.arange(num_taps, dtype=np.float64) - center
    kernel = 2.0 * fc * np.sinc(2.0 * fc * n)
    kernel *= np.hamming(num_taps)
    kernel /= kernel.sum()
    return np.convolve(samples, kernel.astype(np.float32), mode="same")


def _convert_audio_to_16khz_mono_int16(audio):
    """Convert SpeechRecognition AudioData to 16 kHz 16-bit mono PCM without audioop."""
    raw = audio.frame_data
    width = int(audio.sample_width)
    rate = int(audio.sample_rate)

    if rate == VOICE_RECOGNITION_SAMPLE_RATE and width == 2:
        return raw

    if width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    elif width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) * 256.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 65536.0
    else:
        return audio.get_raw_data(
            convert_rate=VOICE_RECOGNITION_SAMPLE_RATE,
            convert_width=2,
        )

    if samples.size == 0:
        return b""

    if rate != VOICE_RECOGNITION_SAMPLE_RATE:
        if rate > VOICE_RECOGNITION_SAMPLE_RATE:
            samples = _antialias_lowpass(samples, rate, VOICE_RECOGNITION_SAMPLE_RATE)
        out_len = max(1, int(round(samples.size * VOICE_RECOGNITION_SAMPLE_RATE / rate)))
        x_old = np.arange(samples.size, dtype=np.float32)
        x_new = np.linspace(0.0, float(samples.size - 1), out_len, dtype=np.float32)
        samples = np.interp(x_new, x_old, samples)

    return np.clip(np.rint(samples), -32768, 32767).astype(np.int16).tobytes()


def _compose_scene_description(objects):
    if not objects:
        return "I don't see anything ahead."
    positions = {
        "in path": "ahead", "ahead": "ahead",
        "left side": "to your left", "left": "to your left",
        "right side": "to your right", "right": "to your right",
    }
    distances = {
        "very_close": "very close", "close": "close",
        "medium": "a few meters away", "far": "far away",
    }
    ordered = sorted(objects, key=lambda item: (-item[4], item[3]))
    return " ".join(
        f"{class_name.capitalize()} {positions[position]}, {distances[level]}."
        for class_name, position, level, distance_m, risk
        in ordered[:SCENE_DESCRIBE_MAX_OBJECTS]
    )


def _trigger_scene_description():
    global last_scene_describe_time, scene_last_trigger_monotonic, voice_scene_triggers
    if not ENABLE_SCENE_DESCRIBE:
        return False
    now = time.time()
    if now - last_scene_describe_time < SCENE_DESCRIBE_COOLDOWN_SECONDS:
        return True
    last_scene_describe_time = now
    scene_last_trigger_monotonic = time.monotonic()
    voice_scene_triggers += 1
    scene_describe_event.set()
    return True


def _enabled_phrases(canonical, aliases, enabled=True):
    if not enabled:
        return []
    return list(dict.fromkeys((*canonical, *aliases)))


def _voice_command_grammar():
    phrases = _enabled_phrases(
        ("detection on",),
        VOICE_DETECTION_ON_ALIASES,
    )
    phrases.extend(_enabled_phrases(
        ("detection off",),
        VOICE_DETECTION_OFF_ALIASES,
    ))
    phrases.extend(_enabled_phrases(
        OCR_READ_PHRASES,
        OCR_READ_ALIASES,
        ENABLE_OCR_READING,
    ))
    phrases.extend(_enabled_phrases(
        SCENE_DESCRIBE_PHRASES,
        SCENE_DESCRIBE_ALIASES,
        ENABLE_SCENE_DESCRIBE,
    ))
    # Keep the grammar unique; duplicate phrases add no accuracy and can make
    # the small model slower to decode.
    phrases = list(dict.fromkeys(phrases))
    phrases.append("[unk]")
    return json.dumps(phrases)


def _voice_command_map():
    """Map normalized recognized text to a canonical command."""
    mapping = {}

    for alias in ("detection on", *VOICE_DETECTION_ON_ALIASES):
        mapping[_normalize_voice_command(alias)] = "detection on"
    for alias in ("detection off", *VOICE_DETECTION_OFF_ALIASES):
        mapping[_normalize_voice_command(alias)] = "detection off"

    if ENABLE_OCR_READING:
        for phrase in (*OCR_READ_PHRASES, *OCR_READ_ALIASES):
            mapping[_normalize_voice_command(phrase)] = "read this"

    if ENABLE_SCENE_DESCRIBE:
        for phrase in (*SCENE_DESCRIBE_PHRASES, *SCENE_DESCRIBE_ALIASES):
            mapping[_normalize_voice_command(phrase)] = "what's there"

    return mapping


def _match_voice_command(text):
    """Return a canonical command, or None when the text is not a command."""
    command = _voice_command_map().get(_normalize_voice_command(text))
    if command is not None:
        return command

    # A small grammar sometimes returns only the leading verb. Route a bare
    # "read" to OCR (it is an existing canonical phrase) but do not fuzzy-match
    # arbitrary speech.
    if ENABLE_OCR_READING and _normalize_voice_command(text) == "read":
        return "read this"
    return None


def _apply_voice_command(text):
    global last_voice_command_time, voice_commands_matched, voice_commands_dispatched

    command = _match_voice_command(text)
    if command is None:
        _voice_debug(
            "command_ignored",
            text=repr(text),
            tts_active=speech_active_event.is_set(),
        )
        return False

    voice_commands_matched += 1
    voice_commands_dispatched += 1
    _voice_heartbeat("command_routed")
    _log_voice_status(force=True)

    if ENABLE_SCENE_DESCRIBE and command == "what's there":
        _voice_debug("command_routed", command="scene_describe", text=repr(text))
        return _trigger_scene_description()

    if ENABLE_OCR_READING and command == "read this":
        _voice_debug("command_routed", command="ocr_read", text=repr(text))
        _trigger_ocr_reading()
        return True

    now = time.time()
    if now - last_voice_command_time < VOICE_COMMAND_COOLDOWN_SECONDS:
        _voice_debug(
            "command_debounced",
            command=command,
            remaining_s=round(VOICE_COMMAND_COOLDOWN_SECONDS - (now - last_voice_command_time), 3),
        )
        return True
    last_voice_command_time = now

    _voice_debug("command_routed", command=command, text=repr(text))
    if command == "detection on":
        set_detection_enabled(True, announce=True)
    else:
        set_detection_enabled(False, announce=True)
    return True


def _voice_playback_cooldown_remaining():
    """Seconds left in the optional post-TTS settle window (0 when disabled)."""
    cooldown = max(0.0, float(VOICE_PLAYBACK_COOLDOWN_SECONDS))
    if cooldown <= 0.0 or speech_active_event.is_set():
        return 0.0
    if speech_last_end_time <= 0.0:
        return 0.0
    return max(0.0, cooldown - (time.time() - speech_last_end_time))


def _voice_control_worker():
    """Listen for offline Vosk commands with stage-by-stage diagnostics."""
    global voice_audio_frames, voice_audio_bytes, voice_audio_timeouts
    global voice_nonempty_results, voice_last_audio_monotonic
    global voice_last_processed_monotonic

    if sr is None:
        print(
            "[voice control] SpeechRecognition is not installed. "
            "Install it with: pip install SpeechRecognition"
        )
        return

    if VoskModel is None or KaldiRecognizer is None:
        print(
            "[voice control] Vosk is not installed. "
            "Install it with: pip install vosk"
        )
        return

    model_path = Path(VOICE_VOSK_MODEL_PATH)
    if not model_path.exists():
        print(f"[voice control] Vosk model folder not found: {model_path}")
        print(
            "[voice control] Download a small English Vosk model once, extract it "
            "beside this script, then voice control works completely offline."
        )
        return

    try:
        vosk_model = VoskModel(str(model_path))
    except Exception as error:
        print(f"[voice control] Could not load Vosk model: {error}")
        return

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    mic_index = _select_voice_microphone_index()

    # Restrict decoding to exactly the phrases this project needs. This makes
    # a small offline model both faster and less prone to accidental commands.
    command_grammar = _voice_command_grammar()
    _voice_debug(
        "grammar_ready",
        phrases=len(json.loads(command_grammar)) - 1,
        tts_active=speech_active_event.is_set(),
    )

    while not voice_control_stop_event.is_set():
        source = None
        microphone = None
        _set_voice_listener_source(None, None)
        _voice_heartbeat("opening")
        try:
            source, microphone, _opened_rate = _open_voice_source(mic_index)
            _set_voice_listener_source(source, microphone)
            _voice_heartbeat("opened")
            print(
                "[voice control] Calibrating USB microphone for ambient noise...",
                flush=True,
            )
            calibrate_started = time.monotonic()
            _voice_heartbeat("calibrating")
            recognizer.adjust_for_ambient_noise(
                source,
                duration=VOICE_AMBIENT_CALIBRATION_SECONDS,
            )
            _voice_debug(
                "ambient_calibrated",
                seconds=round(time.monotonic() - calibrate_started, 3),
                energy_threshold=round(recognizer.energy_threshold, 1),
            )
            if VOICE_PRINT_DEBUG:
                print(f"[voice control] Energy threshold: {recognizer.energy_threshold:.1f}")

            print(
                '[voice control] OFFLINE Vosk ready. Say "detection off" or "detection on".',
                flush=True,
            )
            _voice_heartbeat("listening")

            # Keep the ALSA/PipeWire capture stream open across listen cycles
            # instead of reopening PyAudio (and duplicating stderr FDs) every second.
            while not voice_control_stop_event.is_set():
                _voice_heartbeat("listening")
                cooldown = _voice_playback_cooldown_remaining()
                if cooldown > 0.0:
                    _voice_debug(
                        "playback_cooldown_wait",
                        seconds=round(cooldown, 3),
                        tts_active=speech_active_event.is_set(),
                    )
                    voice_control_stop_event.wait(cooldown)
                    if voice_control_stop_event.is_set():
                        break

                listen_started = time.monotonic()
                stream = getattr(source, "stream", None)
                try:
                    stream_active = bool(stream.is_active())
                except Exception:
                    stream_active = None
                _voice_debug(
                    "listen_wait",
                    timeout_s=VOICE_LISTEN_TIMEOUT,
                    phrase_limit_s=VOICE_PHRASE_TIME_LIMIT,
                    tts_active=speech_active_event.is_set(),
                    queue=speech_queue.qsize(),
                    stream_active=stream_active,
                )
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=VOICE_LISTEN_TIMEOUT,
                        phrase_time_limit=VOICE_PHRASE_TIME_LIMIT,
                    )
                except sr.WaitTimeoutError:
                    voice_audio_timeouts += 1
                    _voice_heartbeat("listening")
                    _voice_debug(
                        "listen_timeout",
                        elapsed_s=round(time.monotonic() - listen_started, 3),
                        tts_active=speech_active_event.is_set(),
                    )
                    continue

                _voice_debug(
                    "listen_return",
                    elapsed_s=round(time.monotonic() - listen_started, 3),
                    tts_active=speech_active_event.is_set(),
                )
                tts_overlap = speech_active_event.is_set()
                voice_audio_frames += 1
                voice_audio_bytes += len(audio.frame_data)
                voice_last_audio_monotonic = time.monotonic()
                _voice_heartbeat("processing")
                duration_s = 0.0
                try:
                    duration_s = len(audio.frame_data) / max(
                        1, int(audio.sample_width) * int(audio.sample_rate)
                    )
                except Exception:
                    pass
                _voice_debug(
                    "audio_captured",
                    bytes=len(audio.frame_data),
                    seconds=round(duration_s, 3),
                    source_rate=audio.sample_rate,
                    tts_overlap=tts_overlap,
                )

                # Convert the captured audio locally to the PCM format Vosk expects.
                raw_audio = _convert_audio_to_16khz_mono_int16(audio)
                if not raw_audio:
                    _voice_debug("audio_dropped", reason="empty_after_conversion")
                    continue

                offline_recognizer = KaldiRecognizer(
                    vosk_model,
                    VOICE_RECOGNITION_SAMPLE_RATE,
                    command_grammar,
                )
                accepted = offline_recognizer.AcceptWaveform(raw_audio)
                partial = json.loads(offline_recognizer.PartialResult())
                result = json.loads(offline_recognizer.FinalResult())
                text = _normalize_voice_command(result.get("text", ""))
                voice_nonempty_results += int(bool(text))
                voice_last_processed_monotonic = time.monotonic()
                _voice_heartbeat("listening")
                _voice_debug(
                    "vosk_result",
                    accepted=bool(accepted),
                    partial=repr(partial.get("partial", "")),
                    final=repr(result.get("text", "")),
                    tts_overlap=tts_overlap,
                )

                if VOICE_PRINT_DEBUG and text:
                    print(f"[voice heard offline] {text}")

                if not text:
                    reason = "tts_overlap" if tts_overlap else "no_speech_in_grammar"
                    _voice_debug("command_ignored", reason=reason, tts_overlap=tts_overlap)
                    continue

                if tts_overlap:
                    print(
                        "[voice notice] Speech was captured while TTS was active; "
                        "retest with the speaker muted before changing the recognizer.",
                        flush=True,
                    )

                _apply_voice_command(text)

        except Exception as error:
            if voice_control_stop_event.is_set():
                break
            _voice_heartbeat("error")
            print(f"[voice control] Offline microphone/recognition error: {error}")
            if source is None:
                print(
                    "[voice control] On Raspberry Pi, install microphone support with: "
                    "sudo apt install -y portaudio19-dev python3-pyaudio flac"
                )
            _voice_debug("worker_error", error=f"{type(error).__name__}: {error}")
            voice_control_stop_event.wait(1.0)
        finally:
            _set_voice_listener_source(None, None)
            if microphone is not None:
                try:
                    with suppress_audio_stderr():
                        microphone.__exit__(None, None, None)
                except Exception:
                    pass
            _voice_heartbeat(
                "stopped" if voice_control_stop_event.is_set() else "reopening"
            )
            _voice_debug("mic_closed", source_open=source is not None)



def _force_unblock_voice_stream():
    """Close the current capture stream so a blocked native read can return."""
    with voice_listener_source_lock:
        source = voice_listener_source
        microphone = voice_listener_microphone

    if source is None and microphone is None:
        _voice_debug("recovery_skipped", reason="no_listener_source")
        return False

    stream = getattr(source, "stream", None)
    closed = False
    if stream is not None:
        for method_name in ("stop_stream", "close"):
            method = getattr(stream, method_name, None)
            if not callable(method):
                continue
            try:
                method()
                closed = True
            except Exception as error:
                _voice_debug(
                    "recovery_stream_error",
                    method=method_name,
                    error=f"{type(error).__name__}: {error}",
                )

    if microphone is not None:
        try:
            with suppress_audio_stderr():
                microphone.__exit__(None, None, None)
            closed = True
        except Exception as error:
            _voice_debug(
                "recovery_mic_error",
                error=f"{type(error).__name__}: {error}",
            )
    return closed


def _voice_watchdog_check(now=None):
    """Recover a dead worker or a stream that stopped making progress.

    Silence is not a failure: the normal 1 s listen timeout updates the
    heartbeat. Recovery only runs when a real stage heartbeat is stale or the
    worker thread is no longer alive. Retries are rate-limited.
    """
    global voice_recovery_count, voice_last_recovery_monotonic
    global current_audio_protected

    if not VOICE_WATCHDOG_ENABLED or voice_control_stop_event.is_set():
        return False

    now = time.monotonic() if now is None else now
    thread = voice_control_thread
    if thread is None:
        return False

    if not thread.is_alive():
        if now - voice_last_recovery_monotonic < VOICE_RECOVERY_MIN_INTERVAL_SECONDS:
            _voice_debug(
                "recovery_rate_limited",
                reason="worker_not_alive",
                remaining_s=round(
                    VOICE_RECOVERY_MIN_INTERVAL_SECONDS
                    - (now - voice_last_recovery_monotonic),
                    1,
                ),
            )
            return False
        if voice_control_stop_event.is_set():
            return False
        print(
            "[voice watchdog] worker thread is not alive; restarting listener.",
            flush=True,
        )
        voice_last_recovery_monotonic = now
        voice_recovery_count += 1
        _voice_heartbeat("restarting")
        start_voice_control(start_watchdog=False)
        return True

    active_stages = {
        "opening", "opened", "calibrating", "listening", "processing",
        "recovering", "reopening",
    }
    heartbeat_age = now - voice_listener_heartbeat
    if voice_listener_stage in active_stages and heartbeat_age > VOICE_WATCHDOG_STALE_SECONDS:
        if now - voice_last_recovery_monotonic < VOICE_RECOVERY_MIN_INTERVAL_SECONDS:
            _voice_debug(
                "recovery_rate_limited",
                reason="stale_stream",
                stage=voice_listener_stage,
                age_s=round(heartbeat_age, 1),
                remaining_s=round(
                    VOICE_RECOVERY_MIN_INTERVAL_SECONDS
                    - (now - voice_last_recovery_monotonic),
                    1,
                ),
            )
            return False

        last_audio_age = (
            None if voice_last_audio_monotonic <= 0.0
            else round(now - voice_last_audio_monotonic, 1)
        )
        print(
            "[voice watchdog] no listener progress for "
            f"{heartbeat_age:.1f}s at stage={voice_listener_stage} "
            f"tts_active={speech_active_event.is_set()} "
            f"last_audio_age_s={last_audio_age}; "
            "closing the capture stream to unblock the read.",
            flush=True,
        )
        voice_last_recovery_monotonic = now
        voice_recovery_count += 1
        _voice_heartbeat("recovering")
        _force_unblock_voice_stream()
        return True

    # A process can exit without the wrapper clearing the event if a native
    # call is interrupted. Do not clear a genuinely active playback.
    if speech_active_event.is_set() and speech_last_start_time > 0.0:
        playback_age = time.time() - speech_last_start_time
        process = current_audio_process
        if playback_age > VOICE_STALE_PLAYBACK_SECONDS and (
            process is None or process.poll() is not None
        ):
            print(
                "[voice watchdog] clearing stale playback suppression "
                f"after {playback_age:.1f}s.",
                flush=True,
            )
            current_audio_protected = False
            _set_speech_active(False)
            return True

    return False


def _voice_watchdog_worker():
    """Run the listener watchdog until voice control is stopped."""
    while not voice_control_stop_event.is_set():
        try:
            _voice_watchdog_check()
        except Exception as error:
            print(
                f"[voice watchdog] check error: {type(error).__name__}: {error}",
                flush=True,
            )
        voice_control_stop_event.wait(VOICE_WATCHDOG_INTERVAL_SECONDS)


def _voice_control_worker_safe():
    """Run voice control and make any unexpected thread failure visible."""
    try:
        _voice_control_worker()
    except Exception as error:
        print(
            f"[voice control] Worker crashed: {type(error).__name__}: {error}",
            flush=True,
        )


def start_voice_control(start_watchdog=True):
    global voice_control_thread, voice_watchdog_thread

    if not ENABLE_VOICE_CONTROL:
        print("[voice control] Disabled in settings.", flush=True)
        return

    with voice_control_lifecycle_lock:
        if voice_control_thread is not None and voice_control_thread.is_alive():
            print("[voice control] Worker is already running.", flush=True)
            return

        # Print from the MAIN thread before launching the worker. This guarantees
        # visible confirmation that the voice startup path was reached.
        print(
            f"[voice control] Starting USB voice control on device "
            f"{VOICE_MIC_DEVICE_INDEX} at {VOICE_MIC_SAMPLE_RATE} Hz...",
            flush=True,
        )

        voice_control_stop_event.clear()
        voice_control_thread = threading.Thread(
            target=_voice_control_worker_safe,
            name="whisper-hat-voice-control",
            daemon=True,
        )
        voice_control_thread.start()

        if start_watchdog and (
            voice_watchdog_thread is None or not voice_watchdog_thread.is_alive()
        ):
            voice_watchdog_thread = threading.Thread(
                target=_voice_watchdog_worker,
                name="whisper-hat-voice-watchdog",
                daemon=True,
            )
            voice_watchdog_thread.start()


def stop_voice_control():
    global voice_listener_stage
    voice_control_stop_event.set()
    if voice_control_thread is not None and voice_control_thread.is_alive():
        voice_control_thread.join(timeout=3.0)
    if voice_watchdog_thread is not None and voice_watchdog_thread.is_alive():
        voice_watchdog_thread.join(timeout=2.0)
    voice_listener_stage = "stopped"


# ============================================================
# VOICE-ACTIVATED OCR READING
# ============================================================

def _ocr_available():
    """True when the Tesseract binary is on PATH (cached after the first check)."""
    global _ocr_available_cache, _ocr_missing_warned
    if _ocr_available_cache is None:
        _ocr_available_cache = shutil.which(OCR_BINARY) is not None
        if not _ocr_available_cache and not _ocr_missing_warned:
            _ocr_missing_warned = True
            print(
                f"[ocr] '{OCR_BINARY}' not found; text reading is unavailable. "
                "Install it with: sudo apt install -y tesseract-ocr",
                flush=True,
            )
    return _ocr_available_cache


def _clean_ocr_text(text):
    """Collapse whitespace and cap spoken length at a whole-word boundary."""
    words = text.split()
    if not words:
        return ""
    cleaned = " ".join(words)
    if len(cleaned) <= OCR_MAX_SPEECH_CHARS:
        return cleaned
    kept = []
    length = 0
    for word in words:
        added = len(word) + (1 if kept else 0)
        if length + added > OCR_MAX_SPEECH_CHARS:
            break
        kept.append(word)
        length += added
    if not kept:
        return cleaned[:OCR_MAX_SPEECH_CHARS]
    return " ".join(kept)


def _split_ocr_speech_chunks(text):
    """Split readable OCR text into short, speakable chunks.

    The first chunk is deliberately short so the user hears useful content
    before a long OCR block can occupy the speech worker. Cuts prefer sentence
    punctuation, then whitespace; never split inside a word.
    """
    remaining = _clean_ocr_text(text)
    if not remaining:
        return []

    chunks = []
    limit = max(40, int(OCR_SPEECH_CHUNK_CHARS))
    for _ in range(max(1, int(OCR_MAX_SPEECH_CHUNKS))):
        if not remaining:
            break
        if len(remaining) <= limit:
            chunks.append(remaining.strip())
            break

        window = remaining[: limit + 1]
        cut = max(
            window.rfind(". "),
            window.rfind("? "),
            window.rfind("! "),
            window.rfind("; "),
            window.rfind(", "),
        )
        if cut < int(limit * 0.5):
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        else:
            cut += 1
        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].strip()

    return [chunk for chunk in chunks if chunk]


def _prepare_ocr_frame(frame):
    """Resize a still/read frame down to the OCR working width."""
    if frame is None:
        return None
    height, width = frame.shape[:2]
    if width <= OCR_MAX_WIDTH or width <= 0:
        return frame
    scale = OCR_MAX_WIDTH / float(width)
    return cv2.resize(
        frame,
        (OCR_MAX_WIDTH, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _grab_ocr_frame(source):
    """Prefer one high-resolution still, then peek/read the detection frame."""
    global ocr_frame_origin
    ocr_frame_origin = "none"
    if source is None:
        return None

    if OCR_USE_STILL:
        capture_still = getattr(source, "capture_still", None)
        if callable(capture_still):
            still_started = time.monotonic()
            try:
                frame = capture_still()
            except Exception as error:
                _voice_debug(
                    "ocr_still_failed",
                    elapsed_s=round(time.monotonic() - still_started, 3),
                    error=f"{type(error).__name__}: {error}",
                )
            else:
                if frame is not None:
                    ocr_frame_origin = "still"
                    _voice_debug(
                        "ocr_still_captured",
                        elapsed_s=round(time.monotonic() - still_started, 3),
                        shape=getattr(frame, "shape", None),
                    )
                    return frame

    peek = getattr(source, "peek_latest", None)
    if callable(peek):
        frame = peek()
        if frame is not None:
            ocr_frame_origin = "peek"
            return frame

    ret, frame = source.read()
    ocr_frame_origin = "read" if ret and frame is not None else "none"
    return frame if ret else None


def _ocr_frame_metrics(frame):
    """Cheap focus/brightness diagnostics for a real book-page test."""
    if frame is None:
        return {}
    gray = (
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3 else frame
    )
    return {
        "blur_score": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "brightness": round(float(gray.mean()), 1),
        "contrast": round(float(gray.std()), 1),
    }


# A small function-word list is enough to separate real prose from the
# letter-shaped noise seen on the Pi. It is deliberately not a full dictionary:
# technical pages often contain names, numbers, and jargon, but prose still
# contains "the", "and", "of", "to", "in", and similar words.
OCR_COMMON_WORDS = frozenset("""
a about after again all also am an and any are as at be because been before
being between both but by can could did do does doing down each even first for
from further get give go had has have having he her here hers herself him
himself his how i if in into is it its itself just know like make me more most
my myself new no nor not now of off on once only or other our ours ourselves
out over own same say see she should so some such than that the their theirs
them themselves then there these they this those through to too under until up
us very was we were what when where which while who whom why will with would
you your yours yourself yourselves
""".split())


def _longest_consonant_run(word):
    vowels = set("aeiouy")
    longest = current = 0
    for char in word.lower():
        if char.isalpha() and char not in vowels:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _ocr_text_metrics(text, words):
    """Confidence and language-likeness metrics for the OCR quality gate."""
    tokens = [str(word).strip() for word, _conf in words if str(word).strip()]
    confidences = [
        float(conf) for _word, conf in words
        if conf is not None and float(conf) >= 0.0
    ]
    word_count = len(tokens)
    visible = [char for char in text if not char.isspace()]
    alpha_ratio = (
        sum(char.isalpha() for char in visible) / len(visible)
        if visible else 0.0
    )
    vowel_words = sum(
        1 for word in tokens
        if len([char for char in word if char.isalpha()]) >= 2
        and any(char.lower() in "aeiouy" for char in word if char.isalpha())
    )
    long_consonant_words = sum(
        1 for word in tokens if _longest_consonant_run(word) >= 6
    )
    normalized = [
        "".join(char.lower() for char in word if char.isalpha())
        for word in tokens
    ]
    common_words = sum(
        1 for word in normalized if word in OCR_COMMON_WORDS
    )
    return {
        "word_count": word_count,
        "mean_conf": (
            sum(confidences) / len(confidences) if confidences else 0.0
        ),
        "good_word_ratio": (
            sum(conf >= OCR_QUALITY_GOOD_CONF for conf in confidences)
            / len(confidences)
            if confidences else 0.0
        ),
        "alpha_ratio": alpha_ratio,
        "vowel_word_ratio": vowel_words / word_count if word_count else 0.0,
        "long_consonant_ratio": (
            long_consonant_words / word_count if word_count else 0.0
        ),
        "common_word_ratio": (
            common_words / word_count if word_count else 0.0
        ),
    }


def _ocr_quality_ok(metrics):
    """True only when the OCR result is plausibly readable text."""
    if not OCR_QUALITY_CHECK:
        return True
    word_count = int(metrics.get("word_count", 0))
    if word_count <= 0:
        return False

    mean_conf = float(metrics.get("mean_conf", 0.0))
    alpha_ratio = float(metrics.get("alpha_ratio", 0.0))
    vowel_ratio = float(metrics.get("vowel_word_ratio", 0.0))
    long_consonant_ratio = float(metrics.get("long_consonant_ratio", 0.0))
    common_word_ratio = float(metrics.get("common_word_ratio", 0.0))

    if (
        word_count >= OCR_QUALITY_COMMON_WORD_MIN_COUNT
        and common_word_ratio < OCR_QUALITY_MIN_COMMON_WORD_RATIO
    ):
        return False

    if word_count < OCR_QUALITY_MIN_WORDS_FOR_CONFIDENCE:
        return (
            mean_conf >= OCR_QUALITY_MIN_SINGLE_WORD_CONF
            and alpha_ratio >= OCR_QUALITY_MIN_ALPHA_RATIO
            and vowel_ratio >= OCR_QUALITY_MIN_VOWEL_WORD_RATIO
            and long_consonant_ratio <= OCR_QUALITY_MAX_LONG_CONSONANT_RATIO
        )

    return (
        mean_conf >= OCR_QUALITY_MIN_MEAN_CONF
        and float(metrics.get("good_word_ratio", 0.0))
        >= OCR_QUALITY_MIN_GOOD_WORD_RATIO
        and alpha_ratio >= OCR_QUALITY_MIN_ALPHA_RATIO
        and vowel_ratio >= OCR_QUALITY_MIN_VOWEL_WORD_RATIO
        and long_consonant_ratio <= OCR_QUALITY_MAX_LONG_CONSONANT_RATIO
    )


def _parse_tesseract_tsv(output):
    """Parse Tesseract TSV into text, word confidences, and metrics."""
    lines = []
    word_rows = []
    text_left = text_top = float("inf")
    text_right = text_bottom = 0.0
    for row in output.splitlines()[1:]:
        parts = row.split("\t")
        if len(parts) < 12:
            continue
        word = parts[11].strip()
        if not word:
            continue
        try:
            confidence = float(parts[10])
            left = float(parts[6])
            top = float(parts[7])
            width = float(parts[8])
            height = float(parts[9])
        except ValueError:
            continue
        if confidence < 0:
            continue
        key = (parts[1], parts[2], parts[3], parts[4])
        if not lines or lines[-1][0] != key:
            lines.append((key, []))
        lines[-1][1].append(word)
        word_rows.append((word, confidence))
        text_left = min(text_left, left)
        text_top = min(text_top, top)
        text_right = max(text_right, left + width)
        text_bottom = max(text_bottom, top + height)

    text = "\n".join(" ".join(words) for _key, words in lines)
    metrics = _ocr_text_metrics(text, word_rows)
    metrics["text_box"] = (
        None if not word_rows
        else (text_left, text_top, text_right, text_bottom)
    )
    return text, word_rows, metrics


def _run_tesseract_command(frame, extra_args=None):
    """Encode the frame as PNG and pipe it through Tesseract."""
    global current_ocr_process
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("could not encode frame for OCR")
    png_bytes = buffer.tobytes()

    command = [
        OCR_BINARY, "stdin", "stdout",
        "-l", OCR_LANGUAGE,
        "--psm", str(OCR_PSM),
    ]
    if extra_args:
        command.extend(extra_args)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with current_ocr_process_lock:
        current_ocr_process = process
    try:
        try:
            stdout, _stderr = process.communicate(
                input=png_bytes, timeout=OCR_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        if process.returncode != 0:
            raise RuntimeError(f"tesseract exited with code {process.returncode}")
        return stdout.decode("utf-8", errors="replace")
    finally:
        with current_ocr_process_lock:
            current_ocr_process = None


def _run_tesseract(frame):
    """Plain-text Tesseract call retained for the benchmark/backward tests."""
    return _run_tesseract_command(frame)


def _run_tesseract_tsv(frame):
    """TSV Tesseract call with word confidences for the quality gate."""
    output = _run_tesseract_command(frame, ["tsv"])
    return _parse_tesseract_tsv(output)


def _perform_ocr_read(source):
    """Grab one frame, read it, and speak the result at caution priority (2).

    Risk 2 never interrupts a STOP (risk 3); it queues behind urgent alerts.
    The answer is protected from routine repeated hazard alerts so a user
    command is not silently dropped by band-flap chatter.
    """
    started = time.monotonic()
    _voice_debug(
        "ocr_start",
        elapsed_s=round(started - ocr_last_trigger_monotonic, 3) if ocr_last_trigger_monotonic else None,
        tts_active=speech_active_event.is_set(),
        source=type(source).__name__ if source is not None else None,
    )

    if not _ocr_available():
        _play_direct_response(
            "Text reading isn't available.",
            origin_perf=ocr_last_trigger_monotonic,
        )
        return

    frame = _grab_ocr_frame(source)
    original_shape = None if frame is None else frame.shape
    original_metrics = _ocr_frame_metrics(frame)
    frame = _prepare_ocr_frame(frame)
    _voice_debug(
        "ocr_frame",
        elapsed_s=round(time.monotonic() - started, 3),
        frame_origin=ocr_frame_origin,
        original_shape=original_shape,
        ocr_shape=None if frame is None else frame.shape,
        original_blur_score=original_metrics.get("blur_score"),
        original_brightness=original_metrics.get("brightness"),
        original_contrast=original_metrics.get("contrast"),
        **_ocr_frame_metrics(frame),
    )
    if frame is None:
        _play_direct_response(
            "I can't see right now.",
            origin_perf=ocr_last_trigger_monotonic,
        )
        return

    try:
        raw_text, words, metrics = _run_tesseract_tsv(frame)
    except Exception as error:
        _voice_debug("ocr_failed", error=f"{type(error).__name__}: {error}")
        _play_direct_response(
            "I couldn't read that.",
            origin_perf=ocr_last_trigger_monotonic,
        )
        return

    text = _clean_ocr_text(raw_text)
    quality_ok = _ocr_quality_ok(metrics)
    _voice_debug(
        "ocr_tesseract_done",
        raw_chars=len(raw_text),
        chars=len(text),
        words=metrics.get("word_count"),
        mean_conf=round(float(metrics.get("mean_conf", 0.0)), 1),
        good_word_ratio=round(float(metrics.get("good_word_ratio", 0.0)), 3),
        alpha_ratio=round(float(metrics.get("alpha_ratio", 0.0)), 3),
        vowel_word_ratio=round(float(metrics.get("vowel_word_ratio", 0.0)), 3),
        long_consonant_ratio=round(
            float(metrics.get("long_consonant_ratio", 0.0)), 3
        ),
        common_word_ratio=round(
            float(metrics.get("common_word_ratio", 0.0)), 3
        ),
        quality_ok=quality_ok,
        frame_origin=ocr_frame_origin,
        text_box=metrics.get("text_box"),
        frame_shape=None if frame is None else frame.shape,
        elapsed_s=round(time.monotonic() - started, 3),
    )
    if not text:
        _play_direct_response(
            "I don't see any text.",
            origin_perf=ocr_last_trigger_monotonic,
        )
        return
    if not quality_ok:
        _voice_debug(
            "ocr_quality_rejected",
            raw_chars=len(raw_text),
            chars=len(text),
            elapsed_s=round(time.monotonic() - started, 3),
        )
        _play_direct_response(
            "I couldn't read the page. Hold it steady and try again.",
            origin_perf=ocr_last_trigger_monotonic,
        )
        return

    chunks = _split_ocr_speech_chunks(text)
    queued_count = 0
    for index, chunk in enumerate(chunks):
        queued = _play_direct_response(
            chunk,
            origin_perf=ocr_last_trigger_monotonic if index == 0 else None,
            origin_label="command" if index == 0 else "speech",
            # First chunk keeps risk-2 priority. Later chunks sit below routine
            # risk-1/2 repeats so those can speak between chunks.
            priority=None if index == 0 else 0,
        )
        if queued:
            queued_count += 1
        else:
            break

    elapsed_s = round(time.monotonic() - started, 3)
    _voice_debug(
        "ocr_speech_queued",
        queued=bool(queued_count),
        chunks=len(chunks),
        queued_chunks=queued_count,
        chars=len(text),
        first_chunk_chars=len(chunks[0]) if chunks else 0,
        elapsed_s=elapsed_s,
        tts_active=speech_active_event.is_set(),
        queue=speech_queue.qsize(),
    )
    if chunks:
        _voice_debug(
            "ocr_first_chunk_queued",
            chars=len(chunks[0]),
            chunks=len(chunks),
            elapsed_s=elapsed_s,
        )


def _trigger_ocr_reading():
    """Signal the OCR worker, debounced so one utterance reads one snapshot."""
    global last_ocr_trigger_time, ocr_last_trigger_monotonic, voice_ocr_triggers
    if not ENABLE_OCR_READING:
        _voice_debug("ocr_trigger_ignored", reason="disabled")
        return False
    now = time.time()
    if now - last_ocr_trigger_time < OCR_TRIGGER_COOLDOWN_SECONDS:
        _voice_debug(
            "ocr_trigger_debounced",
            remaining_s=round(OCR_TRIGGER_COOLDOWN_SECONDS - (now - last_ocr_trigger_time), 3),
        )
        return True
    last_ocr_trigger_time = now
    ocr_last_trigger_monotonic = time.monotonic()
    voice_ocr_triggers += 1
    ocr_trigger_event.set()
    _voice_debug(
        "ocr_triggered",
        worker_alive=bool(ocr_worker_thread and ocr_worker_thread.is_alive()),
        source=type(ocr_camera_source).__name__ if ocr_camera_source is not None else None,
        tts_active=speech_active_event.is_set(),
    )
    return True


def _ocr_reader_worker():
    while not ocr_stop_event.is_set():
        triggered = ocr_trigger_event.wait(timeout=0.5)
        if ocr_stop_event.is_set():
            break
        if not triggered:
            continue
        ocr_trigger_event.clear()
        _voice_debug("ocr_worker_picked_up", queue=speech_queue.qsize())
        try:
            _perform_ocr_read(ocr_camera_source)
        except Exception as error:
            print(f"[ocr] Reader error: {type(error).__name__}: {error}", flush=True)
            _voice_debug("ocr_worker_error", error=f"{type(error).__name__}: {error}")


def start_ocr_reader(source):
    global ocr_worker_thread, ocr_camera_source
    if not ENABLE_OCR_READING:
        return
    if ocr_worker_thread is not None and ocr_worker_thread.is_alive():
        return
    ocr_camera_source = source
    ocr_stop_event.clear()
    ocr_trigger_event.clear()
    ocr_worker_thread = threading.Thread(
        target=_ocr_reader_worker,
        name="whisper-hat-ocr-reader",
        daemon=True,
    )
    ocr_worker_thread.start()


def stop_ocr_reader():
    global current_ocr_process
    ocr_stop_event.set()
    ocr_trigger_event.set()
    with current_ocr_process_lock:
        process = current_ocr_process
        current_ocr_process = None
    if process is not None:
        try:
            process.kill()
        except Exception:
            pass
    if ocr_worker_thread is not None:
        ocr_worker_thread.join(timeout=3.0)


# ============================================================
# DRAWING
# ============================================================

def draw_hud(frame, best_risk, best_reason, fps, detection_ran):
    frame_height, frame_width = frame.shape[:2]

    if best_risk == 3:
        text = f"STOP: {best_reason}"
        color = (0, 0, 255)
    elif best_risk == 2:
        text = f"CAUTION: {best_reason}"
        color = (0, 165, 255)
    elif best_risk == 1:
        text = f"NOTICE: {best_reason}"
        color = (0, 255, 255)
    else:
        text = "SAFE / passive"
        color = (0, 255, 0)

    cv2.rectangle(frame, (10, 35), (min(frame_width - 10, 900), 120), (0, 0, 0), -1)

    cv2.putText(frame, text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)

    mode = "YOLO" if detection_ran else "reuse"
    tracking_text = "track" if USE_TRACKING else "predict"

    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | {tracking_text}/{mode} | imgsz={IMG_SIZE} | every {PROCESS_EVERY_N_FRAMES} frames",
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )


def draw_object_box(frame, x1, y1, x2, y2, class_name, conf, risk, track_id, debug):
    if risk == 3:
        color = (0, 0, 255)
    elif risk == 2:
        color = (0, 165, 255)
    elif risk == 1:
        color = (0, 255, 255)
    else:
        color = (0, 255, 0)

    x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)

    corridor_tag = "PATH" if debug["in_corridor"] else "SIDE"

    display_name = class_name
    if class_name == "person" and debug.get("face_name"):
        display_name = debug["face_name"]

    if class_name == "person" and debug.get("emotion"):
        display_name += f" [{debug['emotion']} {debug.get('emotion_confidence', 0):.0%}]"

    label = (
        f"{display_name} {conf:.2f} risk={risk} {corridor_tag} "
        f"{debug['fake_distance_m']:.1f}m h={debug.get('box_h_px', 0):.0f}px"
    )

    if track_id != -1:
        label += f" id={track_id}"

    label_y = y1_i - 10
    if label_y < 135:
        label_y = y2_i + 22
    label_y = min(frame.shape[0] - 35, max(25, label_y))

    cv2.putText(frame, label, (x1_i, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

    debug_text = (
        f"{debug['corridor_position']} | {debug['distance_level']} | "
        f"src={debug['distance_source']} | approach={debug['approach_mps']:.2f}m/s"
    )

    debug_y = min(frame.shape[0] - 10, label_y + 20)
    cv2.putText(frame, debug_text, (x1_i, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)




def draw_tof_overlay(frame, snapshot=None, overlap=None):
    """Draw approximate camera/ToF overlap and the latest depth grid."""
    if not ENABLE_TOF:
        return
    frame_height, frame_width = frame.shape[:2]
    if overlap is None:
        overlap = tof_overlap_rectangle(frame_width, frame_height)
    left, top, right, bottom = overlap
    color = (255, 200, 0) if TOF_MOUNT_CALIBRATED else (150, 150, 150)

    if snapshot is None:
        snapshot = get_tof_snapshot()
    label = "ToF calibrated" if TOF_MOUNT_CALIBRATED else "ToF overlap estimate - fusion locked"
    if snapshot["fresh"]:
        label += f" | age {snapshot['age_seconds']:.2f}s"
    cv2.putText(
        frame,
        label,
        (int(left), max(145, int(top) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
    )

    grid = snapshot["grid_mm"]
    if not snapshot["fresh"] or grid is None:
        cv2.rectangle(
            frame,
            (int(left), int(top)),
            (int(right), int(bottom)),
            color,
            1,
        )
        return

    cell_w = (right - left) / 8.0
    cell_h = (bottom - top) / 8.0
    xs = [int(left + i * cell_w) for i in range(9)]
    ys = [int(top + i * cell_h) for i in range(9)]
    for x in xs:
        cv2.line(frame, (x, ys[0]), (x, ys[8]), color, 1)
    for y in ys:
        cv2.line(frame, (xs[0], y), (xs[8], y), color, 1)

    grid_rows = grid.tolist()
    for row in range(8):
        y0, y1 = ys[row], ys[row + 1]
        text_y = min(y1 - 2, y0 + 13)
        row_vals = grid_rows[row]
        for col in range(8):
            value = row_vals[col]
            if math.isfinite(value):
                cv2.putText(
                    frame,
                    f"{value * 0.001:.1f}",
                    (xs[col] + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.27,
                    color,
                    1,
                )


# ============================================================
# EMOTION RECOGNITION
# ============================================================

class EmotionRecognizer:
    def __init__(self, model_path):
        self.enabled = False
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_index = 0
        self.output_index = 0
        self.input_height = 48
        self.input_width = 48
        self.input_channels = 1
        self.input_buffer = None
        self.input_lut = None
        self.normalization_mode = None

        if not ENABLE_EMOTION_RECOGNITION:
            print("Emotion recognition disabled.")
            return

        if Interpreter is None:
            print("No LiteRT/TFLite interpreter is installed. Emotion recognition disabled.")
            print("Try one of these inside your virtual environment:")
            print("  pip install ai-edge-litert")
            print("  pip install tflite-runtime")
            return

        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Emotion model not found: {model_path}")
            print("Place emotion_model.tflite beside this script.")
            return

        try:
            # LiteRT/TFLite's default (AUTO/BUILTIN) op resolver already applies
            # the XNNPACK delegate, and the Python Interpreter exposes no
            # apply_xnnpack_delegate(); num_threads is the only CPU knob here.
            self.interpreter = Interpreter(model_path=str(model_path), num_threads=2)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]
            self.input_index = int(self.input_details["index"])
            self.output_index = int(self.output_details["index"])

            shape = self.input_details["shape"]
            # Expected NHWC, usually [1, 48, 48, 1] or [1, 64, 64, 1].
            if len(shape) == 4:
                self.input_height = int(shape[1])
                self.input_width = int(shape[2])
                self.input_channels = int(shape[3])

            input_dtype = self.input_details["dtype"]
            self.input_buffer = np.empty(
                (1, self.input_height, self.input_width, self.input_channels),
                dtype=input_dtype,
            )
            self._build_input_lut(EMOTION_INPUT_NORMALIZATION)

            self.enabled = True
            print(
                "Emotion model loaded: "
                f"{model_path} input={tuple(shape)} "
                f"dtype={self.input_details['dtype']} "
                f"output={tuple(self.output_details['shape'])}"
            )
        except Exception as e:
            print(f"Could not load emotion model: {e}")
            self.enabled = False

    def _build_input_lut(self, mode):
        levels = np.arange(256, dtype=np.float32)
        if mode == "minus_one_to_one":
            float_lut = (levels / 255.0 - 0.5) * 2.0
        elif mode == "zero_to_one":
            float_lut = levels / 255.0
        elif mode == "raw":
            float_lut = levels
        else:
            raise ValueError(f"Unknown EMOTION_INPUT_NORMALIZATION: {mode}")

        input_dtype = self.input_details["dtype"]
        scale, zero_point = self.input_details.get("quantization", (0.0, 0))

        if np.issubdtype(input_dtype, np.floating):
            self.input_lut = float_lut.astype(input_dtype)
        else:
            if scale and scale > 0:
                quant = np.round(float_lut / scale + zero_point)
            else:
                quant = np.arange(256, dtype=np.float32)
            limits = np.iinfo(input_dtype)
            self.input_lut = np.clip(quant, limits.min, limits.max).astype(input_dtype)
        self.normalization_mode = mode

    @staticmethod
    def _softmax(values):
        values = np.asarray(values, dtype=np.float32)
        values = values - np.max(values)
        exp = np.exp(values)
        total = np.sum(exp)
        return exp / total if total > 0 else exp

    def predict(self, face_bgr, face_gray=None):
        if not self.enabled:
            return None, None
        if self.input_channels == 1 and face_gray is not None:
            if face_gray.size == 0:
                return None, None
        elif face_bgr is None or face_bgr.size == 0:
            return None, None

        try:
            if self.normalization_mode != EMOTION_INPUT_NORMALIZATION:
                self._build_input_lut(EMOTION_INPUT_NORMALIZATION)

            if self.input_channels == 1:
                gray = (
                    face_gray
                    if face_gray is not None
                    else cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                )
                image = cv2.resize(
                    gray,
                    (self.input_width, self.input_height),
                    interpolation=cv2.INTER_AREA,
                )
                np.take(self.input_lut, image, out=self.input_buffer[0, ..., 0])
            else:
                image = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                image = cv2.resize(
                    image,
                    (self.input_width, self.input_height),
                    interpolation=cv2.INTER_AREA,
                )
                np.take(self.input_lut, image, out=self.input_buffer[0])

            self.interpreter.set_tensor(self.input_index, self.input_buffer)
            self.interpreter.invoke()

            output = self.interpreter.get_tensor(self.output_index)[0]
            if output.dtype != np.float32 or output.ndim != 1:
                output = np.asarray(output, dtype=np.float32).reshape(-1)

            out_scale, out_zero = self.output_details.get("quantization", (0.0, 0))
            if not np.issubdtype(self.output_details["dtype"], np.floating) and out_scale:
                output = (output - out_zero) * out_scale

            # Some models already output probabilities; others output logits.
            out_sum = float(np.sum(output))
            if (
                np.any(output < 0)
                or np.any(output > 1)
                or abs(out_sum - 1.0) > 0.15
            ):
                probabilities = self._softmax(output)
            else:
                probabilities = output / max(out_sum, 1e-8)

            best_index = int(np.argmax(probabilities))
            if best_index >= len(EMOTION_LABELS):
                return None, None

            confidence = float(probabilities[best_index])
            emotion = EMOTION_LABELS[best_index]

            if confidence < EMOTION_MIN_CONFIDENCE:
                return "uncertain", confidence

            return emotion, confidence

        except Exception as e:
            print(f"Emotion inference failed: {e}")
            return None, None


def smooth_emotion(face_key, emotion, confidence):
    if emotion is None or confidence is None:
        return None, None

    with state_lock:
        emotion_history[face_key].append((emotion, confidence))

        totals = defaultdict(float)
        counts = defaultdict(int)

        for label, score in emotion_history[face_key]:
            totals[label] += float(score)
            counts[label] += 1

    best_label = max(totals, key=totals.get)
    average_confidence = totals[best_label] / max(1, counts[best_label])
    return best_label, average_confidence


# ============================================================
# FACE RECOGNITION HELPERS
# ============================================================

def face_name_from_path(image_path, root_dir):
    """
    Supports either:
      known_faces/Alex.jpg
      known_faces/Alex/1.jpg
      known_faces/Alex/2.jpg
    """
    root_dir = Path(root_dir)

    if image_path.parent == root_dir:
        return image_path.stem

    return image_path.parent.name


def load_known_faces():
    if not ENABLE_FACE_RECOGNITION:
        print("Face recognition disabled.")
        return [], []

    if face_recognition is None:
        print("face_recognition is not installed. Face recognition disabled.")
        print("Install later with:")
        print("  python3 -m pip install face_recognition")
        return [], []

    cache_path = Path(FACE_CACHE_FILE)

    if cache_path.exists() and not REBUILD_FACE_CACHE:
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            print(f"Loaded {len(data['names'])} known face encoding(s) from cache.")
            return data["names"], data["encodings"]
        except Exception as e:
            print(f"Could not load face cache, rebuilding: {e}")

    root = Path(KNOWN_FACES_DIR)
    if not root.exists():
        print(f"Known faces folder not found: {KNOWN_FACES_DIR}")
        print("Create it like:")
        print("  known_faces/Alex/front.jpg")
        print("  known_faces/Alex/left.jpg")
        return [], []

    names = []
    encodings = []

    image_paths = []
    for suffix in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(root.rglob(suffix))

    print(f"Building known face cache from {len(image_paths)} image(s)...")

    for image_path in image_paths:
        try:
            name = face_name_from_path(image_path, root)

            image_rgb = face_recognition.load_image_file(str(image_path))

            face_locations = face_recognition.face_locations(
                image_rgb,
                number_of_times_to_upsample=0,
                model=FACE_DETECTOR_MODEL,
            )

            if not face_locations:
                print(f"No face found in {image_path}")
                continue

            face_encodings = face_recognition.face_encodings(
                image_rgb,
                known_face_locations=face_locations,
                num_jitters=FACE_NUM_JITTERS,
                model="small",
            )

            if not face_encodings:
                print(f"No encoding created for {image_path}")
                continue

            # Use the first face found in each reference photo.
            names.append(name)
            encodings.append(face_encodings[0])
            print(f"Added known face: {name} from {image_path}")

        except Exception as e:
            print(f"Skipping {image_path}: {e}")

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"names": names, "encodings": encodings}, f)
        print(f"Saved face cache: {cache_path}")
    except Exception as e:
        print(f"Could not save face cache: {e}")

    print(f"Loaded {len(names)} known face encoding(s).")
    return names, encodings


def make_face_key(track_id, class_name, cx, cy, frame_width, frame_height, corridor_position):
    """
    Tracking is enabled by default, so a valid track ID is preferred.
    The zone fallback still keeps face memory separated when no ID is available.
    """
    if track_id != -1:
        return f"face:id:{track_id}"

    x_bin = int(clamp(cx / max(1, frame_width) * 6, 0, 5))
    y_bin = int(clamp(cy / max(1, frame_height) * 4, 0, 3))
    return f"face:{class_name}:{corridor_position}:x{x_bin}:y{y_bin}"


def crop_person_box_with_padding(frame, x1, y1, x2, y2):
    frame_height, frame_width = frame.shape[:2]

    w = box_width(x1, x2)
    h = box_height(y1, y2)
    pad = int(max(w, h) * FACE_CROP_PAD_FRAC)

    left = clamp(int(x1) - pad, 0, frame_width)
    top = clamp(int(y1) - pad, 0, frame_height)
    right = clamp(int(x2) + pad, 0, frame_width)
    bottom = clamp(int(y2) + pad, 0, frame_height)

    if right <= left or bottom <= top:
        return None

    return frame[top:bottom, left:right]



def detect_face_locations_for_emotion(crop_bgr, crop_rgb=None):
    """
    Return (locations, crop_rgb, crop_gray) where locations are (top, right, bottom, left).

    First tries face_recognition/HOG when installed. If HOG misses the face,
    or face_recognition is not installed, it falls back to OpenCV Haar.
    This lets emotion recognition work independently from identity recognition
    while reusing RGB/grayscale conversions downstream.
    """
    locations = []

    if crop_bgr is None or crop_bgr.size == 0:
        return locations, crop_rgb, None

    if face_recognition is not None:
        if crop_rgb is None:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        try:
            locations = face_recognition.face_locations(
                crop_rgb,
                number_of_times_to_upsample=1,
                model=FACE_DETECTOR_MODEL,
            )
        except Exception as e:
            print(f"HOG face detection failed: {e}")
            locations = []

    if locations:
        return locations, crop_rgb, None

    if HAAR_FACE_CASCADE.empty():
        return [], crop_rgb, None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    faces = HAAR_FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(32, 32),
    )

    for x, y, w, h in faces:
        locations.append((int(y), int(x + w), int(y + h), int(x)))

    return locations, crop_rgb, gray


def padded_face_crop(crop_bgr, location, padding_fraction=0.12):
    """Crop one detected face with a little context around the expression."""
    top, right, bottom, left = location
    h_img, w_img = crop_bgr.shape[:2]

    face_w = max(1, right - left)
    face_h = max(1, bottom - top)
    pad = int(max(face_w, face_h) * padding_fraction)

    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(w_img, right + pad)
    bottom = min(h_img, bottom + pad)

    if right <= left or bottom <= top:
        return None

    return crop_bgr[top:bottom, left:right]


def maybe_print_emotion(face_key, face_name, emotion, confidence):
    if not PRINT_EMOTION_TO_TERMINAL:
        return
    if not emotion or confidence is None or emotion == "uncertain":
        return

    now = time.time()
    with state_lock:
        old = last_emotion_print.get(face_key)

        changed = old is None or old["emotion"] != emotion
        cooldown_passed = old is None or now - old["time"] >= EMOTION_PRINT_COOLDOWN_SECONDS

        if changed or cooldown_passed:
            person_text = face_name or "person"
            print(f"Expression: {person_text} looks {emotion} ({confidence:.0%})")
            last_emotion_print[face_key] = {
                "emotion": emotion,
                "time": now,
            }


def emotion_speech_text(face_name, emotion):
    """Create cautious wording because a facial-expression model can be wrong."""
    spoken_forms = {
        "angry": "angry",
        "disgust": "disgusted",
        "fear": "fearful",
        "happy": "happy",
        "sad": "sad",
        "surprise": "surprised",
        "neutral": "neutral",
    }

    expression = spoken_forms.get(emotion, emotion)
    person_text = face_name or "The person"
    return f"{person_text} looks {expression}."


def update_emotion_speech_candidate(face_key, face_name, emotion, confidence):
    """Return a speech message only after the expression is stable.

    This function is called only on frames where fresh emotion inference ran.
    It does not mark an expression as spoken until commit_emotion_speech() is
    called after the message has actually entered the speech queue.
    """
    if not SPEAK_DETECTED_EMOTIONS:
        return None
    if not emotion or confidence is None or emotion == "uncertain":
        return None
    if emotion == "neutral" and not EMOTION_SPEAK_NEUTRAL:
        return None
    if confidence < EMOTION_SPEAK_MIN_CONFIDENCE:
        return None

    now = time.time()
    with state_lock:
        mem = emotion_speech_memory.setdefault(face_key, {
            "candidate": None,
            "stable_count": 0,
            "last_spoken_emotion": None,
            "last_spoken_time": 0.0,
        })

        if mem["candidate"] == emotion:
            mem["stable_count"] += 1
        else:
            mem["candidate"] = emotion
            mem["stable_count"] = 1

        if mem["stable_count"] < EMOTION_SPEAK_STABLE_READINGS:
            return None

        same_as_last = mem["last_spoken_emotion"] == emotion
        elapsed = now - mem["last_spoken_time"]

        if same_as_last and elapsed < EMOTION_SPEAK_COOLDOWN_SECONDS:
            return None
        if not same_as_last and mem["last_spoken_time"] > 0:
            if elapsed < EMOTION_SPEAK_CHANGE_GAP_SECONDS:
                return None

    return emotion_speech_text(face_name, emotion)


def commit_emotion_speech(face_key, emotion):
    with state_lock:
        mem = emotion_speech_memory.get(face_key)
        if mem is None:
            return

        mem["last_spoken_emotion"] = emotion
        mem["last_spoken_time"] = time.time()


def recognize_face_in_person_box(frame_bgr, x1, y1, x2, y2, face_key, frame_count, detection_ran):
    """
    Returns:
      face_name, face_distance_score, emotion, emotion_confidence

    Identity recognition is optional. Emotion recognition continues to work
    even when face_recognition is unavailable or no known-face cache exists.
    """
    identity_available = (
        ENABLE_FACE_RECOGNITION
        and face_recognition is not None
        and len(known_face_encodings) > 0
    )
    emotion_available = (
        ENABLE_EMOTION_RECOGNITION
        and emotion_recognizer is not None
        and emotion_recognizer.enabled
    )

    if not identity_available and not emotion_available:
        return None, None, None, None

    if box_height(y1, y2) < FACE_MIN_PERSON_BOX_HEIGHT:
        return None, None, None, None

    with state_lock:
        old = face_memory.get(face_key)
        if old is not None:
            old = dict(old)

    # Identity and emotion have independent schedules. Using one shared minimum
    # interval would accidentally make identity run only when both modulos align.
    identity_due = (
        identity_available
        and detection_ran
        and frame_count % FACE_RECOGNITION_EVERY_N_FRAMES == 0
    )
    emotion_due = (
        emotion_available
        and detection_ran
        and frame_count % EMOTION_EVERY_N_FRAMES == 0
    )

    if not (identity_due or emotion_due):
        if old and time.time() - old["last_seen"] < 3.0:
            return (
                old.get("name"),
                old.get("distance"),
                old.get("emotion"),
                old.get("emotion_confidence"),
            )
        return None, None, None, None

    crop_bgr = crop_person_box_with_padding(frame_bgr, x1, y1, x2, y2)
    if crop_bgr is None:
        return None, None, None, None

    try:
        face_locations, crop_rgb, crop_gray = detect_face_locations_for_emotion(crop_bgr)

        if not face_locations:
            # Do not keep refreshing an old identity when the face is no longer
            # visible. This prevents a name or emotion from transferring to a
            # different person who later occupies the same area.
            with state_lock:
                face_memory.pop(face_key, None)
                emotion_history.pop(face_key, None)
                last_emotion_print.pop(face_key, None)
                emotion_speech_memory.pop(face_key, None)
            return None, None, None, None

        largest_location = max(
            face_locations,
            key=lambda loc: max(1, loc[2] - loc[0]) * max(1, loc[1] - loc[3]),
        )

        face_crop_bgr = padded_face_crop(crop_bgr, largest_location)
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None, None, None, None

        name = old.get("name") if old else None
        best_distance = old.get("distance") if old else None

        # Identity is performed only when its own schedule is due.
        if identity_due:
            if crop_rgb is None:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(
                crop_rgb,
                known_face_locations=[largest_location],
                num_jitters=FACE_NUM_JITTERS,
                model="small",
            )

            if encodings:
                ref_encodings = (
                    known_face_encodings_matrix
                    if known_face_encodings_matrix is not None
                    else known_face_encodings
                )
                distances = face_recognition.face_distance(
                    ref_encodings,
                    encodings[0],
                )

                if len(distances) > 0:
                    best_index = int(np.argmin(distances))
                    best_distance = float(distances[best_index])
                    name = (
                        known_face_names[best_index]
                        if best_distance <= FACE_MATCH_TOLERANCE
                        else None
                    )

        emotion = old.get("emotion") if old else None
        emotion_confidence = old.get("emotion_confidence") if old else None

        if emotion_due:
            face_crop_gray = (
                padded_face_crop(crop_gray, largest_location)
                if crop_gray is not None
                else None
            )
            new_emotion, new_confidence = emotion_recognizer.predict(
                face_crop_bgr,
                face_gray=face_crop_gray,
            )

            if new_emotion is not None and new_confidence is not None:
                emotion, emotion_confidence = smooth_emotion(
                    face_key,
                    new_emotion,
                    new_confidence,
                )

        with state_lock:
            face_memory[face_key] = {
                "name": name,
                "distance": best_distance,
                "emotion": emotion,
                "emotion_confidence": emotion_confidence,
                "last_seen": time.time(),
            }

        return name, best_distance, emotion, emotion_confidence

    except Exception as e:
        print(f"Face/emotion recognition failed: {e}")
        return None, None, None, None


def cleanup_stale_face_memory(now=None):
    if now is None:
        now = time.time()

    with state_lock:
        stale_face_keys = [
            key for key, mem in face_memory.items()
            if now - mem["last_seen"] > TRACK_STALE_SECONDS
        ]

        for key in stale_face_keys:
            del face_memory[key]
            emotion_history.pop(key, None)
            last_emotion_print.pop(key, None)
            emotion_speech_memory.pop(key, None)



# ============================================================
# CAMERA SOURCES
# ============================================================

class OpenCVSource:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def is_opened(self):
        return self.cap.isOpened()

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class LatestFrameSource:
    """Background capture wrapper that exposes only the newest camera frame."""

    def __init__(self, source):
        self.source = source
        self.condition = threading.Condition()
        self.latest_frame = None
        self.sequence = 0
        self.last_returned_sequence = 0
        self.running = True
        self.failed = False
        self.last_frame_monotonic = 0.0
        self.still_capture_started_monotonic = None
        self.still_capture_active = False
        self.last_still_gap_s = None
        self.max_still_gap_s = 0.0
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="latest-camera-frame",
            daemon=True,
        )
        self.thread.start()

    def _capture_loop(self):
        consecutive_failures = 0
        while self.running:
            ret, frame = self.source.read()
            if not ret or frame is None:
                # One transient V4L2/Picamera2 hiccup should not end a long run;
                # only give up after several consecutive failed reads.
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    with self.condition:
                        self.failed = True
                        self.running = False
                        self.condition.notify_all()
                    return
                time.sleep(0.05)
                continue
            consecutive_failures = 0
            self._note_detection_frame(frame, time.monotonic())

    def _note_detection_frame(self, frame, frame_time):
        """Publish a fresh frame and measure any high-res-still blind window."""
        gap_to_log = None
        with self.condition:
            if (
                self.still_capture_started_monotonic is not None
                and self.last_frame_monotonic
            ):
                live_gap = frame_time - self.last_frame_monotonic
                self.max_still_gap_s = max(
                    self.max_still_gap_s, live_gap
                )
            if (
                self.still_capture_started_monotonic is not None
                and not self.still_capture_active
            ):
                gap = frame_time - self.still_capture_started_monotonic
                self.last_still_gap_s = gap
                self.max_still_gap_s = max(self.max_still_gap_s, gap)
                self.still_capture_started_monotonic = None
                if gap >= 0.20:
                    gap_to_log = (
                        gap, self.max_still_gap_s
                    )
            self.last_frame_monotonic = frame_time
            self.latest_frame = frame
            self.sequence += 1
            self.condition.notify_all()
        if gap_to_log is not None:
            print(
                "[camera] high-res still frame gap="
                f"{gap_to_log[0]:.3f}s max={gap_to_log[1]:.3f}s; "
                "detection and obstacle alerts paused until the next frame",
                flush=True,
            )

    def is_opened(self):
        return self.source.is_opened() and not self.failed

    def read(self):
        deadline = time.time() + LATEST_FRAME_WAIT_SECONDS
        with self.condition:
            while (
                self.running
                and self.sequence <= self.last_returned_sequence
                and time.time() < deadline
            ):
                if self.still_capture_active:
                    deadline = max(
                        deadline,
                        time.time() + OCR_STILL_CAPTURE_WAIT_SECONDS,
                    )
                self.condition.wait(timeout=0.10)

            if self.sequence <= self.last_returned_sequence or self.latest_frame is None:
                return False, None

            self.last_returned_sequence = self.sequence
            return True, self.latest_frame.copy()

    def peek_latest(self):
        """Return a copy of the newest frame without consuming it.

        read() advances last_returned_sequence so the main loop sees each frame
        once. The OCR reader must not steal frames or read HUD annotations, so it
        peeks at a clean copy and leaves the sequence untouched.
        """
        with self.condition:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def capture_still(self):
        """Ask the underlying camera for a one-off high-resolution still."""
        capture = getattr(self.source, "capture_still", None)
        if not callable(capture):
            return None

        with self.condition:
            self.still_capture_started_monotonic = (
                self.last_frame_monotonic or time.monotonic()
            )
            self.still_capture_active = True
            self.condition.notify_all()
        try:
            return capture()
        finally:
            with self.condition:
                self.still_capture_active = False
                self.condition.notify_all()

    def release(self):
        self.running = False
        try:
            self.source.release()
        finally:
            with self.condition:
                self.condition.notify_all()
            self.thread.join(timeout=2.0)


def maybe_wrap_latest_frame(source):
    if USE_LATEST_FRAME_THREAD and source is not None:
        print("Using newest-frame camera capture worker.")
        return LatestFrameSource(source)
    return source


class Picamera2Source:
    def __init__(self):
        from picamera2 import Picamera2
        from libcamera import Transform, controls

        self.picam2 = Picamera2(camera_num=PICAMERA2_CAMERA_INDEX)

        transform = Transform(
            hflip=1 if CAMERA_HFLIP else 0,
            vflip=1 if CAMERA_VFLIP else 0,
        )

        # RGB888 is used directly by this OpenCV pipeline.
        # buffer_count=4 helps avoid frame starvation when YOLO is busy.
        config = self.picam2.create_preview_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
            transform=transform,
            buffer_count=4,
            controls={"FrameRate": CAMERA_FPS},
        )

        self.picam2.configure(config)
        self.transform = transform
        self.still_config = None
        self.io_lock = threading.Lock()

        sensor_model = self.picam2.camera_properties.get("Model", "unknown")
        print(f"Picamera2 opened camera index {PICAMERA2_CAMERA_INDEX}: {sensor_model}")

        self.picam2.start()
        time.sleep(0.5)

        # Camera Module 3 uses the IMX708 sensor and supports autofocus.
        # If this is not a Camera Module 3, the control may fail, so keep it safe.
        try:
            if CAMERA3_AF_MODE == "continuous":
                self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                print("Camera Module 3 autofocus: continuous")
            elif CAMERA3_AF_MODE == "auto":
                self.picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
                print("Camera Module 3 autofocus: auto")
            elif CAMERA3_AF_MODE == "manual":
                self.picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": float(CAMERA3_MANUAL_LENS_POSITION),
                })
                print(f"Camera Module 3 autofocus: manual LensPosition={CAMERA3_MANUAL_LENS_POSITION}")
            else:
                print("Camera autofocus unchanged/off")
        except Exception as e:
            print(f"Autofocus control was not applied: {e}")

        time.sleep(0.3)

    def is_opened(self):
        return True

    def read(self):
        try:
            with self.io_lock:
                frame_bgr = self.picam2.capture_array("main")
            return True, frame_bgr
        except Exception as e:
            print(f"Picamera2 frame read failed: {e}")
            return False, None

    def capture_still(self):
        """Capture one OCR still without changing the preview stream size."""
        switch_capture = getattr(
            self.picam2, "switch_mode_and_capture_array", None
        )
        if not callable(switch_capture):
            return None

        with self.io_lock:
            if self.still_config is None:
                try:
                    self.still_config = self.picam2.create_still_configuration(
                        main={
                            "size": (OCR_STILL_WIDTH, OCR_STILL_HEIGHT),
                            "format": "RGB888",
                        },
                        transform=self.transform,
                        buffer_count=1,
                    )
                except Exception as error:
                    print(
                        "[ocr] high-resolution still configuration failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    self.still_config = False
            if self.still_config is False:
                return None
            try:
                return switch_capture(self.still_config, "main")
            except Exception as error:
                print(
                    "[ocr] high-resolution still capture failed: "
                    f"{type(error).__name__}: {error}"
                )
                return None

    def release(self):
        try:
            self.picam2.stop()
        except Exception:
            pass
        try:
            self.picam2.close()
        except Exception:
            pass


def open_source():
    if CAMERA_BACKEND in {"picamera2", "auto"}:
        try:
            print("Trying Picamera2 camera source...")
            src = Picamera2Source()
            print("Using Picamera2.")
            return maybe_wrap_latest_frame(src)
        except Exception as e:
            print(f"Picamera2 not available or failed: {e}")
            if CAMERA_BACKEND == "picamera2":
                return None

    print("Trying OpenCV camera/video source...")
    src = OpenCVSource(OPENCV_SOURCE)
    if src.is_opened():
        print("Using OpenCV source.")
        return maybe_wrap_latest_frame(src)

    return None


# ============================================================
# STANDALONE TOF TEST MODE
# ============================================================

def run_tof_console_test():
    """Print the live 8x8 depth grid without loading camera or YOLO."""
    print("Starting standalone VL53L5CX test.")
    print("Press Ctrl+C to stop. Distances are millimetres; ---- means invalid.")
    start_tof_worker()
    last_counter = -1
    start_time = time.time()

    try:
        while True:
            snapshot = get_tof_snapshot()
            if snapshot["frame_counter"] != last_counter and snapshot["grid_mm"] is not None:
                last_counter = snapshot["frame_counter"]
                grid = snapshot["grid_mm"]
                print("\033[2J\033[H", end="")
                print(
                    "VL53L5CX 8x8 depth grid | "
                    f"age={snapshot['age_seconds']:.3f}s | frame={last_counter}"
                )
                for row in range(8):
                    values = []
                    for col in range(8):
                        value = grid[row, col]
                        values.append(" ----" if not np.isfinite(value) else f"{int(value):5d}")
                    print(" ".join(values))
                print("\nRun the full hat normally after this test succeeds.")

            if snapshot["last_error"] and time.time() - start_time > 2.0:
                print(f"[ToF test] {snapshot['last_error']}")

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nToF test stopped.")
    finally:
        stop_tof_worker()


# ============================================================
# MAIN
# ============================================================

def run_yolo(model, frame):
    kwargs = {
        "verbose": False,
        "conf": CONFIDENCE,
        "imgsz": DETECTION_IMG_SIZE,
        "device": DEVICE,
    }
    if DETECTION_CLASSES is not None:
        kwargs["classes"] = DETECTION_CLASSES

    if USE_TRACKING:
        return model.track(frame, persist=True, tracker=TRACKER_NAME, **kwargs)

    return model.predict(frame, **kwargs)


def extract_detections(results, names):
    """Convert YOLO Results into plain tuples once per detection frame.

    The tensors are moved to numpy a single time here. Reuse frames (the two of
    every three that do not run YOLO) iterate these tuples instead of re-slicing
    torch tensors and rebuilding a Boxes object per box, and the heavy Results
    object (which pins the detection frame) is released immediately.
    """
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    ids = boxes.id
    ids = ids.cpu().numpy() if ids is not None else None

    detections = []
    for i in range(len(boxes)):
        cls_id = int(classes[i])
        x1, y1, x2, y2 = xyxy[i]
        track_id = int(ids[i]) if ids is not None else -1
        detections.append(
            (names[cls_id], float(confs[i]), x1, y1, x2, y2, track_id)
        )
    return detections


def _load_open_vocab_model():
    """Load the opt-in broad-vocabulary detector.

    YOLOE with a fixed prompt list keeps inference offline after the text
    embeddings are built/exported. A prompt-free or exported model loads with
    plain YOLO and must already contain the baked class names.
    """
    model_path = Path(OPEN_VOCAB_MODEL)
    prompt_free = OPEN_VOCAB_PROMPT_FREE or "-pf" in model_path.stem.lower()
    use_yoloe = YOLOE is not None and model_path.suffix.lower() == ".pt"

    if use_yoloe:
        model = YOLOE(OPEN_VOCAB_MODEL)
        if not prompt_free:
            names = list(OPEN_VOCAB_CLASSES)
            print(
                f"[open-vocab] Building text embeddings for {len(names)} classes "
                "(first run may need the CLIP text encoder)."
            )
            model.set_classes(names, model.get_text_pe(names))
    else:
        model = YOLO(OPEN_VOCAB_MODEL)
        if use_yoloe is False and YOLOE is None and model_path.suffix.lower() == ".pt":
            print(
                "[open-vocab] ultralytics YOLOE is unavailable; loading the "
                "checkpoint with YOLO. Install a YOLOE-capable ultralytics or "
                "use an exported model with baked classes."
            )

    try:
        model.fuse(verbose=False)
    except (TypeError, AttributeError):
        pass

    names = getattr(model, "names", None)
    if isinstance(names, dict):
        names = list(names.values())
    if names:
        print(
            f"[open-vocab] {OPEN_VOCAB_MODEL} -> {len(names)} classes: "
            + ", ".join(str(name) for name in names[:40])
            + (" ..." if len(names) > 40 else "")
        )
    return model


def load_detection_model():
    """Load YOLO and fold BatchNorm into Conv2d for faster CPU inference.

    fusion is in-place and numerically equivalent: same weights, same outputs,
    fewer ops per frame. Non-PyTorch exports (e.g. the NCNN model named in
    MODEL_NAME's comment) are already fused and reject fuse(), so skip them.
    """
    global DETECTION_CLASSES, DETECTION_IMG_SIZE

    if OPEN_VOCAB_ENABLED:
        DETECTION_CLASSES = None
        DETECTION_IMG_SIZE = OPEN_VOCAB_IMG_SIZE
        return _load_open_vocab_model()

    DETECTION_CLASSES = YOLO_CLASSES_TO_KEEP
    DETECTION_IMG_SIZE = IMG_SIZE
    model = YOLO(MODEL_NAME)
    try:
        model.fuse(verbose=False)
    except TypeError:
        pass
    return model


def main():
    global known_face_names, known_face_encodings, known_face_encodings_matrix
    global emotion_recognizer, last_cleanup_time

    ensure_log_file()
    start_tof_worker()

    known_face_names, known_face_encodings = load_known_faces()
    # Build the reference matrix once so face_distance() does not rebuild a
    # (N, 128) array from the Python list on every identity inference.
    known_face_encodings_matrix = (
        np.asarray(known_face_encodings, dtype=np.float64)
        if known_face_encodings
        else None
    )
    emotion_recognizer = EmotionRecognizer(EMOTION_MODEL_FILE)

    if OPEN_VOCAB_ENABLED:
        print("Loading opt-in open-vocabulary model...")
        print(f"Model: {OPEN_VOCAB_MODEL}")
        print(f"Open-vocabulary prompts: {', '.join(OPEN_VOCAB_CLASSES)}")
    else:
        print("Loading YOLO model...")
        print(f"Model: {MODEL_NAME}")
    model = load_detection_model()

    print("Opening source...")
    source = open_source()

    if source is None or not source.is_opened():
        print("Could not open camera/video source.")
        print("For Pi 5 + Camera Module 3, test first with:")
        print("  rpicam-hello --list-cameras")
        print("  rpicam-hello -t 5000 --autofocus-mode continuous")
        print("If those fail, check the 22-pin Pi 5 camera cable orientation and connector latch.")
        stop_tof_worker()
        return

    print()
    print("Whispering Hat Pi 5 Trixie Camera Module 3 version is running.")
    start_speech_worker()
    scene_describe_event.clear()
    start_voice_control()
    start_ocr_reader(source)
    if SPEAK_STARTUP_MESSAGE:
        play_alert_sound(2, "Whispering hat is ready.")
    print("Walking corridor is ON." if USE_WALKING_CORRIDOR else "Walking corridor is OFF.")
    if ENABLE_TOF:
        if TOF_MOUNT_CALIBRATED:
            print("ToF fusion is ON.")
        else:
            print("ToF is being tested, but fusion is LOCKED until rigid mounting/calibration.")
            print("After mounting, set TOF_MOUNT_CALIBRATED = True.")
    print(
        "Alert history: first warning, escalation, reason changes, and "
        "occasional reminders are spoken."
    )
    print("Press Q to quit if SHOW_WINDOW=True.")
    print("Use Ctrl+C to stop if running headless.")
    print()

    frame_count = 0
    last_detections = None

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = source.read()

            if not ret:
                print("No more frames or camera read failed.")
                break

            frame_perf = time.perf_counter()
            frame_count += 1

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.90 * fps + 0.10 * (1.0 / dt)
            prev_time = now

            frame_height, frame_width = frame.shape[:2]
            detection_enabled = detection_is_enabled()
            detection_ran = (
                detection_enabled
                and frame_count % PROCESS_EVERY_N_FRAMES == 0
            )

            if not detection_enabled:
                # Keep the live camera/window running, but do zero object/hazard
                # processing until the user says ON again.
                detections = []
                last_detections = None
            elif detection_ran:
                detections = extract_detections(run_yolo(model, frame), model.names)
                last_detections = detections
            else:
                detections = last_detections

            # None means inference has not run yet, not that the scene is empty.
            describe_requested = (
                ENABLE_SCENE_DESCRIBE
                and scene_describe_event.is_set()
                and (not detection_enabled or last_detections is not None)
            )
            scene_objects = None
            if describe_requested:
                scene_describe_event.clear()
                scene_objects = []

            best_risk = 0
            best_reason = ""
            best_alert_key = None
            best_debug = None
            best_class_name = ""
            best_conf = 0.0
            best_track_id = -1
            best_distance_m = 999.0
            emotion_speech_candidates = []
            nearest_matched_tof_m = float("inf")
            nearest_known_object_m = float("inf")

            # Take one ToF snapshot and overlap rectangle per frame and share
            # them across the summary, overlay, every box, and the unknown
            # obstacle check instead of re-locking and recomputing per box.
            tof_snapshot = None
            tof_overlap = None
            if detection_enabled and ENABLE_TOF:
                tof_snapshot = get_tof_snapshot()
                tof_overlap = tof_overlap_rectangle(frame_width, frame_height)

            if detection_enabled:
                maybe_print_tof_summary(tof_snapshot)

            if SHOW_WINDOW:
                draw_walking_corridor(frame)
                if detection_enabled:
                    draw_tof_overlay(frame, snapshot=tof_snapshot, overlap=tof_overlap)

            if detections:
                for class_name, conf, x1, y1, x2, y2, track_id in detections:
                    cx, cy = center_of_box(x1, y1, x2, y2)
                    area = box_area(x1, y1, x2, y2)

                    foot_y = y2
                    in_corridor, corridor_position = corridor_geometry_for_point(
                        cx, foot_y, frame_width, frame_height
                    )

                    alert_key = make_alert_key(
                        track_id=track_id,
                        class_name=class_name,
                        corridor_position=corridor_position,
                    )
                    distance_key = make_distance_key(
                        track_id=track_id,
                        class_name=class_name,
                        corridor_position=corridor_position,
                    )

                    # Only update motion history on real detection frames and if tracking IDs exist.
                    if detection_ran and track_id != -1:
                        track_history[track_id].append({
                            "time": time.time(),
                            "cx": cx,
                            "cy": cy,
                            "area": area,
                            "class_name": class_name,
                        })

                    risk, reason, debug = score_risk(
                        class_name=class_name,
                        cx=cx,
                        cy=cy,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        track_id=track_id,
                        distance_key=distance_key,
                        update_distance=detection_ran,
                        in_corridor=in_corridor,
                        corridor_position=corridor_position,
                        tof_snapshot=tof_snapshot,
                        tof_overlap=tof_overlap,
                    )

                    if describe_requested:
                        scene_objects.append((
                            class_name, corridor_position, debug["distance_level"],
                            debug["fake_distance_m"], risk,
                        ))

                    if debug.get("distance_source") == "tof" and debug.get("in_corridor"):
                        nearest_matched_tof_m = min(
                            nearest_matched_tof_m,
                            debug.get("fake_distance_m", float("inf")),
                        )
                    if (
                        debug.get("in_corridor")
                        and class_name in KNOWN_OBJECT_CLASSES
                    ):
                        nearest_known_object_m = min(
                            nearest_known_object_m,
                            debug.get("fake_distance_m", float("inf")),
                        )

                    if detection_ran:
                        maybe_print_association_debug(
                            class_name, conf, x1, y1, x2, y2, track_id,
                            risk, reason, debug,
                        )

                    debug["face_name"] = None
                    debug["face_distance_score"] = None
                    debug["emotion"] = None
                    debug["emotion_confidence"] = None

                    if class_name == "person":
                        face_key = make_face_key(
                            track_id=track_id,
                            class_name=class_name,
                            cx=cx,
                            cy=cy,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            corridor_position=corridor_position,
                        )

                        (
                            face_name,
                            face_distance_score,
                            emotion,
                            emotion_confidence,
                        ) = recognize_face_in_person_box(
                            frame_bgr=frame,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            face_key=face_key,
                            frame_count=frame_count,
                            detection_ran=detection_ran,
                        )

                        if face_name:
                            debug["face_name"] = face_name
                            debug["face_distance_score"] = face_distance_score

                            # Make alerts friendlier:
                            # "Caution. Alex close in path."
                            if reason.startswith("person"):
                                reason = reason.replace("person", face_name, 1)

                        if emotion:
                            debug["emotion"] = emotion
                            debug["emotion_confidence"] = emotion_confidence
                            maybe_print_emotion(
                                face_key=face_key,
                                face_name=face_name,
                                emotion=emotion,
                                confidence=emotion_confidence,
                            )

                            # Count only fresh inference frames toward
                            # expression stability, not cached frames.
                            fresh_emotion_frame = (
                                detection_ran
                                and frame_count % EMOTION_EVERY_N_FRAMES == 0
                            )
                            close_enough_for_emotion = (
                                debug["fake_distance_m"]
                                <= EMOTION_SPEAK_MAX_DISTANCE_M
                            )

                            if fresh_emotion_frame and close_enough_for_emotion:
                                emotion_message = update_emotion_speech_candidate(
                                    face_key=face_key,
                                    face_name=face_name,
                                    emotion=emotion,
                                    confidence=emotion_confidence,
                                )
                                if emotion_message:
                                    emotion_speech_candidates.append({
                                        "face_key": face_key,
                                        "emotion": emotion,
                                        "confidence": emotion_confidence,
                                        "message": emotion_message,
                                    })

                    update_object_memory(
                        alert_key,
                        risk,
                        reason,
                        distance_m=debug["fake_distance_m"],
                    )

                    if LOG_ALL_DETECTIONS and detection_ran:
                        log_row(
                            event="detection",
                            class_name=class_name,
                            confidence=conf,
                            risk=risk,
                            reason=reason,
                            debug=debug,
                            track_id=track_id,
                        )

                    if (
                        risk > best_risk
                        or (risk == best_risk and risk > 0 and debug["fake_distance_m"] < best_distance_m)
                    ):
                        best_risk = risk
                        best_reason = reason
                        best_alert_key = alert_key
                        best_debug = debug
                        best_class_name = class_name
                        best_conf = conf
                        best_track_id = track_id
                        best_distance_m = debug["fake_distance_m"]

                    if SHOW_WINDOW:
                        draw_object_box(
                            frame=frame,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            class_name=class_name,
                            conf=conf,
                            risk=risk,
                            track_id=track_id,
                            debug=debug,
                        )

            # If real depth reports a close central obstacle but YOLO cannot
            # associate it with a recognized object, warn generically. The raw
            # median is smoothed and banded with hysteresis here so a noisy
            # sensor or a single bad cell cannot re-announce the same obstacle.
            unknown_tof = tof_unknown_path_distance(tof_snapshot) if detection_enabled else None
            if detection_enabled and unknown_tof is None:
                note_unknown_path_absent(now)
                if unknown_obstacle_state["risk"] > 0:
                    print_unknown_obstacle_debug(
                        now, 0.0, 0.0, 0, 0.0, nearest_matched_tof_m,
                        0, "path clear", "path_clear",
                    )
                    unknown_obstacle_state["risk"] = 0

            if unknown_tof is not None:
                raw_unknown_distance = unknown_tof["distance_m"]
                unknown_distance = smooth_unknown_path_distance(
                    raw_unknown_distance, now
                )
                unknown_tof["raw_distance_m"] = raw_unknown_distance
                unknown_tof["distance_m"] = unknown_distance
                # A known detection can be associated by a ToF cell match or,
                # when the box falls outside the calibrated overlap, by its
                # camera/visual distance. Use the nearest of the two so a
                # visible chair/box does not also produce "unknown obstacle".
                nearest_known_m = min(
                    nearest_matched_tof_m, nearest_known_object_m
                )
                unknown_is_separate = unknown_is_separate_from_known(
                    unknown_distance,
                    nearest_matched_tof_m,
                    nearest_known_object_m,
                )
                if not unknown_is_separate:
                    print_unknown_obstacle_debug(
                        now,
                        raw_unknown_distance,
                        unknown_distance,
                        unknown_tof["valid_cells"],
                        unknown_tof.get("spread_m", 0.0),
                        nearest_known_m,
                        0,
                        "matched a known detection",
                        "matched_known_object",
                    )
                    unknown_tof = None
                    unknown_obstacle_state["risk"] = 0

            if unknown_tof is not None:
                unknown_distance = unknown_tof["distance_m"]
                previous_unknown_risk = unknown_obstacle_state["risk"]
                unknown_risk, unknown_reason = unknown_obstacle_risk(
                    unknown_distance, previous_unknown_risk
                )
                unknown_obstacle_state["risk"] = unknown_risk

                if unknown_risk > 0:
                    unknown_key = "tof:unknown:in_path"
                    unknown_debug = make_unknown_tof_debug(
                        unknown_distance,
                        unknown_tof["valid_cells"],
                    )
                    unknown_debug["raw_distance_m"] = unknown_tof["raw_distance_m"]
                    unknown_debug["tof_spread_m"] = unknown_tof.get("spread_m")
                    update_object_memory(
                        unknown_key,
                        unknown_risk,
                        unknown_reason,
                        distance_m=unknown_distance,
                    )
                    print_unknown_obstacle_debug(
                        now,
                        unknown_tof["raw_distance_m"],
                        unknown_distance,
                        unknown_tof["valid_cells"],
                        unknown_tof.get("spread_m", 0.0),
                        nearest_matched_tof_m,
                        unknown_risk,
                        unknown_reason,
                        "candidate",
                    )
                    if (
                        unknown_risk > best_risk
                        or (
                            unknown_risk == best_risk
                            and unknown_distance < best_distance_m
                        )
                    ):
                        best_risk = unknown_risk
                        best_reason = unknown_reason
                        best_alert_key = unknown_key
                        best_debug = unknown_debug
                        best_class_name = "unknown obstacle"
                        best_conf = 1.0
                        best_track_id = -1
                        best_distance_m = unknown_distance

            # Sweep stale state on the existing interval instead of scanning every
            # dict under the lock on every camera frame.
            if now - last_cleanup_time >= CLEANUP_INTERVAL_SECONDS:
                last_cleanup_time = now
                cleanup_stale_memory(now)
                cleanup_stale_face_memory(now)

            safety_message_queued = False

            if best_alert_key is not None:
                print_alert_debug(best_risk, best_reason, best_debug)

                if should_alert(
                    best_alert_key,
                    best_risk,
                    best_reason,
                    distance_m=best_distance_m,
                ):
                    message = alert_text(best_risk, best_reason)

                    if message:
                        print(message)
                        # A new, escalated, or materially closer STOP
                        # interrupts immediately. An unchanged repeat queues
                        # normally so a stationary book does not keep cutting
                        # off its own reading; queued repeats are rechecked
                        # again at dequeue time.
                        urgent = safety_alert_is_urgent(
                            best_risk, last_alert_decision,
                        )
                        safety_message_queued = _play_safety_alert(
                            best_risk, message, urgent,
                            origin_perf=frame_perf,
                            alert_key=best_alert_key,
                            alert_distance_m=best_distance_m,
                            repeat=bool(
                                last_alert_decision.get("repeat")
                            ),
                        )

                        if best_debug is not None:
                            log_row(
                                event="alert",
                                class_name=best_class_name,
                                confidence=best_conf,
                                risk=best_risk,
                                reason=best_reason,
                                debug=best_debug,
                                track_id=best_track_id,
                            )

            if describe_requested:
                with state_lock:
                    description = (
                        _compose_scene_description(scene_objects)
                        if detection_enabled and detection_is_enabled()
                        else "Detection is off."
                    )
                    _play_direct_response(
                        description,
                        2,
                        origin_perf=scene_last_trigger_monotonic,
                        origin_label="command",
                    )

            # Announce at most one expression per loop. Skip this frame when a
            # safety warning was just queued, then try again on a later stable
            # inference. Emotion messages use risk 0 and remain low priority.
            if emotion_speech_candidates and not safety_message_queued:
                candidate = max(
                    emotion_speech_candidates,
                    key=lambda item: item["confidence"],
                )
                print(f"Emotion speech: {candidate['message']}")
                if play_alert_sound(0, candidate["message"]):
                    commit_emotion_speech(
                        candidate["face_key"],
                        candidate["emotion"],
                    )

            if SHOW_WINDOW:
                draw_hud(frame, best_risk, best_reason, fps, detection_ran)
                if not detection_enabled:
                    cv2.rectangle(frame, (10, 125), (330, 170), (0, 0, 0), -1)
                    cv2.putText(
                        frame,
                        "DETECTION OFF - say DETECTION ON",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        2,
                    )
                cv2.imshow("Whispering Hat Pi Corridor", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q")]:
                    break

    except KeyboardInterrupt:
        print()
        print("Stopped by user.")

    finally:
        source.release()
        stop_voice_control()
        print_voice_summary()
        stop_ocr_reader()
        stop_tof_worker()
        stop_speech_worker()
        close_log_file()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--tof-test" in sys.argv:
        run_tof_console_test()
    else:
        main()
