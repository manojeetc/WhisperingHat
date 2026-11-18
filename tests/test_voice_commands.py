"""Regression tests for command normalization, aliases and routing.

The real failure mode reported on the Pi was a phrase being recognized (or not)
and then not reaching OCR. These tests cover the routing layer; microphone
recognition, echo and TTS timing must still be checked on the Pi with
VOICE_DEBUG_TIMING=True.
"""
import json
import threading

import pytest

from conftest import reset_state


def _reset_voice(new, monkeypatch):
    monkeypatch.setattr(new, "ENABLE_OCR_READING", True)
    monkeypatch.setattr(new, "ENABLE_SCENE_DESCRIBE", True)
    monkeypatch.setattr(new, "VOICE_CONFIRM_COMMANDS", False)
    monkeypatch.setattr(new, "ocr_trigger_event", threading.Event())
    monkeypatch.setattr(new, "scene_describe_event", threading.Event())
    monkeypatch.setattr(new, "last_ocr_trigger_time", 0.0)
    monkeypatch.setattr(new, "last_scene_describe_time", 0.0)
    monkeypatch.setattr(new, "last_voice_command_time", 0.0)
    reset_state(new)


@pytest.mark.parametrize("text,expected", [
    ("  What's   THAT say?! ", "whats that say"),
    ("READ-THIS", "read this"),
    ("detection   on.", "detection on"),
    ("", ""),
])
def test_voice_normalization_removes_punctuation(new, text, expected):
    assert new._normalize_voice_command(text) == expected


@pytest.mark.parametrize("phrase", [
    "read this", "read that", "read", "read text",
    "what's that say", "what does it say",
    "read it", "read this page", "read the text",
    "read that text", "what does this say", "what does that say",
    "read aloud",
])
def test_all_read_aliases_route_to_ocr(new, monkeypatch, phrase):
    _reset_voice(new, monkeypatch)
    triggered = []
    monkeypatch.setattr(new, "_trigger_ocr_reading", lambda: triggered.append(1) or True)
    assert new._apply_voice_command(phrase) is True
    assert triggered == [1]
    assert not new.scene_describe_event.is_set()


@pytest.mark.parametrize("phrase", [
    "what's there", "what do you see", "describe",
    "what's around me", "what's ahead",
    "what is there", "what can you see", "describe the scene", "what is ahead",
])
def test_all_scene_aliases_route_to_scene(new, monkeypatch, phrase):
    _reset_voice(new, monkeypatch)
    assert new._apply_voice_command(phrase) is True
    assert new.scene_describe_event.is_set()
    assert not new.ocr_trigger_event.is_set()


@pytest.mark.parametrize("phrase,expected", [
    ("detection on", True), ("turn detection on", True), ("start detection", True),
    ("detection off", False), ("turn detection off", False), ("stop detection", False),
])
def test_detection_aliases_route(new, monkeypatch, phrase, expected):
    _reset_voice(new, monkeypatch)
    seen = {}
    monkeypatch.setattr(
        new, "set_detection_enabled",
        lambda enabled, announce=True: seen.update(enabled=enabled),
    )
    assert new._apply_voice_command(phrase) is True
    assert seen["enabled"] is expected


def test_unmatched_speech_is_ignored(new, monkeypatch):
    _reset_voice(new, monkeypatch)
    assert new._match_voice_command("hello there") is None
    assert new._apply_voice_command("hello there") is False
    assert not new.ocr_trigger_event.is_set()
    assert not new.scene_describe_event.is_set()


@pytest.mark.parametrize("enabled", [True, False])
def test_grammar_contains_aliases_only_when_feature_enabled(new, monkeypatch, enabled):
    monkeypatch.setattr(new, "ENABLE_OCR_READING", enabled)
    monkeypatch.setattr(new, "ENABLE_SCENE_DESCRIBE", enabled)
    grammar = json.loads(new._voice_command_grammar())
    assert len(grammar) == len(set(grammar))
    assert {"detection on", "detection off", "[unk]"}.issubset(grammar)

    for phrase in (*new.OCR_READ_PHRASES, *new.OCR_READ_ALIASES,
                   *new.SCENE_DESCRIBE_PHRASES, *new.SCENE_DESCRIBE_ALIASES):
        assert (phrase in grammar) is enabled


def test_read_disabled_does_not_fall_back_to_scene(new, monkeypatch):
    _reset_voice(new, monkeypatch)
    monkeypatch.setattr(new, "ENABLE_OCR_READING", False)
    assert new._apply_voice_command("read this") is False
    assert not new.ocr_trigger_event.is_set()
    assert not new.scene_describe_event.is_set()


def test_playback_cooldown_is_opt_in(new, monkeypatch):
    reset_state(new)
    new._set_speech_active(False)
    monkeypatch.setattr(new, "VOICE_PLAYBACK_COOLDOWN_SECONDS", 0.0)
    assert new._voice_playback_cooldown_remaining() == 0.0

    monkeypatch.setattr(new, "VOICE_PLAYBACK_COOLDOWN_SECONDS", 0.5)
    monkeypatch.setattr(new.time, "time", lambda: 1000.0)
    monkeypatch.setattr(new, "speech_last_end_time", 999.8)
    remaining = new._voice_playback_cooldown_remaining()
    assert 0.29 <= remaining <= 0.30

    new._set_speech_active(True, "alert")
    assert new._voice_playback_cooldown_remaining() == 0.0
    new._set_speech_active(False)


def test_direct_response_survives_routine_queue_clear(new):
    reset_state(new)
    new.clear_waiting_speech()
    new.speech_queue.put_nowait((0, 1, "notice", False))
    new.speech_queue.put_nowait((-2, 2, "OCR answer", True))

    new.clear_waiting_speech(keep_protected=True)

    remaining = []
    while not new.speech_queue.empty():
        item = new.speech_queue.get_nowait()
        remaining.append(item)
        new.speech_queue.task_done()
    assert [item[2] for item in remaining] == ["OCR answer"]
    assert remaining[0][3] is True
