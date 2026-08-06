"""Shared models for non-contact emotion support sessions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Emotion(str, Enum):
    """Normalized emotion labels used by the support layer."""

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class SupportMethod(str, Enum):
    """Available non-contact support methods."""

    EMPATHETIC_CHAT = "empathetic_chat"
    BREATHING = "breathing"
    RELAXATION_AUDIO = "relaxation_audio"
    GROUNDING = "grounding"
    MINI_GAME = "mini_game"
    SELF_CARE = "self_care"


class SupportState(str, Enum):
    """State machine states for an emotion support session."""

    IDLE = "IDLE"
    DETECTED = "DETECTED"
    CONFIRM_EMOTION = "CONFIRM_EMOTION"
    ASK_INTENSITY = "ASK_INTENSITY"
    OFFER_SUPPORT = "OFFER_SUPPORT"
    WAIT_SELECTION = "WAIT_SELECTION"
    RUN_SUPPORT = "RUN_SUPPORT"
    POST_CHECK = "POST_CHECK"
    SAVE_RESULT = "SAVE_RESULT"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    SAFETY_ESCALATION = "SAFETY_ESCALATION"


@dataclass
class SupportSessionResult:
    """Persistable summary of a support session.

    Raw audio, face images, and sensitive verbatim user content are intentionally
    excluded. Notes should contain only operational information.
    """

    user_id: str
    detected_emotion: str
    confirmed_emotion: str
    confidence: float
    intensity_before: Optional[int]
    selected_method: str
    intensity_after: Optional[int]
    completed: bool
    cancelled: bool
    started_at: str
    ended_at: str
    notes: Optional[str] = None


@dataclass
class StrategyResult:
    """Result returned by a support strategy."""

    completed: bool
    cancelled: bool = False
    notes: Optional[str] = None


def normalize_emotion(value: Optional[str]) -> Emotion:
    """Map model/user labels to normalized support emotions."""
    if not value:
        return Emotion.UNKNOWN
    key = value.strip().lower()
    mapping = {
        "vui": Emotion.HAPPY,
        "happy": Emotion.HAPPY,
        "happiness": Emotion.HAPPY,
        "buon": Emotion.SAD,
        "sad": Emotion.SAD,
        "sadness": Emotion.SAD,
        "gian": Emotion.ANGRY,
        "angry": Emotion.ANGRY,
        "anger": Emotion.ANGRY,
        "tuc": Emotion.ANGRY,
        "binhthuong": Emotion.NEUTRAL,
        "binh thuong": Emotion.NEUTRAL,
        "neutral": Emotion.NEUTRAL,
        "normal": Emotion.NEUTRAL,
        "met": Emotion.NEUTRAL,
        "tired": Emotion.NEUTRAL,
    }
    return mapping.get(key, Emotion.UNKNOWN)


def emotion_vi_label(emotion: Emotion) -> str:
    """Return a short Vietnamese label for prompts."""
    labels = {
        Emotion.HAPPY: "vui",
        Emotion.SAD: "buồn",
        Emotion.ANGRY: "giận hoặc khó chịu",
        Emotion.NEUTRAL: "bình thường",
        Emotion.UNKNOWN: "chưa rõ",
    }
    return labels.get(emotion, "chưa rõ")


def support_method_label(method: SupportMethod) -> str:
    """Return user-facing Vietnamese method labels."""
    labels = {
        SupportMethod.EMPATHETIC_CHAT: "trò chuyện",
        SupportMethod.BREATHING: "bài thở",
        SupportMethod.RELAXATION_AUDIO: "âm thanh thư giãn",
        SupportMethod.GROUNDING: "grounding 5-4-3-2-1",
        SupportMethod.MINI_GAME: "trò chơi ngắn",
        SupportMethod.SELF_CARE: "gợi ý tự chăm sóc",
    }
    return labels.get(method, method.value)
