"""
long_term.py
============
Long-term personalization layer for MONE.

This module converts conversation memory, multimodal emotion history, and
learned communication style into a compact per-user interaction strategy.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .. import config as cfg
from .history import SessionLog, load_all
from .profile import UserProfile


def _user_dir(user_id: str) -> Path:
    path = cfg.USER_DATA_DIR / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_path(user_id: str) -> Path:
    return _user_dir(user_id) / "long_term_profile.json"


def _strategy_path(user_id: str) -> Path:
    return _user_dir(user_id) / "interaction_strategy.txt"


def _session_topic(session: SessionLog) -> str:
    user_turns = [
        turn.get("content", "").strip()
        for turn in session.turns
        if turn.get("role") == "user" and turn.get("content", "").strip()
    ]
    if not user_turns:
        return ""
    return user_turns[-1].replace("\n", " ")[:160]


def _emotion_guidance(dominant: str) -> list[str]:
    guidance = {
        "vui": [
            "Mirror positive energy without becoming too noisy.",
            "Ask open questions that let the user share good moments.",
        ],
        "buon": [
            "Use a slower, warmer tone and validate feelings first.",
            "Avoid rushing into advice unless the user asks for solutions.",
        ],
        "gian": [
            "Stay calm, concise, and non-judgmental.",
            "Help the user name the problem before suggesting action.",
        ],
        "binhthuong": [
            "Keep the conversation practical, friendly, and natural.",
            "Offer light choices such as talking, music, or a short joke.",
        ],
    }
    return guidance.get(dominant, guidance["binhthuong"])


def build_long_term_profile(user_id: str, profile: UserProfile) -> dict:
    """Build a structured long-term personalization profile."""
    sessions = load_all(user_id)
    emotion_counts = Counter(s.final_emotion for s in sessions if s.final_emotion)
    conflict_counts = Counter(s.conflict_key for s in sessions if s.conflict_key)
    voice_counts = Counter(s.voice_emotion for s in sessions if s.voice_emotion)
    face_counts = Counter(s.face_emotion for s in sessions if s.face_emotion)

    dominant = profile.dominant_emotion
    recent_topics = [
        topic for topic in (_session_topic(s) for s in sessions[-cfg.LONG_TERM_RECENT_TOPIC_LIMIT:])
        if topic
    ]

    return {
        "user_id": user_id,
        "display_name": profile.display_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_sessions": profile.total_sessions,
        "dominant_emotion": dominant,
        "emotion_counts": dict(emotion_counts or profile.emotion_counts),
        "voice_emotion_counts": dict(voice_counts),
        "face_emotion_counts": dict(face_counts),
        "conflict_patterns": dict(conflict_counts or profile.conflict_counts),
        "preferred_styles": profile.preferred_styles(),
        "styles_to_avoid": profile.styles_to_avoid(),
        "recent_topics": recent_topics,
        "interaction_guidance": _emotion_guidance(dominant),
        "privacy_mode": "local-first",
    }


def _strategy_text(data: dict) -> str:
    lines = [
        "[Long-Term Personalized Interaction Strategy]",
        f"User: {data['display_name']} ({data['user_id']})",
        f"Sessions observed: {data['total_sessions']}",
        f"Dominant emotion pattern: {data['dominant_emotion']}",
        "Privacy mode: local-first memory; do not reveal private memory unless helpful and natural.",
        "",
        "Preferred communication style:",
    ]

    preferred = data.get("preferred_styles") or []
    avoided = data.get("styles_to_avoid") or []
    lines.append("- " + ", ".join(preferred) if preferred else "- not learned yet")

    lines.append("")
    lines.append("Styles to reduce:")
    lines.append("- " + ", ".join(avoided) if avoided else "- none learned yet")

    lines.append("")
    lines.append("Emotion-aware guidance:")
    for item in data.get("interaction_guidance", []):
        lines.append(f"- {item}")

    recent_topics = data.get("recent_topics") or []
    lines.append("")
    lines.append("Recent recurring topics:")
    if recent_topics:
        for topic in recent_topics:
            lines.append(f"- {topic}")
    else:
        lines.append("- none")

    return "\n".join(lines).strip() + "\n"


def update_long_term_personalization(user_id: str, profile: UserProfile) -> list[Path]:
    """Persist JSON + text strategy for this user's long-term personalization."""
    if not cfg.LONG_TERM_PERSONALIZATION_ENABLED:
        return []

    data = build_long_term_profile(user_id, profile)
    json_path = _json_path(user_id)
    text_path = _strategy_path(user_id)

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(_strategy_text(data), encoding="utf-8")
    return [json_path, text_path]


def load_long_term_strategy(user_id: str) -> str:
    """Load compact long-term strategy for prompt context."""
    if not cfg.LONG_TERM_PERSONALIZATION_ENABLED:
        return ""

    path = _strategy_path(user_id)
    if not path.exists():
        return ""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""

    return text[-cfg.LONG_TERM_CONTEXT_CHARS:]
