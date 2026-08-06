"""Configuration, strategy selection, and support history helpers."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config as runtime_cfg
from .models import Emotion, SupportMethod, SupportSessionResult, normalize_emotion

LOGGER = logging.getLogger(__name__)


DEFAULT_SUPPORT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "confirmation_required": True,
    "intensity_scale_min": 1,
    "intensity_scale_max": 5,
    "ask_timeout_seconds": 25,
    "trigger": {
        "min_profile_sessions": 1,
        "min_confidence": 70.0,
        "emotions": ["sad", "angry"],
    },
    "serial": {
        "enabled": False,
        "port": "",
        "baudrate": 115200,
        "timeout_seconds": 1.0,
    },
    "breathing": {
        "inhale_seconds": 4,
        "hold_seconds": 2,
        "exhale_seconds": 6,
        "cycles": 3,
    },
    "emotion_methods": {
        "happy": ["empathetic_chat", "mini_game"],
        "sad": ["empathetic_chat", "relaxation_audio", "grounding", "self_care"],
        "angry": ["breathing", "grounding", "self_care", "empathetic_chat"],
        "neutral": ["empathetic_chat", "mini_game", "self_care"],
        "unknown": ["empathetic_chat", "self_care"],
    },
    "relaxation_audio": {
        "rain": "assets/audio/rain.wav",
        "ocean": "assets/audio/ocean.wav",
        "nature": "assets/audio/nature.wav",
    },
    "mini_games": [
        {
            "type": "riddle",
            "prompt": "Câu đố vui: Cái gì càng lau càng ướt?",
            "answer": "khăn",
            "response": "Đáp án là cái khăn. Mình đổi nhịp một chút nhé.",
        },
        {
            "type": "positive_question",
            "prompt": "Kể mình nghe một điều nhỏ trong hôm nay vẫn ổn nhé?",
            "response": "Cảm ơn bạn. Điều nhỏ vẫn có giá trị của nó.",
        },
        {
            "type": "countdown",
            "prompt": "Mình cùng đếm chậm từ 5 về 1 nhé: 5, 4, 3, 2, 1.",
            "response": "Tốt rồi, mình quay lại với nhịp chậm hơn một chút.",
        },
    ],
    "self_care_suggestions": [
        "Bạn có thể uống một ngụm nước nếu thấy phù hợp.",
        "Bạn có thể đổi tư thế hoặc thả lỏng vai một chút.",
        "Nếu được, hãy nghỉ mắt khỏi màn hình vài phút.",
        "Bạn có thể liên hệ với một người thân hoặc bạn bè mà bạn tin tưởng.",
    ],
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_config_path() -> Path:
    """Return the project-level support config path."""
    return runtime_cfg.PROJECT_ROOT / "config" / "emotion_support.json"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load support config with safe defaults on missing/invalid files."""
    config_path = path or default_config_path()
    config = dict(DEFAULT_SUPPORT_CONFIG)
    config["_base_dir"] = str(runtime_cfg.PROJECT_ROOT)
    try:
        if not config_path.exists():
            LOGGER.warning("Emotion support config missing: %s", config_path)
            return config
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            LOGGER.warning("Emotion support config root is not an object: %s", config_path)
            return config
        config = _deep_merge(config, loaded)
        config["_base_dir"] = str(runtime_cfg.PROJECT_ROOT)
        return config
    except Exception as exc:
        LOGGER.warning("Invalid emotion support config %s: %s", config_path, exc)
        return config


class EmotionSupportManager:
    """Select support methods and persist support history."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or load_config()

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def should_start_support(
        self,
        detected_emotion: str,
        confidence: float,
        user_id: str = "guest",
        profile_total_sessions: int = 0,
    ) -> bool:
        """Return True when the support flow should interrupt normal chat.

        New users are intentionally routed to normal chat first. Support is a
        follow-up layer, not something HAHA should force into every session.
        """
        if not self.enabled:
            return False

        trigger = _deep_merge(
            DEFAULT_SUPPORT_CONFIG.get("trigger", {}),
            self.config.get("trigger", {}),
        )
        min_sessions = int(trigger.get("min_profile_sessions", 1))
        if profile_total_sessions < min_sessions:
            LOGGER.info(
                "Skipping support for %s: profile sessions %s < %s",
                user_id,
                profile_total_sessions,
                min_sessions,
            )
            return False

        min_confidence = float(trigger.get("min_confidence", 0.0))
        if confidence < min_confidence:
            LOGGER.info(
                "Skipping support for %s: confidence %.1f < %.1f",
                user_id,
                confidence,
                min_confidence,
            )
            return False

        emotion = normalize_emotion(detected_emotion)
        allowed = {str(item) for item in trigger.get("emotions", ["sad", "angry"])}
        if emotion.value not in allowed:
            LOGGER.info(
                "Skipping support for %s: emotion %s not in %s",
                user_id,
                emotion.value,
                sorted(allowed),
            )
            return False

        return True

    def methods_for_emotion(self, emotion: Emotion, user_id: str = "guest") -> List[SupportMethod]:
        """Return configured methods, lightly personalized by previous results."""
        raw_methods = self.config.get("emotion_methods", {}).get(
            emotion.value,
            self.config.get("emotion_methods", {}).get("unknown", []),
        )
        methods: List[SupportMethod] = []
        for item in raw_methods:
            try:
                methods.append(SupportMethod(str(item)))
            except ValueError:
                LOGGER.warning("Unknown support method in config: %s", item)

        if not methods:
            methods = [SupportMethod.EMPATHETIC_CHAT]
        return self._sort_by_history(user_id, methods)

    def _sort_by_history(self, user_id: str, methods: List[SupportMethod]) -> List[SupportMethod]:
        history = self.load_history(user_id)
        if not history:
            return methods

        scores: Dict[str, float] = {method.value: 0.0 for method in methods}
        for item in history[-20:]:
            method = item.get("selected_method")
            if method not in scores:
                continue
            before = item.get("intensity_before")
            after = item.get("intensity_after")
            completed = bool(item.get("completed"))
            cancelled = bool(item.get("cancelled"))
            if completed and isinstance(before, int) and isinstance(after, int):
                scores[method] += float(before - after)
            if cancelled:
                scores[method] -= 1.0

        return sorted(methods, key=lambda method: scores.get(method.value, 0.0), reverse=True)

    def history_path(self, user_id: str) -> Path:
        user_dir = runtime_cfg.USER_DATA_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "support_history.jsonl"

    def load_history(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        path = self.history_path(user_id)
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    rows.append(data)
            except Exception:
                LOGGER.debug("Skipping invalid support history line", exc_info=True)
        return rows[-limit:] if limit else rows

    def save_result(self, result: SupportSessionResult) -> None:
        path = self.history_path(result.user_id)
        line = json.dumps(asdict(result), ensure_ascii=False)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            LOGGER.error("Cannot save support history: %s", exc)
