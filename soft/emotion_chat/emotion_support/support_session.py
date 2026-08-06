"""State machine for non-contact emotion support sessions."""
from __future__ import annotations

import logging
import re
import threading
import unicodedata
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .. import config as runtime_cfg
from .. import voice_io
from .breathing import BreathingStrategy
from .empathetic_chat import EmpatheticChatStrategy
from .grounding import GroundingStrategy
from .manager import EmotionSupportManager
from .mini_game import MiniGameStrategy
from .models import (
    Emotion,
    StrategyResult,
    SupportMethod,
    SupportSessionResult,
    SupportState,
    emotion_vi_label,
    normalize_emotion,
    support_method_label,
)
from .relaxation_audio import RelaxationAudioStrategy
from .safety import detect_risk, safety_response, should_cancel
from .self_care import SelfCareStrategy
from .strategy_base import (
    NullRobotExpressionController,
    RobotExpressionController,
    SerialRobotExpressionController,
    SupportIO,
)

LOGGER = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", (text or "").lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d").replace("ð", "d")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def default_support_io() -> SupportIO:
    """Create runtime IO backed by existing console/STT/TTS helpers."""

    def say(text: str) -> None:
        print("MONE: %s" % text)
        voice_io.speak(text)

    def ask(prompt: str) -> str:
        if runtime_cfg.VOICE_CHAT_ENABLED:
            print(prompt, end="")
            return voice_io.listen_once()
        return input(prompt).strip()

    return SupportIO(say, ask)


def build_robot_controller(config: Dict) -> RobotExpressionController:
    """Build a serial controller when configured, otherwise no-op."""
    serial_cfg = config.get("serial", {})
    if serial_cfg.get("enabled") and serial_cfg.get("port"):
        return SerialRobotExpressionController(
            port=str(serial_cfg.get("port")),
            baudrate=int(serial_cfg.get("baudrate", 115200)),
            timeout=float(serial_cfg.get("timeout_seconds", 1.0)),
        )
    return NullRobotExpressionController()


class EmotionSupportSession:
    """Run the support state machine from detection to persistence."""

    def __init__(
        self,
        manager: Optional[EmotionSupportManager] = None,
        io: Optional[SupportIO] = None,
        controller: Optional[RobotExpressionController] = None,
    ) -> None:
        self.manager = manager or EmotionSupportManager()
        self.config = self.manager.config
        self.io = io or default_support_io()
        self.controller = controller or build_robot_controller(self.config)
        self.cancel_event = threading.Event()
        self.state = SupportState.IDLE

    def run(
        self,
        detected_emotion: str,
        confidence: float,
        user_id: str = "guest",
    ) -> SupportSessionResult:
        """Run one bounded support session and persist its result."""
        started_at = _now_iso()
        detected = normalize_emotion(detected_emotion)
        confirmed = detected
        intensity_before: Optional[int] = None
        intensity_after: Optional[int] = None
        selected_method = ""
        completed = False
        cancelled = False
        notes = ""

        try:
            self.state = SupportState.DETECTED
            self.controller.show_emotion(detected)

            if not self.manager.enabled:
                self.state = SupportState.CANCELLED
                cancelled = True
                notes = "support_disabled"
                return self._finalize(
                    user_id, detected, confirmed, confidence, intensity_before,
                    selected_method, intensity_after, completed, cancelled,
                    started_at, notes,
                )

            self.state = SupportState.CONFIRM_EMOTION
            confirmed, cancelled, safety_note = self._confirm_emotion(detected)
            if safety_note:
                notes = safety_note
                self.state = SupportState.SAFETY_ESCALATION
                return self._finalize(
                    user_id, detected, confirmed, confidence, intensity_before,
                    selected_method, intensity_after, completed, cancelled,
                    started_at, notes,
                )
            if cancelled:
                self.state = SupportState.CANCELLED
                notes = "user_declined_confirmation"
                return self._finalize(
                    user_id, detected, confirmed, confidence, intensity_before,
                    selected_method, intensity_after, completed, cancelled,
                    started_at, notes,
                )

            self.state = SupportState.ASK_INTENSITY
            intensity_before, cancelled = self._ask_intensity("Mức độ hiện tại của bạn từ 1 đến 5 là bao nhiêu?")
            if cancelled:
                self.state = SupportState.CANCELLED
                notes = "user_cancelled_before_method"
                return self._finalize(
                    user_id, detected, confirmed, confidence, intensity_before,
                    selected_method, intensity_after, completed, cancelled,
                    started_at, notes,
                )

            self.state = SupportState.OFFER_SUPPORT
            methods = self.manager.methods_for_emotion(confirmed, user_id)
            self.state = SupportState.WAIT_SELECTION
            method, cancelled = self._choose_method(methods)
            if cancelled or method is None:
                self.state = SupportState.CANCELLED
                notes = "user_declined_methods"
                return self._finalize(
                    user_id, detected, confirmed, confidence, intensity_before,
                    selected_method, intensity_after, completed, True,
                    started_at, notes,
                )

            selected_method = method.value
            self.state = SupportState.RUN_SUPPORT
            strategy_result = self._run_strategy(method, confirmed, intensity_before)
            completed = strategy_result.completed
            cancelled = strategy_result.cancelled
            notes = strategy_result.notes or ""
            if cancelled:
                self.state = SupportState.CANCELLED
                return self._finalize(
                    user_id, detected, confirmed, confidence, intensity_before,
                    selected_method, intensity_after, completed, cancelled,
                    started_at, notes,
                )
            if notes.startswith("safety:"):
                self.state = SupportState.SAFETY_ESCALATION
                return self._finalize(
                    user_id, detected, confirmed, confidence, intensity_before,
                    selected_method, intensity_after, completed, cancelled,
                    started_at, notes,
                )

            self.state = SupportState.POST_CHECK
            intensity_after, _ = self._ask_intensity("Bây giờ mức độ đó còn khoảng bao nhiêu từ 1 đến 5?")
            self.state = SupportState.SAVE_RESULT
            return self._finalize(
                user_id, detected, confirmed, confidence, intensity_before,
                selected_method, intensity_after, completed, cancelled,
                started_at, notes,
            )
        except Exception as exc:
            LOGGER.error("Emotion support session failed: %s", exc, exc_info=True)
            self.state = SupportState.CANCELLED
            return self._finalize(
                user_id, detected, confirmed, confidence, intensity_before,
                selected_method, intensity_after, completed, True,
                started_at, "error:%s" % exc.__class__.__name__,
            )
        finally:
            try:
                self.controller.set_idle()
            except Exception:
                LOGGER.debug("Ignoring controller idle error", exc_info=True)

    def _confirm_emotion(self, detected: Emotion) -> Tuple[Emotion, bool, str]:
        if not self.config.get("confirmation_required", True):
            return detected, False, ""

        self.io.say("Có phải hiện tại bạn đang cảm thấy hơi %s không?" % emotion_vi_label(detected))
        answer = self.io.ask("Bạn: ")
        risk = detect_risk(answer)
        if risk.risk_detected:
            self.io.say(safety_response())
            return detected, False, "safety:%s" % risk.category

        normalized = _normalize(answer)
        if should_cancel(answer):
            return detected, True, ""

        tokens = set(normalized.split())
        has_denial = bool(tokens.intersection({"khong", "chua", "sai"}))
        if has_denial:
            user_emotion = self._infer_emotion_from_text(normalized)
            if user_emotion != Emotion.UNKNOWN:
                return user_emotion, False, "user_corrected_emotion"
            return Emotion.NEUTRAL, False, "user_denied_emotion"

        if tokens.intersection({"co", "dung", "phai", "uh", "u", "ok", "yes"}):
            return detected, False, ""

        user_emotion = self._infer_emotion_from_text(normalized)
        if user_emotion != Emotion.UNKNOWN:
            return user_emotion, False, "user_corrected_emotion"

        return detected, False, ""

    def _infer_emotion_from_text(self, normalized: str) -> Emotion:
        if any(word in normalized for word in ("vui", "hao hung", "phan khoi")):
            return Emotion.HAPPY
        if any(word in normalized for word in ("buon", "chan", "nang long")):
            return Emotion.SAD
        if any(word in normalized for word in ("gian", "tuc", "buc", "kho chiu", "uc che")):
            return Emotion.ANGRY
        if any(word in normalized for word in ("binh thuong", "on", "met", "tam duoc")):
            return Emotion.NEUTRAL
        return Emotion.UNKNOWN

    def _ask_intensity(self, question: str) -> Tuple[Optional[int], bool]:
        min_value = int(self.config.get("intensity_scale_min", 1))
        max_value = int(self.config.get("intensity_scale_max", 5))
        for _ in range(2):
            self.io.say(question)
            answer = self.io.ask("Bạn: ")
            if should_cancel(answer):
                return None, True
            match = re.search(r"\b([0-9]+)\b", answer)
            if match:
                value = int(match.group(1))
                if min_value <= value <= max_value:
                    return value, False
            self.io.say("Mình cần một số từ %d đến %d thôi nhé." % (min_value, max_value))
        return None, False

    def _choose_method(self, methods: List[SupportMethod]) -> Tuple[Optional[SupportMethod], bool]:
        labels = ["[%d] %s" % (idx + 1, support_method_label(method)) for idx, method in enumerate(methods)]
        self.io.say("Bạn muốn thử cách nào: %s? Bạn cũng có thể nói không cần." % "; ".join(labels))
        answer = self.io.ask("Chọn cách hỗ trợ: ")
        if should_cancel(answer):
            return None, True
        normalized = _normalize(answer)
        match = re.search(r"\b([0-9]+)\b", normalized)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < len(methods):
                return methods[index], False

        keyword_map = {
            SupportMethod.EMPATHETIC_CHAT: ("tro chuyen", "noi chuyen", "tam su", "ke"),
            SupportMethod.BREATHING: ("tho", "hit", "thở", "breathing"),
            SupportMethod.RELAXATION_AUDIO: ("nhac", "am thanh", "mua", "song", "nature"),
            SupportMethod.GROUNDING: ("grounding", "5 4 3 2 1", "binh tinh"),
            SupportMethod.MINI_GAME: ("tro choi", "cau do", "game", "dem"),
            SupportMethod.SELF_CARE: ("tu cham soc", "uong nuoc", "nghi", "goi y"),
        }
        for method in methods:
            if any(word in normalized for word in keyword_map.get(method, ())):
                return method, False
        if methods:
            return methods[0], False
        return None, True

    def _run_strategy(
        self,
        method: SupportMethod,
        emotion: Emotion,
        intensity: Optional[int],
    ) -> StrategyResult:
        strategy_cls = {
            SupportMethod.EMPATHETIC_CHAT: EmpatheticChatStrategy,
            SupportMethod.BREATHING: BreathingStrategy,
            SupportMethod.RELAXATION_AUDIO: RelaxationAudioStrategy,
            SupportMethod.GROUNDING: GroundingStrategy,
            SupportMethod.MINI_GAME: MiniGameStrategy,
            SupportMethod.SELF_CARE: SelfCareStrategy,
        }[method]
        strategy = strategy_cls(
            self.config,
            self.io,
            controller=self.controller,
            cancel_event=self.cancel_event,
        )
        return strategy.run(emotion, intensity)

    def _finalize(
        self,
        user_id: str,
        detected: Emotion,
        confirmed: Emotion,
        confidence: float,
        intensity_before: Optional[int],
        selected_method: str,
        intensity_after: Optional[int],
        completed: bool,
        cancelled: bool,
        started_at: str,
        notes: str,
    ) -> SupportSessionResult:
        result = SupportSessionResult(
            user_id=user_id,
            detected_emotion=detected.value,
            confirmed_emotion=confirmed.value,
            confidence=confidence,
            intensity_before=intensity_before,
            selected_method=selected_method,
            intensity_after=intensity_after,
            completed=completed,
            cancelled=cancelled,
            started_at=started_at,
            ended_at=_now_iso(),
            notes=notes or None,
        )
        self.manager.save_result(result)
        if self.state not in {
            SupportState.CANCELLED,
            SupportState.SAFETY_ESCALATION,
        }:
            self.state = SupportState.COMPLETED
        return result
