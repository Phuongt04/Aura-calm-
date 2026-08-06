"""Base interfaces for emotion support strategies and robot expressions."""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

from .models import Emotion, StrategyResult

LOGGER = logging.getLogger(__name__)


class SupportIO:
    """Minimal IO surface used by support sessions and strategies."""

    def __init__(
        self,
        say_func: Callable[[str], None],
        ask_func: Callable[[str], str],
    ) -> None:
        self._say = say_func
        self._ask = ask_func

    def say(self, text: str) -> None:
        self._say(text)

    def ask(self, prompt: str) -> str:
        return self._ask(prompt)


class RobotExpressionController:
    """Best-effort controller for ESP32-backed expressions and animations."""

    def show_emotion(self, emotion: Emotion) -> None:
        raise NotImplementedError

    def start_breathing_animation(self, phase: str, duration: float) -> None:
        raise NotImplementedError

    def stop_animation(self) -> None:
        raise NotImplementedError

    def set_idle(self) -> None:
        raise NotImplementedError


class NullRobotExpressionController(RobotExpressionController):
    """No-op controller used when ESP32 is unavailable."""

    def show_emotion(self, emotion: Emotion) -> None:
        LOGGER.debug("No-op show_emotion(%s)", emotion.value)

    def start_breathing_animation(self, phase: str, duration: float) -> None:
        LOGGER.debug("No-op breathing animation %s %.1fs", phase, duration)

    def stop_animation(self) -> None:
        LOGGER.debug("No-op stop animation")

    def set_idle(self) -> None:
        LOGGER.debug("No-op set idle")


class SerialRobotExpressionController(RobotExpressionController):
    """Text protocol controller for ESP32.

    Serial errors are logged and swallowed so the chatbot can continue without
    the microcontroller.
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None
        self._lock = threading.Lock()

    def _connect(self) -> Any:
        if self._serial is not None:
            return self._serial
        try:
            import serial  # type: ignore

            self._serial = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
        except Exception as exc:
            LOGGER.warning("ESP32 serial unavailable on %s: %s", self.port, exc)
            self._serial = None
        return self._serial

    def _send(self, command: str) -> None:
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return
                conn.write((command.strip() + "\n").encode("utf-8"))
        except Exception as exc:
            LOGGER.warning("ESP32 command failed (%s): %s", command, exc)
            self._serial = None

    def show_emotion(self, emotion: Emotion) -> None:
        face = {
            Emotion.HAPPY: "HAPPY",
            Emotion.SAD: "SAD",
            Emotion.ANGRY: "ANGRY",
            Emotion.NEUTRAL: "CALM",
            Emotion.UNKNOWN: "CALM",
        }.get(emotion, "CALM")
        self._send("FACE:%s" % face)

    def start_breathing_animation(self, phase: str, duration: float) -> None:
        self._send("BREATH:%s:%d" % (phase.upper(), int(duration)))

    def stop_animation(self) -> None:
        self._send("AUDIO:STOP")
        self._send("MODE:IDLE")

    def set_idle(self) -> None:
        self._send("MODE:IDLE")

    def close(self) -> None:
        try:
            if self._serial is not None:
                self._serial.close()
        except Exception:
            LOGGER.debug("Ignoring serial close error", exc_info=True)
        finally:
            self._serial = None


class SupportStrategy:
    """Base class for a concrete support method."""

    method = None

    def __init__(
        self,
        config: Dict[str, Any],
        io: SupportIO,
        controller: Optional[RobotExpressionController] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.config = config
        self.io = io
        self.controller = controller or NullRobotExpressionController()
        self.cancel_event = cancel_event or threading.Event()

    def run(self, emotion: Emotion, intensity: Optional[int]) -> StrategyResult:
        raise NotImplementedError
