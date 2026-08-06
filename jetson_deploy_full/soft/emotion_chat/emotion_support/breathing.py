"""Guided 4-2-6 breathing support strategy."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .models import Emotion, StrategyResult, SupportMethod
from .strategy_base import SupportStrategy

LOGGER = logging.getLogger(__name__)


class BreathingStrategy(SupportStrategy):
    """Run a cancellable breathing exercise without one long blocking sleep."""

    method = SupportMethod.BREATHING

    def _wait(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if self.cancel_event.wait(min(0.2, max(0.0, remaining))):
                return False
        return True

    def run(self, emotion: Emotion, intensity: Optional[int]) -> StrategyResult:
        config = self.config.get("breathing", {})
        inhale = float(config.get("inhale_seconds", 4))
        hold = float(config.get("hold_seconds", 2))
        exhale = float(config.get("exhale_seconds", 6))
        cycles = int(config.get("cycles", 3))

        self.io.say(
            "Mình sẽ hướng dẫn bài thở ngắn. Nếu muốn dừng, bạn có thể nói dừng lại sau nhịp hiện tại."
        )

        for index in range(max(1, cycles)):
            LOGGER.info("Breathing cycle %s/%s", index + 1, cycles)
            for phase, duration, text in (
                ("INHALE", inhale, "Hít vào chậm trong %d giây." % inhale),
                ("HOLD", hold, "Giữ hơi trong %d giây." % hold),
                ("EXHALE", exhale, "Thở ra thật chậm trong %d giây." % exhale),
            ):
                self.controller.start_breathing_animation(phase, duration)
                self.io.say(text)
                if not self._wait(duration):
                    self.controller.stop_animation()
                    self.io.say("Được rồi, mình sẽ dừng bài thở ở đây.")
                    return StrategyResult(completed=False, cancelled=True, notes="user_cancelled")

        self.controller.set_idle()
        self.io.say("Tốt rồi. Bạn cứ giữ nhịp thở chậm thêm một chút nếu thấy dễ chịu.")
        return StrategyResult(completed=True, notes="breathing_completed")


def cancel_after(delay_seconds: float, event: threading.Event) -> threading.Thread:
    """Testing helper that cancels a breathing session after a delay."""
    def _run() -> None:
        time.sleep(delay_seconds)
        event.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
