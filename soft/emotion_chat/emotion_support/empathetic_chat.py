"""Empathetic short conversation support strategy."""
from __future__ import annotations

from typing import Optional

from .models import Emotion, StrategyResult, SupportMethod
from .safety import detect_risk, safety_response, should_cancel
from .strategy_base import SupportStrategy


class EmpatheticChatStrategy(SupportStrategy):
    """Offer a short, bounded empathetic conversation."""

    method = SupportMethod.EMPATHETIC_CHAT

    def run(self, emotion: Emotion, intensity: Optional[int]) -> StrategyResult:
        self.io.say(
            "Mình đang lắng nghe. Bạn có thể kể ngắn gọn điều đang làm bạn thấy như vậy không?"
        )
        answer = self.io.ask("Bạn: ")
        if should_cancel(answer):
            return StrategyResult(completed=False, cancelled=True, notes="user_cancelled")

        risk = detect_risk(answer)
        if risk.risk_detected:
            self.io.say(safety_response())
            return StrategyResult(completed=False, cancelled=False, notes="safety:%s" % risk.category)

        if not answer.strip():
            self.io.say("Không sao, mình sẽ không ép bạn nói. Mình vẫn ở đây nếu bạn muốn quay lại.")
            return StrategyResult(completed=True, notes="empty_share")

        if emotion == Emotion.ANGRY:
            self.io.say("Nghe như chuyện đó thật sự làm bạn khó chịu. Mình sẽ ở đây và nghe bạn nói tiếp khi bạn sẵn sàng.")
        elif emotion == Emotion.SAD:
            self.io.say("Mình nghe thấy chuyện này đang làm bạn nặng lòng. Cảm ơn bạn đã nói ra với mình.")
        elif emotion == Emotion.HAPPY:
            self.io.say("Nghe vui thật đó. Mình mừng vì bạn có một khoảnh khắc tích cực như vậy.")
        else:
            self.io.say("Mình hiểu rồi. Cảm ơn bạn đã chia sẻ với mình.")
        return StrategyResult(completed=True, notes="empathetic_chat")
