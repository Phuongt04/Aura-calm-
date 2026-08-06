"""Guided 5-4-3-2-1 grounding strategy."""
from __future__ import annotations

from typing import Optional

from .models import Emotion, StrategyResult, SupportMethod
from .safety import detect_risk, safety_response, should_cancel
from .strategy_base import SupportStrategy


class GroundingStrategy(SupportStrategy):
    """Guide the user through a simple 5-4-3-2-1 grounding exercise."""

    method = SupportMethod.GROUNDING

    def run(self, emotion: Emotion, intensity: Optional[int]) -> StrategyResult:
        steps = [
            "Hãy nhìn quanh và kể tên 5 vật bạn thấy.",
            "Bây giờ cảm nhận 4 điều đang chạm vào cơ thể bạn.",
            "Lắng nghe và kể 3 âm thanh bạn nghe được.",
            "Nhận biết 2 mùi quanh bạn, hoặc 2 điều bạn cảm nhận được nếu không rõ mùi.",
            "Cuối cùng, tập trung vào 1 hơi thở chậm hoặc 1 điều khiến bạn thấy an toàn hơn.",
        ]
        for step in steps:
            self.io.say(step)
            answer = self.io.ask("Bạn: ")
            if should_cancel(answer):
                self.io.say("Được rồi, mình sẽ dừng bài grounding.")
                return StrategyResult(completed=False, cancelled=True, notes="user_cancelled")
            risk = detect_risk(answer)
            if risk.risk_detected:
                self.io.say(safety_response())
                return StrategyResult(completed=False, notes="safety:%s" % risk.category)

        self.io.say("Bạn đã làm xong bài grounding. Hãy để ý xem cơ thể mình có dịu hơn chút nào không.")
        return StrategyResult(completed=True, notes="grounding_completed")
