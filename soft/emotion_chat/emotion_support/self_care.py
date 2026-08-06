"""Self-care suggestion strategy."""
from __future__ import annotations

import random
from typing import Optional

from .models import Emotion, StrategyResult, SupportMethod
from .strategy_base import SupportStrategy


class SelfCareStrategy(SupportStrategy):
    """Suggest one gentle self-care action without making it an order."""

    method = SupportMethod.SELF_CARE

    def run(self, emotion: Emotion, intensity: Optional[int]) -> StrategyResult:
        suggestions = self.config.get("self_care_suggestions", [])
        if not suggestions:
            suggestions = [
                "Bạn có thể uống một ngụm nước nếu thấy phù hợp.",
                "Bạn có thể đứng dậy đổi tư thế một chút.",
                "Nếu có thể, hãy rời màn hình vài phút để mắt và đầu được nghỉ.",
                "Bạn có thể nhắn cho một người thân hoặc bạn bè mà bạn tin tưởng.",
            ]
        self.io.say(random.choice(suggestions))
        return StrategyResult(completed=True, notes="self_care")
