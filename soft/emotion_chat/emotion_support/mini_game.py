"""Short diversion activities for emotion support."""
from __future__ import annotations

import random
from typing import Optional

from .models import Emotion, StrategyResult, SupportMethod
from .safety import should_cancel
from .strategy_base import SupportStrategy


class MiniGameStrategy(SupportStrategy):
    """Run one short riddle or positive reflection activity."""

    method = SupportMethod.MINI_GAME

    def run(self, emotion: Emotion, intensity: Optional[int]) -> StrategyResult:
        activities = self.config.get("mini_games", [])
        if not activities:
            activities = [
                {
                    "type": "positive_question",
                    "prompt": "Kể mình nghe một điều nhỏ trong hôm nay vẫn ổn nhé?",
                    "response": "Cảm ơn bạn. Những điều nhỏ như vậy cũng đáng được ghi nhận.",
                },
                {
                    "type": "riddle",
                    "prompt": "Câu đố vui: Cái gì càng lau càng ướt?",
                    "answer": "khăn",
                    "response": "Đáp án là cái khăn. Một câu nhỏ để mình đổi nhịp chút nhé.",
                },
            ]

        item = random.choice(activities)
        self.io.say(str(item.get("prompt", "Mình có một hoạt động nhỏ cho bạn.")))
        answer = self.io.ask("Bạn: ")
        if should_cancel(answer):
            return StrategyResult(completed=False, cancelled=True, notes="user_cancelled")
        self.io.say(str(item.get("response", "Cảm ơn bạn đã thử hoạt động này.")))
        return StrategyResult(completed=True, notes="mini_game:%s" % item.get("type", "unknown"))
