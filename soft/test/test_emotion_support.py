import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_chat import config as runtime_cfg
from emotion_chat.emotion_support.breathing import BreathingStrategy
from emotion_chat.emotion_support.manager import EmotionSupportManager, load_config
from emotion_chat.emotion_support.models import Emotion
from emotion_chat.emotion_support.relaxation_audio import RelaxationAudioStrategy
from emotion_chat.emotion_support.safety import detect_risk
from emotion_chat.emotion_support.strategy_base import NullRobotExpressionController, SupportIO
from emotion_chat.emotion_support.support_session import EmotionSupportSession


class MockIO(SupportIO):
    def __init__(self, answers):
        self.answers = list(answers)
        self.spoken = []
        super().__init__(self.spoken.append, self._ask)

    def _ask(self, prompt):
        if not self.answers:
            return ""
        return self.answers.pop(0)


class EmotionSupportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_user_data_dir = runtime_cfg.USER_DATA_DIR
        runtime_cfg.USER_DATA_DIR = Path(self.tmp.name) / "user_data"

    def tearDown(self):
        runtime_cfg.USER_DATA_DIR = self.old_user_data_dir
        self.tmp.cleanup()

    def _manager(self, extra=None):
        config = {
            "enabled": True,
            "confirmation_required": True,
            "intensity_scale_min": 1,
            "intensity_scale_max": 5,
            "_base_dir": self.tmp.name,
            "breathing": {
                "inhale_seconds": 0.01,
                "hold_seconds": 0.01,
                "exhale_seconds": 0.01,
                "cycles": 1,
            },
            "emotion_methods": {
                "happy": ["empathetic_chat"],
                "sad": ["empathetic_chat"],
                "angry": ["breathing"],
                "neutral": ["empathetic_chat"],
                "unknown": ["empathetic_chat"],
            },
            "relaxation_audio": {"rain": "missing/rain.wav"},
            "mini_games": [
                {
                    "type": "positive_question",
                    "prompt": "Mot dieu on?",
                    "response": "Cam on ban.",
                }
            ],
            "self_care_suggestions": ["Ban co the uong nuoc neu thay phu hop."],
        }
        if extra:
            config.update(extra)
        return EmotionSupportManager(config)

    def test_sad_confirmed_empathetic_chat_saves_result(self):
        io = MockIO(["Đúng", "4", "trò chuyện", "Mình buồn vì điểm thấp", "2"])
        session = EmotionSupportSession(manager=self._manager(), io=io)

        result = session.run("buon", 88.0, user_id="demo")

        self.assertTrue(result.completed)
        self.assertEqual(result.detected_emotion, "sad")
        self.assertEqual(result.confirmed_emotion, "sad")
        self.assertEqual(result.intensity_before, 4)
        self.assertEqual(result.intensity_after, 2)
        history = self._manager().load_history("demo")
        self.assertEqual(len(history), 1)

    def test_user_denies_detected_emotion_priority_to_user(self):
        io = MockIO(["Không, mình chỉ hơi mệt", "3", "1", "Mình ổn", "2"])
        session = EmotionSupportSession(manager=self._manager(), io=io)

        result = session.run("buon", 60.0, user_id="demo")

        self.assertEqual(result.confirmed_emotion, "neutral")

    def test_angry_user_selects_breathing(self):
        io = MockIO(["đúng", "5", "1", "3"])
        session = EmotionSupportSession(manager=self._manager(), io=io)

        result = session.run("gian", 91.0, user_id="demo")

        self.assertTrue(result.completed)
        self.assertEqual(result.selected_method, "breathing")
        self.assertEqual(result.intensity_after, 3)

    def test_breathing_can_be_cancelled(self):
        event = threading.Event()
        event.set()
        io = MockIO([])
        strategy = BreathingStrategy(
            self._manager().config,
            io,
            controller=NullRobotExpressionController(),
            cancel_event=event,
        )

        result = strategy.run(Emotion.ANGRY, 5)

        self.assertTrue(result.cancelled)

    def test_relaxation_audio_missing_file(self):
        io = MockIO(["rain"])
        strategy = RelaxationAudioStrategy(
            self._manager().config,
            io,
            controller=NullRobotExpressionController(),
        )

        result = strategy.run(Emotion.SAD, 4)

        self.assertFalse(result.completed)
        self.assertIn("missing_audio", result.notes)

    def test_invalid_intensity_is_reprompted(self):
        io = MockIO(["đúng", "9", "4", "trò chuyện", "ok", "2"])
        session = EmotionSupportSession(manager=self._manager(), io=io)

        result = session.run("buon", 70.0, user_id="demo")

        self.assertEqual(result.intensity_before, 4)

    def test_risk_content_triggers_safety(self):
        io = MockIO(["đúng", "4", "trò chuyện", "mình không muốn sống nữa"])
        session = EmotionSupportSession(manager=self._manager(), io=io)

        result = session.run("buon", 92.0, user_id="demo")

        self.assertFalse(result.completed)
        self.assertIn("safety", result.notes)
        self.assertTrue(detect_risk("tôi muốn tự tử").risk_detected)

    def test_missing_and_invalid_config_use_defaults(self):
        missing = load_config(Path(self.tmp.name) / "missing.json")
        self.assertTrue(missing["enabled"])

        bad_path = Path(self.tmp.name) / "bad.json"
        bad_path.write_text("{bad json", encoding="utf-8")
        bad = load_config(bad_path)
        self.assertTrue(bad["enabled"])

    def test_user_declines_all_methods(self):
        io = MockIO(["đúng", "3", "không cần"])
        session = EmotionSupportSession(manager=self._manager(), io=io)

        result = session.run("buon", 72.0, user_id="demo")

        self.assertTrue(result.cancelled)
        self.assertEqual(result.notes, "user_declined_methods")

    def test_history_prioritizes_effective_method(self):
        manager = self._manager(
            {
                "emotion_methods": {
                    "sad": ["empathetic_chat", "grounding"],
                    "unknown": ["empathetic_chat"],
                }
            }
        )
        path = manager.history_path("demo")
        path.write_text(
            json.dumps(
                {
                    "selected_method": "grounding",
                    "intensity_before": 5,
                    "intensity_after": 2,
                    "completed": True,
                    "cancelled": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        methods = manager.methods_for_emotion(Emotion.SAD, "demo")

        self.assertEqual(methods[0].value, "grounding")

    def test_support_trigger_skips_new_user(self):
        manager = self._manager()

        should_start = manager.should_start_support(
            "buon",
            90.0,
            user_id="new_user",
            profile_total_sessions=0,
        )

        self.assertFalse(should_start)

    def test_support_trigger_requires_target_emotion_and_confidence(self):
        manager = self._manager()

        self.assertTrue(
            manager.should_start_support(
                "gian",
                90.0,
                user_id="known_user",
                profile_total_sessions=1,
            )
        )
        self.assertFalse(
            manager.should_start_support(
                "vui",
                95.0,
                user_id="known_user",
                profile_total_sessions=1,
            )
        )
        self.assertFalse(
            manager.should_start_support(
                "buon",
                40.0,
                user_id="known_user",
                profile_total_sessions=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
