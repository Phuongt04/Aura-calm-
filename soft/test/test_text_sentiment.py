import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_chat.text_sentiment import infer_text_sentiment


class TextSentimentTests(unittest.TestCase):
    def test_detects_sad_content(self):
        result = infer_text_sentiment("Hôm nay mình áp lực và buồn quá")

        self.assertTrue(result.found)
        self.assertEqual(result.emotion, "buon")
        self.assertGreaterEqual(result.score, 2)

    def test_detects_angry_content(self):
        result = infer_text_sentiment("Mình rất bực bội và khó chịu")

        self.assertTrue(result.found)
        self.assertEqual(result.emotion, "gian")

    def test_negated_happy_is_not_happy(self):
        result = infer_text_sentiment("Mình không vui")

        self.assertNotEqual(result.emotion, "vui")


if __name__ == "__main__":
    unittest.main()
