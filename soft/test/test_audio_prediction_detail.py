import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_chat import config as cfg

try:
    from emotion_chat.audio import _prediction_detail_from_probs
except ModuleNotFoundError as exc:
    _prediction_detail_from_probs = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class AudioPredictionDetailTests(unittest.TestCase):
    @unittest.skipIf(_prediction_detail_from_probs is None, "audio dependency unavailable: %s" % _IMPORT_ERROR)
    def test_top2_margin_and_agreement(self):
        old_margin = cfg.AUDIO_UNCERTAIN_MARGIN
        old_agreement = cfg.AUDIO_UNCERTAIN_AGREEMENT
        cfg.AUDIO_UNCERTAIN_MARGIN = 12.0
        cfg.AUDIO_UNCERTAIN_AGREEMENT = 0.55
        try:
            detail = _prediction_detail_from_probs(
                np.array(
                    [
                        [0.52, 0.48, 0.00],
                        [0.51, 0.49, 0.00],
                        [0.49, 0.51, 0.00],
                    ],
                    dtype=np.float32,
                ),
                ["vui", "binhthuong", "buon"],
            )
        finally:
            cfg.AUDIO_UNCERTAIN_MARGIN = old_margin
            cfg.AUDIO_UNCERTAIN_AGREEMENT = old_agreement

        self.assertEqual(detail.emotion, "vui")
        self.assertEqual(detail.top2_emotion, "binhthuong")
        self.assertLess(detail.margin, 12.0)
        self.assertTrue(detail.is_uncertain)
        self.assertAlmostEqual(detail.agreement, 2 / 3)


if __name__ == "__main__":
    unittest.main()
