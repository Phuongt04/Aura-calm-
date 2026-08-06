"""
Quick TTS test.

Run:
    python soft/test_tts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emotion_chat import config as cfg
from emotion_chat import voice_io


if __name__ == "__main__":
    print(f"TTS enabled: {cfg.VOICE_CHAT_TTS_ENABLED}")
    print(f"TTS backend: {cfg.VOICE_CHAT_TTS_BACKEND}")
    print(f"TTS voice filter: {cfg.VOICE_CHAT_TTS_VOICE or '(default)'}")
    print(f"Piper model: {cfg.VOICE_CHAT_PIPER_MODEL or '(not set)'}")
    print(f"Speed map: {cfg.VOICE_RESPONSE_SPEED_BY_EMOTION}")
    voice_io.speak(
        "Xin chao, day la MONE dang thu giong noi.",
        rate=cfg.VOICE_RESPONSE_SPEED_BY_EMOTION["binhthuong"],
    )
    print("Done.")
