"""
List installed Windows SAPI voices.

Run:
    python soft/list_tts_voices.py
"""
from __future__ import annotations

import sys


def main() -> None:
    if sys.platform != "win32":
        print("Windows SAPI voices are only available on Windows.")
        return

    try:
        import win32com.client

        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voices = voice.GetVoices()
        for index in range(voices.Count):
            print(voices.Item(index).GetDescription())
    except Exception as exc:
        print(f"Could not list voices: {exc}")


if __name__ == "__main__":
    main()
