"""
voice_io.py
===========
Speech input/output helpers for assistant-style conversations.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import sounddevice as sd
import speech_recognition as sr

from . import config as cfg

_tts_engine = None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").strip()


def _record_fixed_int16(seconds: float) -> np.ndarray:
    audio = sd.rec(
        int(seconds * cfg.AUDIO_SAMPLE_RATE),
        samplerate=cfg.AUDIO_SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    return np.asarray(audio).reshape(-1)


def record_until_silence_int16(
    max_seconds: float | None = None,
    silence_seconds: float | None = None,
    min_seconds: float | None = None,
) -> np.ndarray:
    """Record until the speaker pauses, with a max duration as a safety stop."""
    max_seconds = max_seconds or cfg.VOICE_RECORD_MAX_SECONDS
    silence_seconds = silence_seconds or cfg.VOICE_RECORD_SILENCE_SECONDS
    min_seconds = min_seconds or cfg.VOICE_RECORD_MIN_SECONDS

    chunk_seconds = 0.2
    chunk_size = int(cfg.AUDIO_SAMPLE_RATE * chunk_seconds)
    max_chunks = max(1, int(max_seconds / chunk_seconds))
    silence_chunks_needed = max(1, int(silence_seconds / chunk_seconds))
    min_chunks = max(1, int(min_seconds / chunk_seconds))

    chunks: list[np.ndarray] = []
    speech_started = False
    silent_chunks = 0

    with sd.InputStream(
        samplerate=cfg.AUDIO_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
    ) as stream:
        for index in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            mono = np.asarray(chunk).reshape(-1)
            chunks.append(mono.copy())

            rms = float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2)))
            is_voice = rms >= cfg.VOICE_RECORD_RMS_THRESHOLD

            if is_voice:
                speech_started = True
                silent_chunks = 0
            elif speech_started:
                silent_chunks += 1

            if (
                speech_started
                and index + 1 >= min_chunks
                and silent_chunks >= silence_chunks_needed
            ):
                break

    if not chunks or not speech_started:
        return np.array([], dtype=np.int16)
    return np.concatenate(chunks).astype(np.int16)


def _transcribe_int16(mono: np.ndarray) -> str:
    if mono.size == 0:
        return ""

    recognizer = sr.Recognizer()
    data = sr.AudioData(mono.tobytes(), cfg.AUDIO_SAMPLE_RATE, 2)

    try:
        text = recognizer.recognize_google(data, language=cfg.VOICE_CHAT_LANGUAGE)
    except sr.UnknownValueError:
        print("  [voice] Minh chua nghe ro.")
        return ""
    except sr.RequestError as exc:
        print(f"  [voice] Khong goi duoc dich vu nhan dang giong noi: {exc}")
        return ""

    text = text.strip()
    print(f"  Ban noi: {text}")
    return text


def listen_once(prompt: str | None = None) -> str:
    """Record one utterance until silence and transcribe Vietnamese speech."""
    prompt = prompt or f"{cfg.VOICE_ASSISTANT_NAME} dang nghe..."
    print(
        f"\n  {prompt} "
        f"(noi xong thi im lang {cfg.VOICE_RECORD_SILENCE_SECONDS:.1f}s de ket thuc)"
    )
    return _transcribe_int16(record_until_silence_int16())


def wait_for_wake_word() -> bool:
    """Listen in short windows until the assistant name is heard."""
    if not cfg.WAKE_WORD_ENABLED:
        return True

    print(
        f"\n[Dang cho goi ten: {', '.join(cfg.WAKE_WORDS)}] "
        "Noi 'thoat' de dung."
    )

    empty_count = 0
    while True:
        text = _transcribe_int16(_record_fixed_int16(cfg.WAKE_LISTEN_SECONDS))
        normalized = _normalize(text)

        if any(phrase in normalized for phrase in {"thoat", "dung lai", "tam biet"}):
            return False

        if any(word in normalized for word in cfg.WAKE_WORDS):
            print(f"  [{cfg.VOICE_ASSISTANT_NAME}] Minh nghe day.")
            speak("Minh nghe day.")
            return True

        empty_count += 1
        if cfg.WAKE_MAX_EMPTY > 0 and empty_count >= cfg.WAKE_MAX_EMPTY:
            return False


def speak(text: str) -> None:
    """Speak text when TTS is enabled; otherwise only print is used by caller."""
    global _tts_engine

    if not cfg.VOICE_CHAT_TTS_ENABLED:
        return

    try:
        import pyttsx3

        if _tts_engine is None:
            _tts_engine = pyttsx3.init()
            _tts_engine.setProperty("rate", 165)
        _tts_engine.say(text)
        _tts_engine.runAndWait()
    except Exception as exc:
        print(f"  [voice] TTS khong kha dung: {exc}")
