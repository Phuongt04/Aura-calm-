"""
voice_io.py
===========
Speech input/output helpers for assistant-style conversations.
"""
from __future__ import annotations

import unicodedata
import re
import os
import subprocess
import sys
import shutil
import tempfile
import socket

import numpy as np

from . import config as cfg

_tts_engine = None
_sapi_voice = None


def _clean_for_tts(text: str) -> str:
    """Remove display-only markup so speech sounds more natural."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[_#>|~]+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:()/%+\-À-ỹĐđ]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _select_tts_voice(engine) -> None:
    wanted = cfg.VOICE_CHAT_TTS_VOICE.strip().lower()
    if not wanted:
        return

    for voice in engine.getProperty("voices"):
        haystack = " ".join(
            str(value).lower()
            for value in (
                getattr(voice, "id", ""),
                getattr(voice, "name", ""),
                " ".join(getattr(voice, "languages", []) or []),
            )
        )
        if wanted in haystack:
            engine.setProperty("voice", voice.id)
            return


def _speak_windows_sapi(text: str, rate: int) -> bool:
    """Speak through Windows SAPI using PowerShell."""
    if sys.platform != "win32":
        return False

    env = os.environ.copy()
    env["AURA_TTS_TEXT"] = text
    env["AURA_TTS_RATE"] = str(rate)
    env["AURA_TTS_VOICE"] = cfg.VOICE_CHAT_TTS_VOICE
    script = r"""
$synth = New-Object -ComObject SAPI.SpVoice
$rate = [int]([Environment]::GetEnvironmentVariable('AURA_TTS_RATE'))
$voice = [Environment]::GetEnvironmentVariable('AURA_TTS_VOICE')
if ($voice) {
    $match = $synth.GetVoices() |
        Where-Object { $_.GetDescription() -like "*$voice*" } |
        Select-Object -First 1
    if ($match) { $synth.Voice = $match }
}
$synth.Rate = [Math]::Max(-10, [Math]::Min(10, [int](($rate - 165) / 20)))
[void]$synth.Speak([Environment]::GetEnvironmentVariable('AURA_TTS_TEXT'))
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def _select_sapi_voice(voice) -> None:
    wanted = cfg.VOICE_CHAT_TTS_VOICE.strip().lower()
    if not wanted:
        return

    for index in range(voice.GetVoices().Count):
        candidate = voice.GetVoices().Item(index)
        if wanted in candidate.GetDescription().lower():
            voice.Voice = candidate
            return


def _speak_windows_com(text: str, rate: int) -> bool:
    """Fast in-process Windows SAPI path using pywin32/win32com."""
    global _sapi_voice

    if sys.platform != "win32":
        return False

    try:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        if _sapi_voice is None:
            _sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
            _select_sapi_voice(_sapi_voice)
        _sapi_voice.Rate = max(-10, min(10, int((rate - 165) / 20)))
        _sapi_voice.Speak(text)
        return True
    except Exception:
        _sapi_voice = None
        return False


def _speak_pyttsx3(text: str, rate: int) -> bool:
    global _tts_engine

    try:
        import pyttsx3

        if _tts_engine is None:
            _tts_engine = pyttsx3.init()
            _select_tts_voice(_tts_engine)
        _tts_engine.setProperty("rate", rate)
        _tts_engine.say(text)
        _tts_engine.runAndWait()
        return True
    except Exception:
        return False


def _speak_linux_command(text: str, rate: int) -> bool:
    """Speak on Linux/Jetson using a system TTS command."""
    if not sys.platform.startswith("linux"):
        return False

    command = cfg.VOICE_CHAT_TTS_COMMAND.strip()
    voice = cfg.VOICE_CHAT_TTS_VOICE.strip() or "vi"

    if command:
        try:
            subprocess.run(
                command.split() + [text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except Exception:
            return False

    if shutil.which("espeak-ng"):
        speed = max(80, min(450, int(rate)))
        try:
            subprocess.run(
                ["espeak-ng", "-v", voice, "-s", str(speed), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except Exception:
            return False

    if shutil.which("espeak"):
        speed = max(80, min(450, int(rate)))
        try:
            subprocess.run(
                ["espeak", "-v", voice, "-s", str(speed), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except Exception:
            return False

    if shutil.which("spd-say"):
        try:
            subprocess.run(
                ["spd-say", "-l", voice, text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except Exception:
            return False

    return False


def _play_audio_file(path: str) -> bool:
    players = (
        ("aplay", [path]),
        ("paplay", [path]),
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", path]),
    )
    for executable, args in players:
        if not shutil.which(executable):
            continue
        try:
            subprocess.run(
                [executable] + args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except Exception:
            continue
    return False


def _piper_length_scale(rate: int) -> float:
    base = max(80, min(450, int(rate)))
    scale = cfg.VOICE_CHAT_PIPER_LENGTH_SCALE * (cfg.VOICE_CHAT_TTS_RATE / float(base))
    return max(0.55, min(1.8, float(scale)))


def _speak_piper(text: str, rate: int) -> bool:
    """Speak with Piper TTS on Linux/Jetson."""
    if not sys.platform.startswith("linux"):
        return False

    piper_bin = cfg.VOICE_CHAT_PIPER_BIN.strip() or "piper"
    if not shutil.which(piper_bin):
        return False

    model_path = cfg.VOICE_CHAT_PIPER_MODEL
    if not model_path or not model_path.exists():
        return False

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    command = [
        piper_bin,
        "--model",
        str(model_path),
        "--output_file",
        wav_path,
        "--length_scale",
        str(_piper_length_scale(rate)),
        "--noise_scale",
        str(cfg.VOICE_CHAT_PIPER_NOISE_SCALE),
    ]
    if cfg.VOICE_CHAT_PIPER_SPEAKER.strip():
        command.extend(["--speaker", cfg.VOICE_CHAT_PIPER_SPEAKER.strip()])
    if cfg.VOICE_CHAT_PIPER_USE_CUDA:
        command.append("--cuda")

    try:
        subprocess.run(
            command,
            input=text,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return _play_audio_file(wav_path)
    except Exception:
        return False
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").strip()


def _record_fixed_int16(seconds: float) -> np.ndarray:
    import sounddevice as sd

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
    import sounddevice as sd

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

    import speech_recognition as sr

    recognizer = sr.Recognizer()
    data = sr.AudioData(mono.tobytes(), cfg.AUDIO_SAMPLE_RATE, 2)

    old_timeout = socket.getdefaulttimeout()
    try:
        if cfg.VOICE_CHAT_STT_TIMEOUT > 0:
            socket.setdefaulttimeout(cfg.VOICE_CHAT_STT_TIMEOUT)
        text = recognizer.recognize_google(data, language=cfg.VOICE_CHAT_LANGUAGE)
    except sr.UnknownValueError:
        print("  [voice] Minh chua nghe ro.")
        return ""
    except sr.RequestError as exc:
        print(f"  [voice] Khong goi duoc dich vu nhan dang giong noi: {exc}")
        return ""
    except (TimeoutError, OSError) as exc:
        print(f"  [voice] Dich vu nhan dang giong noi bi timeout/loi mang: {exc}")
        return ""
    except Exception as exc:
        print(f"  [voice] Loi nhan dang giong noi: {exc}")
        return ""
    finally:
        socket.setdefaulttimeout(old_timeout)

    text = text.strip()
    print(f"  Ban noi: {text}")
    return text


def listen_once_with_audio(prompt: str | None = None) -> tuple[str, np.ndarray]:
    """Record one utterance, return recognized text and raw float32 audio."""
    prompt = prompt or f"{cfg.VOICE_ASSISTANT_NAME} dang nghe..."
    print(
        f"\n  {prompt} "
        f"(noi xong thi im lang {cfg.VOICE_RECORD_SILENCE_SECONDS:.1f}s de ket thuc)"
    )
    audio_i16 = record_until_silence_int16()
    text = _transcribe_int16(audio_i16)
    raw_audio = (audio_i16.astype(np.float32) / 32768.0).flatten()
    return text, raw_audio


def listen_once(prompt: str | None = None) -> str:
    """Record one utterance until silence and transcribe Vietnamese speech."""
    text, _ = listen_once_with_audio(prompt)
    return text


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


def speak(text: str, rate: int | None = None) -> None:
    """Speak text when TTS is enabled; otherwise only print is used by caller."""
    if not cfg.VOICE_CHAT_TTS_ENABLED:
        return

    text = _clean_for_tts(text)
    if not text:
        return
    rate = int(rate or cfg.VOICE_CHAT_TTS_RATE)

    backend = cfg.VOICE_CHAT_TTS_BACKEND.strip().lower()
    if backend in {"windows", "sapi"}:
        if _speak_windows_com(text, rate) or _speak_windows_sapi(text, rate) or _speak_pyttsx3(text, rate):
            return
    elif backend == "piper":
        if _speak_piper(text, rate) or _speak_linux_command(text, rate) or _speak_pyttsx3(text, rate):
            return
    elif backend in {"linux", "espeak", "espeak-ng", "spd-say"}:
        if _speak_piper(text, rate) or _speak_linux_command(text, rate) or _speak_pyttsx3(text, rate):
            return
    else:
        if (
            _speak_pyttsx3(text, rate)
            or _speak_piper(text, rate)
            or _speak_linux_command(text, rate)
            or _speak_windows_com(text, rate)
            or _speak_windows_sapi(text, rate)
        ):
            return

    print("  [voice] TTS khong kha dung.")
