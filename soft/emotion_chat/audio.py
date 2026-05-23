"""
audio.py
========
Voice pipeline: microphone or demo WAV -> preprocessing -> emotion model input.
"""
from __future__ import annotations

import cv2
import librosa
import numpy as np

from . import config as cfg
from . import voice_io


def record() -> np.ndarray:
    """Return a mono float32 waveform from a demo file or the microphone."""
    if cfg.AUDIO_DEMO_PATH and cfg.AUDIO_DEMO_PATH.is_file():
        print(f"\n[demo audio] Using file: {cfg.AUDIO_DEMO_PATH}")
        y, _ = librosa.load(
            str(cfg.AUDIO_DEMO_PATH),
            sr=cfg.AUDIO_SAMPLE_RATE,
            mono=True,
            duration=cfg.AUDIO_DURATION,
        )
        return y.astype(np.float32)

    print(
        f"\n  Dang thu am cam xuc "
        f"(toi da {cfg.VOICE_RECORD_MAX_SECONDS:.0f}s, dung khi im lang "
        f"{cfg.VOICE_RECORD_SILENCE_SECONDS:.1f}s)..."
    )
    audio_i16 = voice_io.record_until_silence_int16()
    print("  Da ghi xong.")
    return (audio_i16.astype(np.float32) / 32768.0).flatten()


def preprocess(y: np.ndarray) -> np.ndarray:
    """Convert raw audio into (N_FRAMES, IMG_SIZE, IMG_SIZE, 1) float32."""
    req_len = int(cfg.AUDIO_SAMPLE_RATE * cfg.AUDIO_DURATION)
    y = np.pad(y, (0, max(0, req_len - len(y))))[:req_len]
    chunk_len = len(y) // cfg.AUDIO_N_FRAMES
    frames = []
    for i in range(cfg.AUDIO_N_FRAMES):
        chunk = y[i * chunk_len: (i + 1) * chunk_len]
        mel = librosa.feature.melspectrogram(
            y=chunk, sr=cfg.AUDIO_SAMPLE_RATE, n_mels=cfg.AUDIO_IMG_SIZE
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_r = cv2.resize(mel_db, (cfg.AUDIO_IMG_SIZE, cfg.AUDIO_IMG_SIZE))
        denom = mel_r.max() - mel_r.min() or 1.0
        frames.append((mel_r - mel_r.min()) / denom)
    return np.expand_dims(np.array(frames, dtype=np.float32), axis=-1)


def sliding_windows(y: np.ndarray) -> list[np.ndarray]:
    """Split variable-length speech into fixed-length windows for the model."""
    y = np.asarray(y, dtype=np.float32)
    window_len = int(cfg.AUDIO_SAMPLE_RATE * cfg.AUDIO_DURATION)
    stride_len = max(1, int(cfg.AUDIO_SAMPLE_RATE * cfg.AUDIO_WINDOW_STRIDE))

    if len(y) <= window_len:
        return [np.pad(y, (0, max(0, window_len - len(y))))[:window_len]]

    windows = []
    for start in range(0, len(y) - window_len + 1, stride_len):
        windows.append(y[start:start + window_len])

    last = y[-window_len:]
    if not windows or not np.array_equal(windows[-1], last):
        windows.append(last)

    return windows


def predict_emotion_sliding(model, classes: list, y: np.ndarray) -> tuple[str, float, int]:
    """Predict emotion from any-length speech by averaging window predictions."""
    windows = sliding_windows(y)
    batch = np.stack([preprocess(window) for window in windows], axis=0)
    preds = model.predict(batch, verbose=0)
    mean_preds = np.mean(preds, axis=0)
    idx = int(np.argmax(mean_preds))
    return classes[idx], float(mean_preds[idx]) * 100, len(windows)


def quality_stats(y: np.ndarray) -> tuple[float, float]:
    """Return (rms, peak) for a mono waveform."""
    if len(y) == 0:
        return 0.0, 0.0
    y = np.asarray(y, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(y))))
    peak = float(np.max(np.abs(y)))
    return rms, peak


def is_too_quiet(y: np.ndarray) -> bool:
    """Detect silence/very weak recordings that the emotion model misreads."""
    rms, peak = quality_stats(y)
    return rms < cfg.AUDIO_MIN_RMS or peak < cfg.AUDIO_MIN_PEAK


def detect(model, classes: list) -> tuple:
    """Record audio, predict, and return (emotion_label, confidence_percent)."""
    raw_audio = record()
    emotion, confidence, _ = predict_emotion_sliding(model, classes, raw_audio)
    return emotion, confidence
