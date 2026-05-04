"""
audio.py
========
Pipeline giọng nói: thu âm microphone → tiền xử lý → dự đoán cảm xúc.
"""
from __future__ import annotations

import numpy as np
import sounddevice as sd
import librosa
import cv2

from . import config as cfg


def record() -> np.ndarray:
    """Thu âm từ microphone, trả về mảng 1-D float32."""
    print(f"\n🎤  Đang thu âm ({cfg.AUDIO_DURATION:.0f}s)... Hãy nói gì đó!")
    audio = sd.rec(
        int(cfg.AUDIO_DURATION * cfg.AUDIO_SAMPLE_RATE),
        samplerate=cfg.AUDIO_SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("  Đã ghi xong.")
    return audio.flatten()


def preprocess(y: np.ndarray) -> np.ndarray:
    """Chuyển raw audio → (N_FRAMES, IMG_SIZE, IMG_SIZE, 1) float32."""
    req_len = int(cfg.AUDIO_SAMPLE_RATE * cfg.AUDIO_DURATION)
    y = np.pad(y, (0, max(0, req_len - len(y))))[:req_len]
    chunk_len = len(y) // cfg.AUDIO_N_FRAMES
    frames = []
    for i in range(cfg.AUDIO_N_FRAMES):
        chunk  = y[i * chunk_len: (i + 1) * chunk_len]
        mel    = librosa.feature.melspectrogram(
            y=chunk, sr=cfg.AUDIO_SAMPLE_RATE, n_mels=cfg.AUDIO_IMG_SIZE
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_r  = cv2.resize(mel_db, (cfg.AUDIO_IMG_SIZE, cfg.AUDIO_IMG_SIZE))
        denom  = mel_r.max() - mel_r.min() or 1.0
        frames.append((mel_r - mel_r.min()) / denom)
    return np.expand_dims(np.array(frames, dtype=np.float32), axis=-1)  # (10,64,64,1)


def detect(model, classes: list) -> tuple:
    """
    Thu âm → dự đoán → trả về (nhãn_cảm_xúc: str, độ_tin_cậy%: float).
    Nhãn là một trong các key của EMOTION_CONFIG trong config.py.
    """
    raw_audio = record()
    inp       = np.expand_dims(preprocess(raw_audio), axis=0)   # (1,10,64,64,1)
    preds     = model.predict(inp, verbose=0)[0]
    idx       = int(np.argmax(preds))
    return classes[idx], float(preds[idx]) * 100
