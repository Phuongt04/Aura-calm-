"""
audio.py
========
Voice pipeline: microphone or demo WAV -> preprocessing -> emotion model input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import librosa
import numpy as np

from . import config as cfg
from . import voice_io


@dataclass
class EmotionPredictionDetail:
    """Detailed prediction diagnostics for a sliding-window emotion pass."""

    emotion: str
    confidence: float
    window_count: int
    top2_emotion: str = ""
    top2_confidence: float = 0.0
    margin: float = 0.0
    agreement: float = 1.0
    window_votes: dict = field(default_factory=dict)

    @property
    def top1(self) -> str:
        """Alias for the winning emotion label."""
        return self.emotion

    @property
    def top1_confidence(self) -> float:
        """Alias for the winning emotion confidence."""
        return self.confidence

    @property
    def is_uncertain(self) -> bool:
        return (
            self.margin < cfg.AUDIO_UNCERTAIN_MARGIN
            or self.agreement < cfg.AUDIO_UNCERTAIN_AGREEMENT
        )


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
    if cfg.AUDIO_FEATURE_MODE == "cnn_gru_logmel_delta":
        return preprocess_cnn_gru_logmel(y)

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


def prevent_clipping(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale only clips that exceed the requested peak."""
    y = np.asarray(y, dtype=np.float32)
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs > peak:
        y = y / max_abs * peak
    return y.astype(np.float32)


def pad_or_crop(y: np.ndarray, n_samples: int | None = None) -> np.ndarray:
    """Center-pad or center-crop to the CNN-GRU duration."""
    y = np.asarray(y, dtype=np.float32)
    if n_samples is None:
        n_samples = int(cfg.AUDIO_SAMPLE_RATE * cfg.AUDIO_DURATION)
    if len(y) < n_samples:
        pad = n_samples - len(y)
        left = pad // 2
        return np.pad(y, (left, pad - left), mode="constant").astype(np.float32)
    if len(y) > n_samples:
        start = (len(y) - n_samples) // 2
        return y[start:start + n_samples].astype(np.float32)
    return y.astype(np.float32)


def preprocess_cnn_gru_logmel(y: np.ndarray) -> np.ndarray:
    """Match kaggle_angry_boost_train_cells.py inference features."""
    y = prevent_clipping(pad_or_crop(y))
    mel = librosa.feature.melspectrogram(
        y=np.asarray(y, dtype=np.float32),
        sr=cfg.AUDIO_SAMPLE_RATE,
        n_fft=cfg.AUDIO_N_FFT,
        win_length=cfg.AUDIO_WIN_LENGTH,
        hop_length=cfg.AUDIO_HOP_LENGTH,
        n_mels=cfg.AUDIO_N_MELS,
        fmin=cfg.AUDIO_FMIN,
        fmax=cfg.AUDIO_FMAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    delta = librosa.feature.delta(log_mel, width=9).astype(np.float32)
    delta2 = librosa.feature.delta(log_mel, width=9, order=2).astype(np.float32)
    x = np.stack([log_mel, delta, delta2], axis=-1)

    time_steps = 1 + int(cfg.AUDIO_SAMPLE_RATE * cfg.AUDIO_DURATION) // cfg.AUDIO_HOP_LENGTH
    if x.shape[1] < time_steps:
        x = np.pad(x, ((0, 0), (0, time_steps - x.shape[1]), (0, 0)), mode="edge")
    elif x.shape[1] > time_steps:
        x = x[:, :time_steps, :]

    mean = float(np.mean(x))
    std = float(np.std(x))
    if std > 1e-6:
        x = (x - mean) / std
    else:
        x = x - mean
    return x.astype(np.float32)


def reduce_noise_light(y: np.ndarray) -> np.ndarray:
    """Apply light denoising/high-pass behavior before feature extraction."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0 or not cfg.AUDIO_NOISE_REDUCTION_ENABLED:
        return y

    # Pre-emphasis attenuates steady low-frequency rumble from motors/fans.
    filtered = librosa.effects.preemphasis(y, coef=cfg.AUDIO_PREEMPHASIS_COEF)

    try:
        stft = librosa.stft(filtered, n_fft=1024, hop_length=256)
        mag, phase = np.abs(stft), np.exp(1j * np.angle(stft))
        frame_energy = np.mean(mag, axis=0)
        if frame_energy.size == 0:
            return filtered.astype(np.float32)

        quiet_limit = np.percentile(frame_energy, 25)
        quiet_frames = mag[:, frame_energy <= quiet_limit]
        if quiet_frames.size == 0:
            noise_profile = np.percentile(mag, 20, axis=1, keepdims=True)
        else:
            noise_profile = np.median(quiet_frames, axis=1, keepdims=True)

        cleaned_mag = np.maximum(
            mag - cfg.AUDIO_NOISE_REDUCTION_STRENGTH * noise_profile,
            0.0,
        )
        cleaned = librosa.istft(cleaned_mag * phase, hop_length=256, length=len(filtered))
        return cleaned.astype(np.float32)
    except Exception:
        return filtered.astype(np.float32)


def trim_silence(y: np.ndarray) -> np.ndarray:
    """Remove leading/trailing low-energy audio while keeping a small margin."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0 or not cfg.AUDIO_TRIM_SILENCE_ENABLED:
        return y

    frame_len = max(1, int(cfg.AUDIO_SAMPLE_RATE * cfg.AUDIO_TRIM_FRAME_MS / 1000.0))
    hop = max(1, int(cfg.AUDIO_SAMPLE_RATE * cfg.AUDIO_TRIM_HOP_MS / 1000.0))
    if len(y) <= frame_len:
        return y

    rms_values = []
    starts = []
    for start in range(0, len(y) - frame_len + 1, hop):
        frame = y[start:start + frame_len]
        rms_values.append(float(np.sqrt(np.mean(np.square(frame)))))
        starts.append(start)

    if not rms_values:
        return y

    rms = np.array(rms_values, dtype=np.float32)
    peak_rms = float(np.max(rms))
    if peak_rms <= 1e-6:
        return y

    threshold = max(
        cfg.VOICE_RECORD_RMS_THRESHOLD * 0.5,
        peak_rms * cfg.AUDIO_TRIM_THRESHOLD_RATIO,
    )
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return y

    padding = int(cfg.AUDIO_SAMPLE_RATE * cfg.AUDIO_TRIM_PADDING_SECONDS)
    start = max(0, starts[int(active[0])] - padding)
    end = min(len(y), starts[int(active[-1])] + frame_len + padding)
    if end <= start:
        return y
    return y[start:end].astype(np.float32)


def prepare_for_model(y: np.ndarray) -> np.ndarray:
    """Runtime audio cleanup used only for model inference."""
    if cfg.AUDIO_FEATURE_MODE == "cnn_gru_logmel_delta":
        y = np.asarray(y, dtype=np.float32)
        if len(y) == 0:
            return y
        if cfg.AUDIO_TRIM_SILENCE_ENABLED:
            try:
                trimmed, _ = librosa.effects.trim(y, top_db=35)
                if trimmed.size:
                    y = trimmed
            except Exception:
                y = trim_silence(y)
        return prevent_clipping(y)

    cleaned = reduce_noise_light(y)
    cleaned = trim_silence(cleaned)
    peak = np.max(np.abs(cleaned)) if len(cleaned) else 0.0
    if peak > 1.0:
        cleaned = cleaned / peak
    return cleaned.astype(np.float32)


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
    detail = predict_emotion_sliding_detail(model, classes, y)
    return detail.emotion, detail.confidence, detail.window_count


def _prediction_detail_from_probs(preds: np.ndarray, classes: list) -> EmotionPredictionDetail:
    """Build top-2, margin, and vote diagnostics from model probabilities."""
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)
    if preds.size == 0:
        return EmotionPredictionDetail("binhthuong", 0.0, 0)

    mean_preds = np.mean(preds, axis=0)
    order = np.argsort(mean_preds)[::-1]
    top1_idx = int(order[0])
    top2_idx = int(order[1]) if len(order) > 1 else top1_idx
    top1_conf = float(mean_preds[top1_idx]) * 100.0
    top2_conf = float(mean_preds[top2_idx]) * 100.0 if len(order) > 1 else 0.0

    window_winners = np.argmax(preds, axis=1)
    vote_counts = {}
    for idx in window_winners:
        label = classes[int(idx)]
        vote_counts[label] = vote_counts.get(label, 0) + 1
    agreement = max(vote_counts.values()) / len(window_winners) if len(window_winners) else 1.0

    return EmotionPredictionDetail(
        emotion=classes[top1_idx],
        confidence=top1_conf,
        window_count=int(preds.shape[0]),
        top2_emotion=classes[top2_idx] if len(order) > 1 else "",
        top2_confidence=top2_conf,
        margin=top1_conf - top2_conf,
        agreement=float(agreement),
        window_votes=vote_counts,
    )


def predict_emotion_sliding_detail(model, classes: list, y: np.ndarray) -> EmotionPredictionDetail:
    """Predict emotion and return top-2/window diagnostics."""
    windows = sliding_windows(y)
    batch = np.stack([preprocess(window) for window in windows], axis=0)
    preds = model.predict(batch, verbose=0)
    return _prediction_detail_from_probs(preds, classes)


def quality_stats(y: np.ndarray) -> tuple[float, float]:
    """Return (rms, peak) for a mono waveform."""
    if len(y) == 0:
        return 0.0, 0.0
    y = np.asarray(y, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(y))))
    peak = float(np.max(np.abs(y)))
    return rms, peak


def noise_floor_ratio(y: np.ndarray) -> float:
    """
    Estimate how constant the background is.

    Values near 1.0 mean the whole clip has similarly high energy, which often
    indicates motor/fan/noisy-room dominance rather than clear speech.
    """
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return 1.0

    frame_len = max(1, int(cfg.AUDIO_SAMPLE_RATE * 0.03))
    hop = max(1, frame_len // 2)
    rms_values = []
    for start in range(0, max(1, len(y) - frame_len + 1), hop):
        frame = y[start:start + frame_len]
        if len(frame) == 0:
            continue
        rms_values.append(float(np.sqrt(np.mean(np.square(frame)))))

    if not rms_values:
        return 1.0

    values = np.array(rms_values, dtype=np.float32)
    speech_level = float(np.percentile(values, 90))
    noise_floor = float(np.percentile(values, 20))
    if speech_level <= 1e-6:
        return 1.0
    return noise_floor / speech_level


def is_too_quiet(y: np.ndarray) -> bool:
    """Detect silence/very weak recordings that the emotion model misreads."""
    rms, peak = quality_stats(y)
    return rms < cfg.AUDIO_MIN_RMS or peak < cfg.AUDIO_MIN_PEAK


def is_too_noisy(y: np.ndarray) -> bool:
    """Detect clips likely dominated by steady background noise or clipping."""
    rms, peak = quality_stats(y)
    if rms > cfg.AUDIO_MAX_RMS or peak >= 0.99:
        return True
    return noise_floor_ratio(y) > cfg.AUDIO_NOISE_FLOOR_RATIO_MAX


def detect(model, classes: list) -> tuple:
    """Record audio, predict, and return (emotion_label, confidence_percent)."""
    raw_audio = record()
    model_audio = prepare_for_model(raw_audio)
    emotion, confidence, _ = predict_emotion_sliding(model, classes, model_audio)
    return emotion, confidence
