import math
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000
DURATION_SECONDS = 4.0
NUM_SAMPLES = 64000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 64
FMIN = 50
FMAX = 8000


def _librosa():
    import librosa
    return librosa


def _read_wav_pcm(path):
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Unsupported WAV sample width: %s" % sample_width)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), int(sample_rate)


def read_audio(path, sample_rate=SAMPLE_RATE, mono=True):
    path = Path(path)
    try:
        librosa = _librosa()
        y, sr = librosa.load(str(path), sr=sample_rate, mono=mono)
        return np.asarray(y, dtype=np.float32), int(sr)
    except ModuleNotFoundError:
        if path.suffix.lower() != ".wav":
            raise
        y, sr = _read_wav_pcm(path)
        if sample_rate is not None and int(sr) != int(sample_rate):
            y = resample_audio(y, sr, sample_rate)
            sr = sample_rate
        return y.astype(np.float32), int(sr)


def to_mono(y):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    return np.mean(y, axis=0).astype(np.float32)


def resample_audio(y, orig_sr, target_sr=SAMPLE_RATE):
    if int(orig_sr) == int(target_sr):
        return np.asarray(y, dtype=np.float32)
    try:
        librosa = _librosa()
        return librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ModuleNotFoundError:
        y = np.asarray(y, dtype=np.float32)
        try:
            from scipy.signal import resample_poly
            gcd = math.gcd(int(orig_sr), int(target_sr))
            return resample_poly(y, int(target_sr) // gcd, int(orig_sr) // gcd).astype(np.float32)
        except Exception:
            old_x = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
            new_len = int(round(len(y) * float(target_sr) / float(orig_sr)))
            new_x = np.linspace(0.0, 1.0, num=max(1, new_len), endpoint=False)
            return np.interp(new_x, old_x, y).astype(np.float32)


def trim_silence(y, sample_rate=SAMPLE_RATE, top_db=35):
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    try:
        librosa = _librosa()
        trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        if trimmed.size > 0:
            return trimmed.astype(np.float32)
    except ModuleNotFoundError:
        frame = max(1, int(sample_rate * 0.03))
        hop = max(1, frame // 2)
        if y.size <= frame:
            return y
        rms = []
        starts = []
        for start in range(0, y.size - frame + 1, hop):
            chunk = y[start:start + frame]
            rms.append(float(np.sqrt(np.mean(chunk * chunk))))
            starts.append(start)
        if not rms:
            return y
        rms = np.asarray(rms, dtype=np.float32)
        threshold = max(float(np.max(rms)) * 0.08, 1e-5)
        active = np.flatnonzero(rms >= threshold)
        if active.size == 0:
            return y
        pad = int(sample_rate * 0.10)
        start = max(0, starts[int(active[0])] - pad)
        end = min(y.size, starts[int(active[-1])] + frame + pad)
        return y[start:end].astype(np.float32)
    except Exception:
        pass
    return y


def prevent_clipping(y, peak=0.99):
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    max_abs = float(np.max(np.abs(y)))
    if max_abs > peak:
        y = y / max_abs * peak
    return y.astype(np.float32)


def symmetric_pad_or_center_crop(y, num_samples=NUM_SAMPLES):
    y = np.asarray(y, dtype=np.float32)
    if y.size == num_samples:
        return y
    if y.size < num_samples:
        total_pad = num_samples - y.size
        left = total_pad // 2
        right = total_pad - left
        return np.pad(y, (left, right), mode="constant").astype(np.float32)
    start = (y.size - num_samples) // 2
    return y[start:start + num_samples].astype(np.float32)


def sliding_windows(y, sample_rate=SAMPLE_RATE, duration_seconds=DURATION_SECONDS, overlap=0.5):
    y = np.asarray(y, dtype=np.float32)
    window_samples = int(round(sample_rate * duration_seconds))
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if y.size <= window_samples:
        return [symmetric_pad_or_center_crop(y, window_samples)]

    overlap = min(max(float(overlap), 0.0), 0.95)
    hop = max(1, int(round(window_samples * (1.0 - overlap))))
    windows = []
    for start in range(0, y.size - window_samples + 1, hop):
        windows.append(y[start:start + window_samples].astype(np.float32))
    last = y[-window_samples:].astype(np.float32)
    if not windows or not np.array_equal(windows[-1], last):
        windows.append(last)
    return windows


def normalize_feature(x):
    x = np.asarray(x, dtype=np.float32)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-6:
        return x - mean
    return (x - mean) / std


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax):
    fmax = min(float(fmax), float(sample_rate) / 2.0)
    mel_points = np.linspace(_hz_to_mel(float(fmin)), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / float(sample_rate)).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / float(center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / float(right - center)
    return fb


def _fallback_melspectrogram(y, sample_rate, n_fft, hop_length, n_mels, fmin, fmax):
    y = np.asarray(y, dtype=np.float32)
    padded = np.pad(y, (n_fft // 2, n_fft // 2), mode="constant")
    if padded.size < n_fft:
        padded = np.pad(padded, (0, n_fft - padded.size), mode="constant")
    frame_count = 1 + int((padded.size - n_fft) // hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    power = np.empty((n_fft // 2 + 1, frame_count), dtype=np.float32)
    for idx in range(frame_count):
        start = idx * hop_length
        frame = padded[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame, n=n_fft)
        power[:, idx] = (np.abs(spectrum) ** 2).astype(np.float32)
    mel_fb = _mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
    mel = np.dot(mel_fb, power)
    mel = np.maximum(mel, 1e-10)
    ref = max(float(np.max(mel)), 1e-10)
    return (10.0 * np.log10(mel / ref)).astype(np.float32)


def _fallback_delta(x, order=1):
    out = np.asarray(x, dtype=np.float32)
    for _ in range(order):
        padded = np.pad(out, ((0, 0), (1, 1)), mode="edge")
        out = (padded[:, 2:] - padded[:, :-2]) * 0.5
    return out.astype(np.float32)


def log_mel_delta_features(
    y,
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    fmin=FMIN,
    fmax=FMAX,
):
    try:
        librosa = _librosa()
        mel = librosa.feature.melspectrogram(
            y=np.asarray(y, dtype=np.float32),
            sr=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        width = min(9, log_mel.shape[1] if log_mel.shape[1] % 2 == 1 else log_mel.shape[1] - 1)
        if width < 3:
            delta = np.zeros_like(log_mel, dtype=np.float32)
            delta2 = np.zeros_like(log_mel, dtype=np.float32)
        else:
            delta = librosa.feature.delta(log_mel, width=width).astype(np.float32)
            delta2 = librosa.feature.delta(log_mel, width=width, order=2).astype(np.float32)
    except ModuleNotFoundError:
        log_mel = _fallback_melspectrogram(y, sample_rate, n_fft, hop_length, n_mels, fmin, fmax)
        delta = _fallback_delta(log_mel, order=1)
        delta2 = _fallback_delta(log_mel, order=2)
    channels = [normalize_feature(log_mel), normalize_feature(delta), normalize_feature(delta2)]
    return np.stack(channels, axis=-1).astype(np.float32)


def preprocess_waveform(
    y,
    sample_rate=SAMPLE_RATE,
    target_sample_rate=SAMPLE_RATE,
    duration_seconds=DURATION_SECONDS,
    trim=True,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    fmin=FMIN,
    fmax=FMAX,
):
    y = to_mono(y)
    y = resample_audio(y, sample_rate, target_sample_rate)
    if trim:
        y = trim_silence(y, target_sample_rate)
    y = prevent_clipping(y)
    num_samples = int(round(target_sample_rate * duration_seconds))
    y = symmetric_pad_or_center_crop(y, num_samples)
    return log_mel_delta_features(
        y,
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )


def preprocess_file(path, config=None):
    audio_cfg = (config or {}).get("audio", {}) if isinstance(config, dict) else {}
    sample_rate = int(audio_cfg.get("sample_rate", SAMPLE_RATE))
    y, sr = read_audio(path, sample_rate=sample_rate, mono=True)
    return preprocess_waveform(
        y,
        sample_rate=sr,
        target_sample_rate=sample_rate,
        duration_seconds=float(audio_cfg.get("duration_seconds", DURATION_SECONDS)),
        n_fft=int(audio_cfg.get("n_fft", N_FFT)),
        win_length=int(audio_cfg.get("win_length", WIN_LENGTH)),
        hop_length=int(audio_cfg.get("hop_length", HOP_LENGTH)),
        n_mels=int(audio_cfg.get("n_mels", N_MELS)),
        fmin=int(audio_cfg.get("fmin", FMIN)),
        fmax=int(audio_cfg.get("fmax", FMAX)),
    )


def preprocess_file_sliding(path, config=None, overlap=0.5):
    audio_cfg = (config or {}).get("audio", {}) if isinstance(config, dict) else {}
    sample_rate = int(audio_cfg.get("sample_rate", SAMPLE_RATE))
    duration = float(audio_cfg.get("duration_seconds", DURATION_SECONDS))
    y, sr = read_audio(path, sample_rate=sample_rate, mono=True)
    y = trim_silence(y, sr)
    y = prevent_clipping(y)
    windows = sliding_windows(y, sample_rate=sr, duration_seconds=duration, overlap=overlap)
    return np.stack([
        log_mel_delta_features(
            window,
            sample_rate=sr,
            n_fft=int(audio_cfg.get("n_fft", N_FFT)),
            win_length=int(audio_cfg.get("win_length", WIN_LENGTH)),
            hop_length=int(audio_cfg.get("hop_length", HOP_LENGTH)),
            n_mels=int(audio_cfg.get("n_mels", N_MELS)),
            fmin=int(audio_cfg.get("fmin", FMIN)),
            fmax=int(audio_cfg.get("fmax", FMAX)),
        )
        for window in windows
    ], axis=0)


def expected_time_steps(sample_rate=SAMPLE_RATE, duration_seconds=DURATION_SECONDS, hop_length=HOP_LENGTH):
    num_samples = int(round(sample_rate * duration_seconds))
    return int(1 + math.floor(num_samples / float(hop_length)))
