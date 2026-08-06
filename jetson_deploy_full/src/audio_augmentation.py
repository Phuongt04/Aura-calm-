from pathlib import Path

import numpy as np

from .audio_preprocessing import read_audio, resample_audio, symmetric_pad_or_center_crop


def random_gain(y, rng, gain_min=0.8, gain_max=1.2):
    return (np.asarray(y, dtype=np.float32) * rng.uniform(gain_min, gain_max)).astype(np.float32)


def time_shift(y, rng, sample_rate=16000, max_shift_ms=200):
    y = np.asarray(y, dtype=np.float32)
    max_shift = int(sample_rate * max_shift_ms / 1000.0)
    if max_shift <= 0 or y.size == 0:
        return y
    shift = int(rng.randint(-max_shift, max_shift + 1))
    if shift == 0:
        return y
    return np.roll(y, shift).astype(np.float32)


def time_stretch(y, rng, rate_min=0.95, rate_max=1.05):
    try:
        import librosa
        rate = float(rng.uniform(rate_min, rate_max))
        stretched = librosa.effects.time_stretch(np.asarray(y, dtype=np.float32), rate=rate)
        return symmetric_pad_or_center_crop(stretched, len(y))
    except Exception:
        return np.asarray(y, dtype=np.float32)


def pitch_shift(y, rng, sample_rate=16000, max_semitones=1.0):
    try:
        import librosa
        steps = float(rng.uniform(-max_semitones, max_semitones))
        return librosa.effects.pitch_shift(np.asarray(y, dtype=np.float32), sr=sample_rate, n_steps=steps).astype(np.float32)
    except Exception:
        return np.asarray(y, dtype=np.float32)


def add_noise_snr(y, rng, snr_min=15.0, snr_max=30.0, noise=None):
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    if noise is None:
        noise = rng.normal(0.0, 1.0, size=y.shape).astype(np.float32)
    else:
        noise = symmetric_pad_or_center_crop(noise, len(y))
    signal_power = float(np.mean(y * y))
    noise_power = float(np.mean(noise * noise))
    if signal_power <= 1e-10 or noise_power <= 1e-10:
        return y
    snr_db = float(rng.uniform(snr_min, snr_max))
    desired_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scaled_noise = noise * np.sqrt(desired_noise_power / noise_power)
    return (y + scaled_noise).astype(np.float32)


class NoiseLibrary(object):
    def __init__(self, noise_dir=None, sample_rate=16000):
        self.sample_rate = sample_rate
        self.paths = []
        if noise_dir:
            root = Path(noise_dir)
            if root.exists():
                for suffix in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
                    self.paths.extend(sorted(root.rglob(suffix)))

    def sample(self, rng, length):
        if not self.paths:
            return None
        path = self.paths[int(rng.randint(0, len(self.paths)))]
        try:
            noise, sr = read_audio(path, sample_rate=self.sample_rate, mono=True)
            noise = resample_audio(noise, sr, self.sample_rate)
            return symmetric_pad_or_center_crop(noise, length)
        except Exception:
            return None


def augment_waveform(y, rng, config, noise_library=None, sample_rate=16000):
    aug_cfg = config.get("augmentation", {}) if isinstance(config, dict) else {}
    if not aug_cfg.get("enabled", True):
        return np.asarray(y, dtype=np.float32)
    out = np.asarray(y, dtype=np.float32)
    if rng.rand() < 0.8:
        out = random_gain(out, rng, aug_cfg.get("gain_min", 0.8), aug_cfg.get("gain_max", 1.2))
    if rng.rand() < 0.6:
        out = time_shift(out, rng, sample_rate, aug_cfg.get("time_shift_ms", 200))
    if rng.rand() < 0.35:
        out = time_stretch(out, rng, aug_cfg.get("time_stretch_min", 0.95), aug_cfg.get("time_stretch_max", 1.05))
    if rng.rand() < 0.35:
        out = pitch_shift(out, rng, sample_rate, aug_cfg.get("pitch_shift_max_semitones", 1))
    if rng.rand() < 0.6:
        real_noise = noise_library.sample(rng, len(out)) if noise_library is not None else None
        out = add_noise_snr(
            out,
            rng,
            aug_cfg.get("noise_snr_min", 15),
            aug_cfg.get("noise_snr_max", 30),
            real_noise,
        )
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = out / peak * 0.99
    return out.astype(np.float32)


def spec_augment(feature, rng, config):
    aug_cfg = config.get("augmentation", {}) if isinstance(config, dict) else {}
    if not aug_cfg.get("enabled", True):
        return feature
    x = np.array(feature, copy=True, dtype=np.float32)
    freq_max = int(aug_cfg.get("frequency_mask_max", 8))
    time_max = int(aug_cfg.get("time_mask_max", 20))
    if freq_max > 0 and rng.rand() < 0.6:
        width = int(rng.randint(1, min(freq_max, x.shape[0]) + 1))
        start = int(rng.randint(0, x.shape[0] - width + 1))
        x[start:start + width, :, :] = 0.0
    if time_max > 0 and rng.rand() < 0.6:
        width = int(rng.randint(1, min(time_max, x.shape[1]) + 1))
        start = int(rng.randint(0, x.shape[1] - width + 1))
        x[:, start:start + width, :] = 0.0
    return x.astype(np.float32)
