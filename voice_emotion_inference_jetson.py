import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_preprocessing import (
    DURATION_SECONDS,
    SAMPLE_RATE,
    log_mel_delta_features,
    prevent_clipping,
    read_audio,
    sliding_windows,
    trim_silence,
)
from src.config_utils import load_config
from src.voice_emotion_labels import LABELS


class VoiceEmotionRecognizer(object):
    def __init__(self, model_path, label_path=None, config_path=None, thresholds_path=None):
        self.config = load_config(config_path or (PROJECT_ROOT / "config" / "voice_emotion_config.yaml"))
        self.model = tf.keras.models.load_model(str(model_path), compile=False)
        self.labels = self._load_labels(label_path)
        self.thresholds = self._load_thresholds(thresholds_path)
        self._warm_up()

    def _load_labels(self, label_path):
        candidates = []
        if label_path:
            candidates.append(Path(label_path))
        candidates.append(PROJECT_ROOT / "models" / "voice_emotion_label_mapping.pkl")
        candidates.append(PROJECT_ROOT / "label_mapping_final.pkl")
        for path in candidates:
            if path.exists():
                try:
                    with path.open("rb") as handle:
                        labels = pickle.load(handle)
                    labels = [str(item) for item in labels]
                    if len(labels) == 4:
                        old_to_new = {
                            "binhthuong": "neutral",
                            "vui": "happy",
                            "buon": "sad",
                            "gian": "angry",
                        }
                        return [old_to_new.get(label, label) for label in labels]
                except Exception:
                    pass
        return list(LABELS)

    def _load_thresholds(self, thresholds_path):
        default_cfg = self.config.get("inference", {})
        thresholds = {
            "confidence_threshold": float(default_cfg.get("confidence_threshold", 0.60)),
            "margin_threshold": float(default_cfg.get("margin_threshold", 0.15)),
        }
        path = Path(thresholds_path) if thresholds_path else PROJECT_ROOT / "config" / "voice_emotion_thresholds.json"
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                thresholds["confidence_threshold"] = float(loaded.get("confidence_threshold", thresholds["confidence_threshold"]))
                thresholds["margin_threshold"] = float(loaded.get("margin_threshold", thresholds["margin_threshold"]))
            except Exception:
                pass
        return thresholds

    def _warm_up(self):
        shape = self.model.input_shape
        dummy_shape = [1]
        for dim in shape[1:]:
            dummy_shape.append(1 if dim is None else int(dim))
        dummy = np.zeros(dummy_shape, dtype=np.float32)
        for _ in range(3):
            self.model.predict(dummy, batch_size=1, verbose=0)

    def _features_from_waveform(self, waveform, sample_rate):
        audio_cfg = self.config.get("audio", {})
        target_sr = int(audio_cfg.get("sample_rate", SAMPLE_RATE))
        duration = float(audio_cfg.get("duration_seconds", DURATION_SECONDS))
        overlap = float(self.config.get("inference", {}).get("window_overlap", 0.50))
        y = np.asarray(waveform, dtype=np.float32).flatten()
        y = trim_silence(y, target_sr)
        y = prevent_clipping(y)
        windows = sliding_windows(y, sample_rate=target_sr, duration_seconds=duration, overlap=overlap)
        features = []
        for window in windows:
            features.append(log_mel_delta_features(
                window,
                sample_rate=target_sr,
                n_fft=int(audio_cfg.get("n_fft", 512)),
                win_length=int(audio_cfg.get("win_length", 400)),
                hop_length=int(audio_cfg.get("hop_length", 160)),
                n_mels=int(audio_cfg.get("n_mels", 64)),
                fmin=int(audio_cfg.get("fmin", 50)),
                fmax=int(audio_cfg.get("fmax", 8000)),
            ))
        return np.stack(features, axis=0).astype(np.float32)

    def predict_waveform(self, waveform, sample_rate=16000):
        t0 = time.time()
        features = self._features_from_waveform(waveform, sample_rate)
        t1 = time.time()
        probabilities = []
        for index in range(features.shape[0]):
            pred = self.model.predict(features[index:index + 1], batch_size=1, verbose=0)[0]
            probabilities.append(pred)
        t2 = time.time()
        mean_prob = np.mean(np.asarray(probabilities, dtype=np.float32), axis=0)
        order = np.argsort(mean_prob)[::-1]
        top1 = int(order[0])
        top2 = int(order[1]) if len(order) > 1 else top1
        confidence = float(mean_prob[top1])
        margin = float(mean_prob[top1] - mean_prob[top2])
        is_uncertain = (
            confidence < self.thresholds["confidence_threshold"]
            or margin < self.thresholds["margin_threshold"]
        )
        emotion = "uncertain" if is_uncertain else self.labels[top1]
        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": {
                self.labels[i]: float(mean_prob[i]) for i in range(min(len(self.labels), len(mean_prob)))
            },
            "is_uncertain": bool(is_uncertain),
            "preprocessing_time_ms": (t1 - t0) * 1000.0,
            "inference_time_ms": (t2 - t1) * 1000.0,
            "total_time_ms": (t2 - t0) * 1000.0,
        }

    def predict_file(self, audio_path):
        target_sr = int(self.config.get("audio", {}).get("sample_rate", SAMPLE_RATE))
        waveform, sr = read_audio(audio_path, sample_rate=target_sr, mono=True)
        return self.predict_waveform(waveform, sr)


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson-compatible voice emotion inference.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "voice_emotion_vietnamese_finetuned.h5")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "models" / "voice_emotion_label_mapping.pkl")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "voice_emotion_config.yaml")
    parser.add_argument("--thresholds", type=Path, default=PROJECT_ROOT / "config" / "voice_emotion_thresholds.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recognizer = VoiceEmotionRecognizer(args.model, args.labels, args.config, args.thresholds)
    result = recognizer.predict_file(args.audio)
    print(json.dumps(result, ensure_ascii=False, indent=2))
