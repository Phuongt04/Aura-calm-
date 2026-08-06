import json
from pathlib import Path


DEFAULT_CONFIG = {
    "audio": {
        "sample_rate": 16000,
        "duration_seconds": 4.0,
        "n_fft": 512,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": 64,
        "fmin": 50,
        "fmax": 8000,
    },
    "split": {
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_seed": 42,
    },
    "training": {
        "batch_size": 16,
        "stage1_epochs": 60,
        "stage1_learning_rate": 0.0003,
        "stage2_epochs": 20,
        "stage2_learning_rate": 0.00001,
        "label_smoothing": 0.05,
        "early_stopping_patience": 8,
    },
    "augmentation": {
        "enabled": True,
        "gain_min": 0.8,
        "gain_max": 1.2,
        "time_shift_ms": 200,
        "time_stretch_min": 0.95,
        "time_stretch_max": 1.05,
        "pitch_shift_max_semitones": 1,
        "noise_snr_min": 15,
        "noise_snr_max": 30,
        "frequency_mask_max": 8,
        "time_mask_max": 20,
    },
    "sampling": {
        "tess_weight": 0.25,
        "vietnamese_kaggle_weight": 0.55,
        "self_recorded_weight": 0.20,
    },
    "inference": {
        "confidence_threshold": 0.60,
        "margin_threshold": 0.15,
        "window_overlap": 0.50,
    },
}


def _coerce_value(value):
    value = value.strip()
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _simple_yaml_load(text):
    data = {}
    current = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith(" "):
            key = raw_line.rstrip(":").strip()
            data[key] = {}
            current = key
            continue
        if current is None or ":" not in raw_line:
            continue
        key, value = raw_line.strip().split(":", 1)
        data[current][key.strip()] = _coerce_value(value)
    return data


def deep_update(base, updates):
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    config = DEFAULT_CONFIG
    if path is None:
        return config
    path = Path(path)
    if not path.exists():
        return config
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        try:
            import yaml
            loaded = yaml.safe_load(text) or {}
        except Exception:
            loaded = _simple_yaml_load(text)
    return deep_update(config, loaded)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
