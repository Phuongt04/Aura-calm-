"""
model_loader.py
===============
Nạp model Keras/TF — hỗ trợ cả model cũ h5 (batch_shape legacy).
Tách riêng để dễ hoán đổi backend (TFLite, ONNX, ...) sau này.
"""
from __future__ import annotations

import json
import pickle
import h5py
from pathlib import Path

from tensorflow.keras.models import load_model, model_from_json
from . import config as cfg


# ── Helper: chuẩn hóa config Keras cũ ───────────────────────────────────────
def _normalize_keras_config(value):
    if isinstance(value, dict):
        if value.get("class_name") == "DTypePolicy":
            return value.get("config", {}).get("name", "float32")
        return {k: _normalize_keras_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_keras_config(i) for i in value]
    return value


def _load_legacy_h5(path: Path):
    """Nạp model h5 cũ có batch_shape thay vì batch_input_shape."""
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("model_config")
    if raw is None:
        raise ValueError(f"Không tìm thấy model_config trong {path}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    cfg_data = _normalize_keras_config(json.loads(raw))
    for layer in cfg_data.get("config", {}).get("layers", []):
        lc = layer.get("config", {})
        if layer.get("class_name") == "InputLayer" and "batch_shape" in lc:
            lc["batch_input_shape"] = lc.pop("batch_shape")
    model = model_from_json(json.dumps(cfg_data))
    model.load_weights(str(path))
    return model


def safe_load_keras(path: Path):
    """Nạp model, tự động xử lý lỗi tương thích batch_shape."""
    try:
        return load_model(str(path))
    except ValueError as exc:
        if "batch_shape" not in str(exc):
            raise
        print(f"  [model_loader] Chế độ legacy cho {path.name}...")
        return _load_legacy_h5(path)


# ── Nạp model âm thanh ───────────────────────────────────────────────────────
def load_audio_model():
    """Trả về (model, list[str] classes)."""
    if not cfg.AUDIO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model âm thanh: {cfg.AUDIO_MODEL_PATH}")
    if not cfg.AUDIO_LABEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file nhãn: {cfg.AUDIO_LABEL_PATH}")

    model = safe_load_keras(cfg.AUDIO_MODEL_PATH)
    with open(cfg.AUDIO_LABEL_PATH, "rb") as f:
        classes = list(pickle.load(f))

    num_out = model.output_shape[-1]
    if len(classes) != num_out:
        raise ValueError(
            f"Số nhãn trong file ({len(classes)}) "
            f"không khớp output model ({num_out})"
        )
    print(f"  [audio model] OK — Nhãn: {classes}")
    return model, classes


# ── Nạp model khuôn mặt ──────────────────────────────────────────────────────
def load_face_model():
    """Trả về model. Chỉ gọi khi FACE_ENABLED = True."""
    if not cfg.FACE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model khuôn mặt: {cfg.FACE_MODEL_PATH}")
    model = safe_load_keras(cfg.FACE_MODEL_PATH)
    print(f"  [face model]  OK — Input shape: {model.input_shape}")
    return model
