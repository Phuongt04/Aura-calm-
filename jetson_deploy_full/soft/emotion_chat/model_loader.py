"""
model_loader.py
===============
Nạp model Keras/TF — hỗ trợ cả model cũ h5 (batch_shape legacy).
Tách riêng để dễ hoán đổi backend (TFLite, ONNX, ...) sau này.
"""
from __future__ import annotations

import json
import pickle
import tempfile
import zipfile
import h5py
import numpy as np
from pathlib import Path

from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import MobileNetV2
from . import config as cfg


# ── Helper: chuẩn hóa config Keras cũ ───────────────────────────────────────
def _normalize_keras_config(value):
    if isinstance(value, dict):
        if "quantization_config" in value:
            value = {k: v for k, v in value.items() if k != "quantization_config"}
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
    try:
        model = model_from_json(json.dumps(cfg_data))
    except TypeError:
        model = _build_audio_model()
    model.load_weights(str(path))
    return model


def _build_audio_model():
    """Rebuild the known CNN+BiLSTM audio architecture for old H5 weights."""
    return Sequential([
        layers.InputLayer(shape=(cfg.AUDIO_N_FRAMES, cfg.AUDIO_IMG_SIZE, cfg.AUDIO_IMG_SIZE, 1)),
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding="same", activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding="same", activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Conv2D(128, (3, 3), padding="same", activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Flatten()),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.4),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(4, activation="softmax"),
    ])


def _build_cnn_gru_audio_model():
    """Rebuild the CNN-GRU log-mel model used by the angry-boost checkpoint."""
    time_steps = 1 + int(cfg.AUDIO_SAMPLE_RATE * cfg.AUDIO_DURATION) // cfg.AUDIO_HOP_LENGTH
    input_shape = (cfg.AUDIO_N_MELS, time_steps, 3)
    inputs = layers.Input(shape=input_shape, name="logmel_delta_input")

    x = layers.SeparableConv2D(32, (3, 3), padding="same", name="cnn_block1_sepconv")(inputs)
    x = layers.BatchNormalization(name="cnn_block1_bn")(x)
    x = layers.ReLU(name="cnn_block1_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_block1_pool")(x)

    x = layers.SeparableConv2D(64, (3, 3), padding="same", name="cnn_block2_sepconv")(x)
    x = layers.BatchNormalization(name="cnn_block2_bn")(x)
    x = layers.ReLU(name="cnn_block2_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_block2_pool")(x)

    x = layers.SeparableConv2D(128, (3, 3), padding="same", name="cnn_block3_sepconv")(x)
    x = layers.BatchNormalization(name="cnn_block3_bn")(x)
    x = layers.ReLU(name="cnn_block3_relu")(x)

    x = layers.Permute((2, 1, 3), name="time_major")(x)
    x = layers.Reshape((time_steps // 4, (cfg.AUDIO_N_MELS // 4) * 128), name="time_sequence")(x)
    x = layers.GRU(64, name="gru_64")(x)
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    x = layers.Dropout(0.35, name="dropout_035")(x)
    outputs = layers.Dense(len(cfg.AUDIO_RAW_CLASSES), activation="softmax", name="emotion_softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="CNN_GRU_Vietnamese_SER")


def _load_cnn_gru_keras_archive(path: Path):
    """Load a Keras .keras archive by rebuilding architecture and loading weights."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(path) as archive:
            archive.extract("model.weights.h5", tmp_dir)
        model = _build_cnn_gru_audio_model()
        model.load_weights(str(tmp_dir / "model.weights.h5"))
        return model


def _build_mobilenet_face_model():
    """Rebuild the MobileNetV2 face model used by best_model.h5."""
    inputs = layers.Input(
        shape=(cfg.FACE_IMG_SIZE, cfg.FACE_IMG_SIZE, 3),
        name="input_layer_1",
    )
    base = MobileNetV2(
        input_shape=(cfg.FACE_IMG_SIZE, cfg.FACE_IMG_SIZE, 3),
        include_top=False,
        weights=None,
        alpha=1.0,
    )
    x = base(inputs)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(512, activation="relu", name="dense")(x)
    x = layers.BatchNormalization(name="batch_normalization_1")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    outputs = layers.Dense(len(cfg.FACE_RAW_CLASSES), activation="softmax", name="dense_1")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="functional")


def _load_mobilenet_face_h5(path: Path):
    model = _build_mobilenet_face_model()
    _load_h5_weights_by_layer_name(path, model)
    return model


def _weight_dataset_name(weight) -> str:
    return weight.name.split("/")[-1].split(":")[0]


def _set_layer_weights_from_group(layer, group) -> None:
    if not layer.weights:
        return
    if layer.name not in group:
        return
    layer_group = group[layer.name]
    if layer.name in layer_group and isinstance(layer_group[layer.name], h5py.Group):
        layer_group = layer_group[layer.name]

    values = []
    for weight in layer.weights:
        dataset_name = _weight_dataset_name(weight)
        if dataset_name not in layer_group:
            return
        values.append(np.asarray(layer_group[dataset_name]))
    layer.set_weights(values)


def _load_h5_group_by_layer_name(model, group) -> None:
    for layer in model.layers:
        if isinstance(layer, models.Model):
            if layer.name in group:
                _load_h5_group_by_layer_name(layer, group[layer.name])
            continue
        _set_layer_weights_from_group(layer, group)


def _load_h5_weights_by_layer_name(path: Path, model) -> None:
    with h5py.File(path, "r") as handle:
        group = handle["model_weights"] if "model_weights" in handle else handle
        _load_h5_group_by_layer_name(model, group)


def safe_load_keras(path: Path):
    """Nạp model, tự động xử lý lỗi tương thích batch_shape."""
    try:
        return load_model(str(path), compile=False)
    except (TypeError, ValueError) as exc:
        message = str(exc)
        legacy_markers = ("batch_shape", "Could not locate class", "DTypePolicy", "quantization_config")
        if not any(marker in message for marker in legacy_markers):
            raise
        print(f"  [model_loader] legacy mode for {path.name}...")
        return _load_legacy_h5(path)


def safe_load_audio_model(path: Path):
    """Load the configured audio model, including the CNN-GRU .keras checkpoint."""
    if cfg.AUDIO_FEATURE_MODE == "cnn_gru_logmel_delta" and path.suffix.lower() == ".keras":
        try:
            return load_model(str(path), compile=False)
        except Exception:
            print(f"  [model_loader] rebuilding CNN-GRU audio model for {path.name}...")
            return _load_cnn_gru_keras_archive(path)

    try:
        return safe_load_keras(path)
    except Exception as exc:
        if cfg.AUDIO_FEATURE_MODE != "cnn_gru_logmel_delta":
            raise
        message = str(exc)
        if path.suffix.lower() != ".keras" and "quantization_config" not in message:
            raise
        print(f"  [model_loader] rebuilding CNN-GRU audio model for {path.name}...")
        return _load_cnn_gru_keras_archive(path)


def safe_load_face_model(path: Path):
    """Load the configured face model, with fallback for Keras-version H5 drift."""
    try:
        return safe_load_keras(path)
    except Exception as exc:
        message = str(exc)
        fallback_markers = ("quantization_config", "Layer count mismatch")
        if path.suffix.lower() != ".h5" or not any(marker in message for marker in fallback_markers):
            raise
        print(f"  [model_loader] rebuilding MobileNetV2 face model for {path.name}...")
        return _load_mobilenet_face_h5(path)


# ── Nạp model âm thanh ───────────────────────────────────────────────────────
def load_audio_model():
    """Trả về (model, list[str] classes)."""
    if not cfg.AUDIO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model âm thanh: {cfg.AUDIO_MODEL_PATH}")
    if cfg.AUDIO_FEATURE_MODE != "cnn_gru_logmel_delta" and not cfg.AUDIO_LABEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file nhãn: {cfg.AUDIO_LABEL_PATH}")

    model = safe_load_audio_model(cfg.AUDIO_MODEL_PATH)
    if cfg.AUDIO_FEATURE_MODE == "cnn_gru_logmel_delta":
        classes = [
            cfg.AUDIO_LABEL_MAP.get(str(label).lower(), str(label).lower())
            for label in cfg.AUDIO_RAW_CLASSES
        ]
    else:
        with open(cfg.AUDIO_LABEL_PATH, "rb") as f:
            classes = [str(label) for label in pickle.load(f)]

    num_out = model.output_shape[-1]
    if len(classes) != num_out:
        raise ValueError(
            f"Số nhãn trong file ({len(classes)}) "
            f"không khớp output model ({num_out})"
        )
    print(f"  [audio model] OK - labels: {classes}")
    return model, classes


# ── Nạp model khuôn mặt ──────────────────────────────────────────────────────
def load_face_model():
    """Trả về model. Chỉ gọi khi FACE_ENABLED = True."""
    if not cfg.FACE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model khuôn mặt: {cfg.FACE_MODEL_PATH}")
    model = safe_load_face_model(cfg.FACE_MODEL_PATH)
    expected_input = (None, cfg.FACE_IMG_SIZE, cfg.FACE_IMG_SIZE, 3)
    if tuple(model.input_shape) != expected_input:
        raise ValueError(
            f"Face model input shape {model.input_shape} does not match {expected_input}"
        )
    if model.output_shape[-1] != len(cfg.FACE_RAW_CLASSES):
        raise ValueError(
            f"Face model output classes {model.output_shape[-1]} "
            f"does not match FACE_RAW_CLASSES {len(cfg.FACE_RAW_CLASSES)}"
        )
    print(
        f"  [face model] OK - input shape: {model.input_shape}, "
        f"labels: {cfg.FACE_RAW_CLASSES}"
    )
    return model
