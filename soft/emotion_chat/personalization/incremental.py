"""
incremental.py
==============
On-device incremental learning cho model cảm xúc — chạy được trên Jetson Nano.

Chiến lược: FROZEN FEATURE EXTRACTOR + HEAD-ONLY UPDATE
  ┌─────────────────────────────────────────┐
  │  CNN + BiLSTM  (FROZEN — không đổi)     │  ← phần nặng, đã học tốt rồi
  │  Dense(128) + BN + Dropout              │
  ├─────────────────────────────────────────┤
  │  Dense(n_classes, softmax)  ← CHỈ TẦY  │  ← chỉ update lớp này, rất nhanh
  └─────────────────────────────────────────┘

Tại sao đủ để cá nhân hóa?
  - Tầng CNN/LSTM đã học biểu diễn âm thanh tổng quát.
  - Tầng cuối học cách MAP biểu diễn đó → nhãn theo ĐẶC THÙ của từng người.
  - Mỗi người nói khác nhau (giọng, nhịp, ngữ điệu) — tầng cuối cần điều chỉnh.

Luồng:
  1. Sau khi user xác nhận nhãn → lưu vào feedback buffer (RAM).
  2. Sau INCREMENTAL_TRIGGER_N mẫu xác nhận → thực hiện head-only update.
  3. Backup weights cũ trước khi update — tự động rollback nếu accuracy giảm.
  4. Lưu weights mới vào user_data/{user_id}/head_weights.h5.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from .. import config as cfg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _user_dir(user_id: str) -> Path:
    d = cfg.USER_DATA_DIR / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _head_weights_path(user_id: str) -> Path:
    return _user_dir(user_id) / "head_weights.h5"


def _backup_path(user_id: str) -> Path:
    return _user_dir(user_id) / "head_weights_backup.h5"


def _log_path(user_id: str) -> Path:
    return _user_dir(user_id) / "incremental_log.jsonl"


def _log(user_id: str, record: dict) -> None:
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(_log_path(user_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Quản lý model head ────────────────────────────────────────────────────────

def freeze_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """
    Đóng băng tất cả lớp trừ lớp Dense cuối cùng (classification head).
    Trả về đúng model đó (in-place).
    """
    last_dense_idx = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            last_dense_idx = i

    if last_dense_idx is None:
        raise ValueError("Không tìm thấy lớp Dense nào trong model.")

    for i, layer in enumerate(model.layers):
        layer.trainable = (i == last_dense_idx)

    trainable_params = sum(
        np.prod(v.shape) for v in model.trainable_variables
    )
    print(
        f"  [incremental] Freeze xong — chỉ train {trainable_params:,} tham số "
        f"(lớp '{model.layers[last_dense_idx].name}')"
    )
    return model


def load_user_head(model: tf.keras.Model, user_id: str) -> tf.keras.Model:
    """
    Nạp weights đầu phân loại đã cá nhân hóa của user (nếu có).
    Nếu chưa có → dùng weights gốc của model.
    """
    path = _head_weights_path(user_id)
    if path.exists():
        try:
            model.load_weights(str(path), by_name=False, skip_mismatch=False)
            print(f"  [incremental] Đã nạp head weights của '{user_id}'")
        except Exception as exc:
            print(f"  [incremental] Không nạp được head weights: {exc} — dùng mặc định")
    return model


def save_user_head(model: tf.keras.Model, user_id: str) -> None:
    """Lưu toàn bộ weights model (chỉ head thay đổi, phần còn lại giữ nguyên)."""
    path = _head_weights_path(user_id)
    model.save_weights(str(path))


# ── Buffer per-user (in-memory, lưu xuống khi đủ batch) ──────────────────────

class FeedbackBuffer:
    """
    Buffer RAM lưu các mẫu đã xác nhận của 1 user trong phiên hiện tại.
    Khi đủ INCREMENTAL_TRIGGER_N mẫu → train, rồi xả buffer.
    """

    def __init__(self, user_id: str, classes: list):
        self.user_id = user_id
        self.classes = classes
        self._X: list = []   # list of np.ndarray (N_FRAMES, H, W, 1)
        self._y: list = []   # list of int (class index)

        # Nạp lại buffer chưa được train từ phiên trước (nếu có)
        self._load_pending()

    def add(self, features: np.ndarray, label: str) -> int:
        """Thêm 1 mẫu. Trả về tổng số mẫu hiện có trong buffer."""
        if label not in self.classes:
            print(f"  [buffer] Nhãn '{label}' không hợp lệ — bỏ qua")
            return len(self._X)
        self._X.append(features)
        self._y.append(self.classes.index(label))
        self._save_pending()
        return len(self._X)

    def ready(self) -> bool:
        return len(self._X) >= cfg.INCREMENTAL_TRIGGER_N

    def get_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array(self._X, dtype=np.float32)
        y = tf.keras.utils.to_categorical(self._y, num_classes=len(self.classes))
        return X, y

    def clear(self) -> None:
        self._X.clear()
        self._y.clear()
        pending = _user_dir(self.user_id) / "_pending_buffer.npz"
        if pending.exists():
            pending.unlink()

    # ── Persistence (giữ buffer qua restart) ─────────────────────────────
    def _pending_path(self) -> Path:
        return _user_dir(self.user_id) / "_pending_buffer.npz"

    def _save_pending(self) -> None:
        if not self._X:
            return
        np.savez_compressed(
            str(self._pending_path()),
            X=np.array(self._X, dtype=np.float32),
            y=np.array(self._y, dtype=np.int32),
        )

    def _load_pending(self) -> None:
        p = self._pending_path()
        if not p.exists():
            return
        try:
            data = np.load(str(p), allow_pickle=False)
            self._X = list(data["X"])
            self._y = list(data["y"])
            print(
                f"  [buffer] Nạp lại {len(self._X)} mẫu chưa train "
                f"từ phiên trước của '{self.user_id}'"
            )
        except Exception:
            pass


# ── Incremental training ──────────────────────────────────────────────────────

def _evaluate_head(model: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> float:
    """Tính accuracy trên tập buffer (dùng làm baseline và validate)."""
    preds = model.predict(X, verbose=0)
    correct = int(np.sum(np.argmax(preds, axis=1) == np.argmax(y, axis=1)))
    return correct / len(X)


def run_incremental_update(
    model:    tf.keras.Model,
    buffer:   FeedbackBuffer,
    user_id:  str,
) -> tf.keras.Model:
    """
    Thực hiện head-only update trên buffer hiện tại.

    Quy trình:
      1. Backup weights cũ
      2. Đo accuracy trước
      3. Train đúng lớp cuối
      4. Đo accuracy sau
      5. Nếu giảm → rollback tự động
      6. Nếu tăng → lưu weights mới
      7. Xả buffer

    Trả về model (có thể đã được cập nhật hoặc rollback).
    """
    X, y = buffer.get_dataset()
    n    = len(X)
    print(f"\n  ⚙️  [incremental] Bắt đầu update với {n} mẫu xác nhận...")

    # Backup
    backup = _backup_path(user_id)
    model.save_weights(str(backup))

    # Accuracy trước
    acc_before = _evaluate_head(model, X, y)
    print(f"  Accuracy trước: {acc_before*100:.1f}%")

    # Freeze và compile
    freeze_backbone(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.INCREMENTAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train — số epoch nhỏ, không cần validation set riêng
    model.fit(
        X, y,
        epochs=cfg.INCREMENTAL_EPOCHS,
        batch_size=min(cfg.INCREMENTAL_BATCH_SIZE, n),
        verbose=1,
        shuffle=True,
    )

    # Accuracy sau
    acc_after = _evaluate_head(model, X, y)
    print(f"  Accuracy sau : {acc_after*100:.1f}%")

    if acc_after < acc_before - 0.05:   # cho phép giảm tối đa 5%
        print("  ⚠️  Accuracy giảm quá nhiều → Rollback về weights cũ!")
        model.load_weights(str(backup))
        _log(user_id, {
            "event": "rollback",
            "n_samples": n,
            "acc_before": round(acc_before, 4),
            "acc_after":  round(acc_after,  4),
        })
    else:
        save_user_head(model, user_id)
        _log(user_id, {
            "event": "updated",
            "n_samples": n,
            "acc_before": round(acc_before, 4),
            "acc_after":  round(acc_after,  4),
        })
        print(f"  ✅  Đã cập nhật model cho '{user_id}'")

    buffer.clear()
    return model
