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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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

def _evaluate_head(model: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Tính loss và accuracy trên tập dữ liệu.
    Trả về (loss, accuracy).
    """
    if len(X) == 0:
        return 1e9, 0.0  # Trả về loss cao, accuracy 0 cho tập rỗng
    loss, acc = model.evaluate(X, y, verbose=0, batch_size=cfg.INCREMENTAL_BATCH_SIZE)
    return float(loss), float(acc)


def run_incremental_update(
    model:    tf.keras.Model,
    buffer:   FeedbackBuffer,
    user_id:  str,
) -> tf.keras.Model:
    """
    Thực hiện head-only update trên buffer hiện tại.

    Quy trình cải tiến:
      1. Chia buffer thành tập train/validation (80/20) để đánh giá khách quan.
      2. Backup weights cũ.
      3. Đo loss/accuracy trên tập validation (baseline).
      4. Tính class_weight để xử lý mất cân bằng dữ liệu.
      5. Train head trên tập train.
      6. Đo loss/accuracy trên tập validation sau khi train.
      7. Nếu validation loss TĂNG → rollback (dấu hiệu overfitting).
      8. Nếu không → lưu weights mới.
      9. Xả buffer.

    Trả về model (có thể đã được cập nhật hoặc rollback).
    """
    X_full, y_full = buffer.get_dataset()
    n = len(X_full)
    print(f"\n  ⚙️  [incremental] Bắt đầu update với {n} mẫu xác nhận...")

    # 1. Chia train/validation set (tỷ lệ 80/20)
    # stratify để đảm bảo phân phối nhãn tương đồng ở cả 2 tập
    # Nếu quá ít mẫu, dùng tất cả để train và chấp nhận rủi ro
    if n >= 5:  # Cần ít nhất 1 mẫu cho mỗi tập
        y_indices = np.argmax(y_full, axis=1)
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_indices
        )
        print(f"  Phân chia: {len(X_train)} train, {len(X_val)} validation.")
    else:
        X_train, X_val, y_train, y_val = X_full, np.array([]), y_full, np.array([])
        print("  Ít mẫu, dùng tất cả để train (không có validation).")

    # 2. Backup
    backup = _backup_path(user_id)
    model.save_weights(str(backup))

    # 3. Đánh giá trước trên validation set
    loss_before, acc_before = _evaluate_head(model, X_val, y_val)
    if len(X_val) > 0:
        print(f"  Validation trước: Loss={loss_before:.4f}, Acc={acc_before*100:.1f}%")

    # 4. Tính class weights trên tập train để xử lý mất cân bằng
    train_indices = np.argmax(y_train, axis=1)
    class_labels = np.unique(train_indices)
    class_weights_values = compute_class_weight(
        class_weight='balanced', classes=class_labels, y=train_indices
    )
    class_weights = dict(zip(class_labels, class_weights_values))
    print(f"  Class weights: { {buffer.classes[k]: round(v, 2) for k, v in class_weights.items()} }")

    # Freeze và compile
    freeze_backbone(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.INCREMENTAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 5. Train
    model.fit(
        X_train, y_train,
        epochs=cfg.INCREMENTAL_EPOCHS,
        batch_size=min(cfg.INCREMENTAL_BATCH_SIZE, len(X_train)),
        verbose=1,
        shuffle=True,
        class_weight=class_weights,
    )

    # 6. Đánh giá sau trên validation set
    loss_after, acc_after = _evaluate_head(model, X_val, y_val)
    if len(X_val) > 0:
        print(f"  Validation sau : Loss={loss_after:.4f}, Acc={acc_after*100:.1f}%")

    # 7. Quyết định rollback dựa trên validation loss
    # Nếu không có val set, mặc định là thành công
    should_rollback = len(X_val) > 0 and loss_after > loss_before

    if should_rollback:
        print("  ⚠️  Validation loss tăng → Dấu hiệu overfitting. Rollback về weights cũ!")
        model.load_weights(str(backup))
        _log(user_id, {
            "event": "rollback",
            "reason": "validation_loss_increase",
            "n_samples": n,
            "val_loss_before": round(loss_before, 4),
            "val_loss_after":  round(loss_after,  4),
            "val_acc_before": round(acc_before, 4),
            "val_acc_after":  round(acc_after,  4),
        })
    else:
        save_user_head(model, user_id)
        _log(user_id, {
            "event": "updated",
            "n_samples": n,
            "val_loss_before": round(loss_before, 4),
            "val_loss_after":  round(loss_after,  4),
            "val_acc_before": round(acc_before, 4),
            "val_acc_after":  round(acc_after,  4),
        })
        print(f"  ✅  Đã cập nhật model cho '{user_id}'")

    buffer.clear()
    return model
