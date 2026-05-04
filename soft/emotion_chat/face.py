"""
face.py
=======
Pipeline khuôn mặt: chụp camera → phát hiện mặt → tiền xử lý → dự đoán cảm xúc.
Sử dụng Haar Cascade có sẵn trong OpenCV (không cần cài thêm gì).

Để tích hợp model mặt:
  1. Đặt FACE_ENABLED = True trong config.py
  2. Đặt đúng đường dẫn FACE_MODEL_PATH
  3. Điền FACE_RAW_CLASSES theo đúng thứ tự output của model
  4. Điều chỉnh FACE_LABEL_MAP nếu nhãn model khác FER2013
"""
from __future__ import annotations

import numpy as np
import cv2

from . import config as cfg

# Lazy-load cascade để tránh tải khi module chưa cần dùng
_face_cascade = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def capture_frame(camera_index: int = 0):
    """
    Chụp 1 frame từ camera.
    Trả về frame (np.ndarray BGR) hoặc None nếu không mở được camera.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"  [face] Không thể mở camera (index={camera_index}).")
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def preprocess(frame: np.ndarray):
    """
    Phát hiện mặt → cắt vùng mặt lớn nhất → resize → normalize.
    Trả về (H, W, 1) float32 hoặc None nếu không tìm thấy mặt.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None

    # Lấy khuôn mặt lớn nhất (diện tích w*h)
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face_roi    = gray[y: y + h, x: x + w]
    face_resize = cv2.resize(face_roi, (cfg.FACE_IMG_SIZE, cfg.FACE_IMG_SIZE))
    face_norm   = face_resize.astype("float32") / 255.0
    return np.expand_dims(face_norm, axis=-1)   # (H, W, 1)


def _map_label(raw_label: str) -> str:
    """Ánh xạ nhãn gốc của model mặt → nhãn chuẩn hệ thống."""
    return cfg.FACE_LABEL_MAP.get(raw_label.lower(), "binhthuong")


def detect(model, face_raw_classes: list, camera_index: int = 0) -> tuple:
    """
    Chụp ảnh → nhận diện khuôn mặt → dự đoán cảm xúc.

    Trả về:
        (nhãn_chuẩn: str, độ_tin_cậy%: float)  — khi thành công
        (None, None)                             — khi không có camera / không thấy mặt

    face_raw_classes: list nhãn theo đúng thứ tự output của model (từ FACE_RAW_CLASSES).
    """
    frame = capture_frame(camera_index)
    if frame is None:
        return None, None

    processed = preprocess(frame)
    if processed is None:
        print("  [face] Không phát hiện khuôn mặt trong frame.")
        return None, None

    inp       = np.expand_dims(processed, axis=0)   # (1, H, W, 1)
    preds     = model.predict(inp, verbose=0)[0]
    idx       = int(np.argmax(preds))
    raw_label = face_raw_classes[idx]
    mapped    = _map_label(raw_label)
    conf      = float(preds[idx]) * 100

    print(f"  [face] {raw_label} → {mapped} ({conf:.1f}%)")
    return mapped, conf
