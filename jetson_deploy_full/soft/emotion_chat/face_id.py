"""
face_id.py
==========
Nhận dạng DANH TÍNH người dùng qua khuôn mặt (khác với face.py chuyên nhận cảm xúc).
Sử dụng OpenCV LBPH Face Recognizer — chạy offline, không cần cài thêm thư viện.

Cấu trúc dữ liệu:
  user_data/
    face_model/
      lbph_model.yml      ← model LBPH đã train (tự động cập nhật khi enroll)
      label_map.json      ← {"0": "alice", "1": "bob", ...}
    {user_id}/
      face_samples/
        0.png, 1.png, ... ← ảnh grayscale khuôn mặt đã cắt + resize

Luồng đăng ký (enroll):
  1. Chụp FACE_ID_SAMPLES_N frame liên tiếp từ camera
  2. Phát hiện mặt bằng Haar Cascade → cắt + resize về FACE_ID_IMG_SIZE
  3. Lưu vào face_samples/, sau đó retrain LBPH ngay lập tức

Luồng nhận dạng (recognize):
  1. Nạp model LBPH đã train (hoặc skip nếu chưa có ai đăng ký)
  2. Chụp 1 frame → phát hiện mặt → predict
  3. confidence < FACE_ID_THRESHOLD  → trả về user_id (đã biết)
     confidence ≥ FACE_ID_THRESHOLD  → trả về None (người lạ / chưa đăng ký)

Ghi chú về LBPH:
  - confidence là khoảng cách (thấp hơn = khớp tốt hơn, ngược với xác suất)
  - Ngưỡng 80 thường phù hợp cho ánh sáng phòng bình thường
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as cfg

# Kích thước chuẩn hóa ảnh khuôn mặt dùng cho LBPH
_FACE_SIZE = 100  # pixel

# Lazy-load Haar Cascade
_cascade: cv2.CascadeClassifier | None = None


# ── Helpers nội bộ ────────────────────────────────────────────────────────────

def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(path)
    return _cascade


def _face_model_dir() -> Path:
    d = cfg.USER_DATA_DIR / "face_model"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _samples_dir(user_id: str) -> Path:
    d = cfg.USER_DATA_DIR / user_id / "face_samples"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _label_map_path() -> Path:
    return _face_model_dir() / "label_map.json"


def _model_path() -> Path:
    return _face_model_dir() / "lbph_model.yml"


def _load_label_map() -> dict[str, str]:
    """Trả về {str(int_label): user_id}."""
    p = _label_map_path()
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_label_map(label_map: dict[str, str]) -> None:
    _label_map_path().write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _detect_face(frame: np.ndarray) -> np.ndarray | None:
    """
    Phát hiện khuôn mặt lớn nhất trong frame.
    Trả về ảnh grayscale đã resize (H, W) hoặc None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    roi = gray[y: y + h, x: x + w]
    return cv2.resize(roi, (_FACE_SIZE, _FACE_SIZE))


# ── Train / retrain ───────────────────────────────────────────────────────────

def _collect_all_samples() -> tuple[list[np.ndarray], list[int], dict[str, str]]:
    """
    Duyệt qua tất cả thư mục face_samples/ đã có.
    Trả về (faces, labels, label_map) để train LBPH.
    """
    faces: list[np.ndarray] = []
    labels: list[int] = []
    label_map: dict[str, str] = {}   # {str(int): user_id}

    # Xây dựng label_map từ file cũ để giữ nguyên index
    existing = _load_label_map()
    # Đảo ngược: user_id → int_label
    user_to_int: dict[str, int] = {v: int(k) for k, v in existing.items()}
    next_label = max(user_to_int.values(), default=-1) + 1

    if not cfg.USER_DATA_DIR.exists():
        return faces, labels, label_map

    for user_dir in sorted(cfg.USER_DATA_DIR.iterdir()):
        if not user_dir.is_dir() or user_dir.name == "face_model":
            continue
        samples_dir = user_dir / "face_samples"
        if not samples_dir.exists():
            continue

        user_id = user_dir.name
        if user_id not in user_to_int:
            user_to_int[user_id] = next_label
            next_label += 1

        int_label = user_to_int[user_id]
        label_map[str(int_label)] = user_id

        for img_path in sorted(samples_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(int_label)

    return faces, labels, label_map


def _retrain() -> None:
    """Train lại LBPH trên toàn bộ face samples và lưu model."""
    faces, labels, label_map = _collect_all_samples()
    if len(faces) < 2:
        # Cần ít nhất 2 mẫu để train
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    recognizer.save(str(_model_path()))
    _save_label_map(label_map)
    print(f"  [face_id] Đã train LBPH — {len(faces)} mẫu, {len(label_map)} người dùng.")


# ── API công khai ─────────────────────────────────────────────────────────────

def enroll_user(user_id: str, n_samples: int | None = None) -> bool:
    """
    Đăng ký khuôn mặt cho user_id bằng cách chụp n_samples ảnh từ camera.

    Trả về True nếu đủ mẫu, False nếu không phát hiện đủ khuôn mặt.
    """
    if n_samples is None:
        n_samples = cfg.FACE_ID_SAMPLES_N

    samples_dir = _samples_dir(user_id)
    # Xóa mẫu cũ nếu có
    for old in samples_dir.glob("*.png"):
        old.unlink()

    print(f"\n  [face_id] Đăng ký khuôn mặt cho '{user_id}'.")
    print(f"  → Nhìn thẳng vào camera. Đang chụp {n_samples} ảnh...\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [face_id] Không mở được camera.")
        return False

    collected = 0
    attempts  = 0
    max_tries = n_samples * 10  # tránh vòng lặp vô tận

    try:
        while collected < n_samples and attempts < max_tries:
            ret, frame = cap.read()
            attempts += 1
            if not ret:
                continue

            face = _detect_face(frame)
            if face is None:
                time.sleep(0.1)
                continue

            img_path = samples_dir / f"{collected}.png"
            cv2.imwrite(str(img_path), face)
            collected += 1
            print(f"  [face_id]  {collected}/{n_samples}", end="\r", flush=True)
            time.sleep(0.15)   # giãn cách giữa các ảnh để đa dạng góc nhìn
    finally:
        cap.release()

    print()
    if collected < n_samples:
        print(f"  [face_id] Chỉ chụp được {collected}/{n_samples} mẫu — "
              f"hãy đảm bảo ánh sáng đủ và nhìn thẳng camera.")
        if collected == 0:
            return False

    # Retrain ngay sau khi enroll
    _retrain()
    print(f"  [face_id] ✅ Đã đăng ký {collected} mẫu cho '{user_id}'.")
    return True


def recognize(threshold: float | None = None) -> tuple[str | None, float]:
    """
    Chụp 1 frame từ camera và nhận dạng người dùng.

    Trả về:
        (user_id: str, confidence: float)  — nếu nhận ra (confidence < threshold)
        (None, confidence)                 — nếu là người lạ / chưa đăng ký
        (None, -1.0)                       — nếu không có camera / không thấy mặt
                                             hoặc chưa có ai đăng ký
    """
    if threshold is None:
        threshold = cfg.FACE_ID_THRESHOLD

    model_file = _model_path()
    if not model_file.exists():
        return None, -1.0   # Chưa có ai đăng ký

    label_map = _load_label_map()
    if not label_map:
        return None, -1.0

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_file))

    # Chụp frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [face_id] Không mở được camera.")
        return None, -1.0

    frame = None
    for _ in range(10):   # thử tối đa 10 frame để tránh frame đầu tối
        ret, f = cap.read()
        if ret:
            frame = f
            break
        time.sleep(0.05)
    cap.release()

    if frame is None:
        return None, -1.0

    face = _detect_face(frame)
    if face is None:
        print("  [face_id] Không phát hiện khuôn mặt trong frame.")
        return None, -1.0

    int_label, confidence = recognizer.predict(face)
    user_id = label_map.get(str(int_label))

    if user_id is None or confidence >= threshold:
        return None, float(confidence)

    return user_id, float(confidence)


def is_enrolled(user_id: str) -> bool:
    """Kiểm tra xem user đã đăng ký khuôn mặt chưa."""
    samples_dir = _samples_dir(user_id)
    return any(samples_dir.glob("*.png"))
