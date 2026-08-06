"""
face.py
=======
Face emotion pipeline: capture camera frames, detect the largest face,
preprocess it, then run the face emotion model.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from . import config as cfg

_face_cascade: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            raise RuntimeError(f"Could not load Haar Cascade: {cascade_path}")
    return _face_cascade


def _open_camera(camera_index: int):
    if getattr(cfg, "DEPLOY_TARGET", "") == "jetson":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)


def capture_frames(
    camera_index: int = 0,
    frame_count: int | None = None,
    warmup_frames: int | None = None,
) -> list[np.ndarray]:
    """Capture several frames while keeping the camera open."""
    if frame_count is None:
        frame_count = max(1, cfg.FACE_CAPTURE_FRAMES)
    if warmup_frames is None:
        warmup_frames = max(0, cfg.FACE_CAPTURE_WARMUP_FRAMES)

    cap = _open_camera(camera_index)
    if not cap.isOpened():
        print(f"  [face] Khong the mo camera (index={camera_index}).")
        return []

    frames: list[np.ndarray] = []
    try:
        for _ in range(warmup_frames):
            cap.read()
            time.sleep(0.03)
        for _ in range(frame_count):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
            time.sleep(0.03)
    finally:
        cap.release()
    return frames


def capture_frame(camera_index: int = 0):
    """Backward-compatible helper for callers that only need one frame."""
    frames = capture_frames(camera_index, frame_count=1)
    return frames[0] if frames else None


def preprocess(frame: np.ndarray):
    """
    Detect the largest face, crop it with a small margin, resize, normalize.
    Returns an RGB float32 image or None when no face is found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(
        gray,
        scaleFactor=cfg.FACE_DETECT_SCALE_FACTOR,
        minNeighbors=cfg.FACE_DETECT_MIN_NEIGHBORS,
        minSize=(cfg.FACE_DETECT_MIN_SIZE, cfg.FACE_DETECT_MIN_SIZE),
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    margin = 10
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return None

    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    face_resize = cv2.resize(face_rgb, (cfg.FACE_IMG_SIZE, cfg.FACE_IMG_SIZE))
    return face_resize.astype("float32") / 255.0


def _map_label(raw_label: str) -> str:
    return cfg.FACE_LABEL_MAP.get(raw_label.lower(), "binhthuong")


def detect(model, face_raw_classes: list, camera_index: int | None = None) -> tuple:
    """
    Capture frames, find a face, then predict emotion.

    Returns:
        (normalized_label, confidence_percent) on success
        (None, None) when camera/face detection fails
    """
    if camera_index is None:
        camera_index = cfg.FACE_CAMERA_INDEX

    frames = capture_frames(camera_index)
    if not frames:
        return None, None

    processed = None
    for frame in frames:
        processed = preprocess(frame)
        if processed is not None:
            break

    if processed is None:
        print(f"  [face] Khong phat hien khuon mat trong {len(frames)} frame.")
        return None, None

    inp = np.expand_dims(processed, axis=0)
    preds = model.predict(inp, verbose=0)[0]
    if preds.shape[0] != len(face_raw_classes):
        raise ValueError(
            f"Face model output has {preds.shape[0]} classes, "
            f"but FACE_RAW_CLASSES has {len(face_raw_classes)}"
        )
    if not np.all(np.isfinite(preds)):
        raise ValueError("Face model prediction contains NaN or Inf")

    idx = int(np.argmax(preds))
    raw_label = face_raw_classes[idx]
    mapped = _map_label(raw_label)
    conf = float(preds[idx]) * 100

    print(f"  [face] {raw_label} -> {mapped} ({conf:.1f}%)")
    return mapped, conf
