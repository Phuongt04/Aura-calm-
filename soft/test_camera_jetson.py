from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


CLASSES = ["angry", "happy", "normal", "sad"]
IMG_SIZE = 224
VOTE_WINDOW = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson-compatible camera test for facial emotion recognition.")
    parser.add_argument("--model", default="vision_emotion_model_compatible.h5", help="Path to Keras H5 model.")
    parser.add_argument("--camera-index", type=int, default=0, help="V4L2 camera index.")
    parser.add_argument("--headless", action="store_true", help="Disable cv2.imshow() for headless runs.")
    parser.add_argument("--width", type=int, default=0, help="Optional camera frame width.")
    parser.add_argument("--height", type=int, default=0, help="Optional camera frame height.")
    return parser.parse_args()


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError("Model file not found: {}".format(model_path))
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print("Loaded model:", model_path)
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Classes:", CLASSES)
    return model


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Could not load Haar Cascade: {}".format(cascade_path))
    return face_cascade


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_array = face_resized.astype("float32") / 255.0
    return np.expand_dims(face_array, axis=0)


def predict_probs(model, face_bgr: np.ndarray):
    x = preprocess_face(face_bgr)
    start = time.perf_counter()
    probs = model.predict(x, verbose=0)[0].astype("float32")
    inference_ms = (time.perf_counter() - start) * 1000.0
    return probs, inference_ms


def largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def draw_label(frame, box, label):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label_height = 35
    text_y = max(label_height, y1)
    cv2.rectangle(frame, (x1, text_y - label_height), (x2, text_y), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 5, text_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


def main() -> int:
    args = parse_args()
    model = load_model(Path(args.model).resolve())
    face_cascade = load_face_detector()

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Could not open camera index {} with CAP_V4L2.".format(args.camera_index))
        return 1

    window_name = "Jetson Facial Emotion Recognition"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prob_window = deque(maxlen=VOTE_WINDOW)
    prev_time = time.perf_counter()
    printed_size = False
    last_inference_ms = 0.0

    print("Camera started. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Empty frame or camera read failed.")
                break

            if not printed_size:
                print("Original camera frame size:", frame.shape)
                printed_size = True

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            face_box = largest_face(faces)

            if face_box is not None:
                x, y, w, h = face_box
                margin = 10
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                face = frame[y1:y2, x1:x2]

                if face.size != 0:
                    probs, last_inference_ms = predict_probs(model, face)
                    prob_window.append(probs)

                    mean_probs = np.mean(np.stack(prob_window, axis=0), axis=0)
                    final_idx = int(np.argmax(mean_probs))
                    final_emotion = CLASSES[final_idx]
                    confidence = float(mean_probs[final_idx])
                    label = "{}: {:.2f}%".format(final_emotion, confidence * 100.0)
                    draw_label(frame, (x1, y1, x2, y2), label)

            current_time = time.perf_counter()
            frame_dt = current_time - prev_time
            prev_time = current_time
            fps = 1.0 / frame_dt if frame_dt > 0 else 0.0

            cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(
                frame,
                "Infer: {:.1f} ms".format(last_inference_ms),
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            if args.headless:
                print("FPS: {:.2f}, inference: {:.1f} ms".format(fps, last_inference_ms))
            else:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
