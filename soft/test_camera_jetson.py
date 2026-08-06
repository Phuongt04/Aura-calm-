from __future__ import annotations

from collections import deque
import time

import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = "/mnt/mone_usb/MONE/model/vision_emotion_model_tf210.h5"
classes = ["angry", "happy", "normal", "sad"]
IMG_SIZE = 224
CAMERA_INDEX = 0
HEADLESS = False
INFERENCE_INTERVAL = 0.25
EXPECTED_INPUT_SHAPE = (None, 224, 224, 3)
EXPECTED_OUTPUT_SHAPE = (None, 4)


def validate_model(model) -> None:
    print("TensorFlow version:", tf.__version__)
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
    print("Model params:", model.count_params())
    print("Classes:", classes)

    if tuple(model.input_shape) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError("Invalid input shape: {}. Expected: {}".format(model.input_shape, EXPECTED_INPUT_SHAPE))
    if tuple(model.output_shape) != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError("Invalid output shape: {}. Expected: {}".format(model.output_shape, EXPECTED_OUTPUT_SHAPE))
    if len(classes) != model.output_shape[-1]:
        raise RuntimeError("Class count {} does not match model output {}".format(len(classes), model.output_shape[-1]))


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Could not load Haar Cascade: {}".format(cascade_path))
    return face_cascade


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    backend = "CAP_V4L2"
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)
        backend = "default"
    return cap, backend


def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_array = face_resized.astype("float32") / 255.0
    return np.expand_dims(face_array, axis=0)


def predict_probs(model, face_bgr):
    x = preprocess_face(face_bgr)
    start = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    inference_ms = (time.perf_counter() - start) * 1000.0

    if probs.shape != (len(classes),):
        raise RuntimeError("Invalid prediction shape: {}".format(probs.shape))
    if not np.all(np.isfinite(probs)):
        raise RuntimeError("Prediction contains NaN or Inf")

    return probs.astype("float32"), inference_ms


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
    load_start = time.perf_counter()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model_load_ms = (time.perf_counter() - load_start) * 1000.0
    print("Model load time: {:.1f} ms".format(model_load_ms))
    validate_model(model)

    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("Model warm-up done.")

    face_cascade = load_face_detector()
    cap, backend = open_camera()
    print("Camera backend:", backend)
    if not cap.isOpened():
        print("Could not open camera index {}.".format(CAMERA_INDEX))
        return 1

    window_name = "Jetson Facial Emotion Recognition"
    if not HEADLESS:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    probability_window = deque(maxlen=10)
    printed_size = False
    prev_time = time.perf_counter()
    last_inference_time = 0.0
    last_inference_ms = 0.0
    last_label = ""
    frame_count = 0

    print("Camera started. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Empty frame or camera read failed.")
                continue

            frame_count += 1
            if not printed_size:
                print("Frame size:", frame.shape)
                printed_size = True

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            face_box = largest_face(faces)
            now = time.perf_counter()

            if face_box is not None and now - last_inference_time >= INFERENCE_INTERVAL:
                x, y, w, h = face_box
                margin = 10
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                face = frame[y1:y2, x1:x2]

                if face.size != 0:
                    probs, last_inference_ms = predict_probs(model, face)
                    last_inference_time = now
                    probability_window.append(probs)
                    mean_probs = np.mean(np.stack(probability_window, axis=0), axis=0)
                    final_idx = int(np.argmax(mean_probs))
                    final_emotion = classes[final_idx]
                    final_confidence = float(mean_probs[final_idx])
                    last_label = "{}: {:.2f}%".format(final_emotion, final_confidence * 100.0)

            if face_box is not None and last_label:
                x, y, w, h = face_box
                x1 = max(0, x - 10)
                y1 = max(0, y - 10)
                x2 = min(frame.shape[1], x + w + 10)
                y2 = min(frame.shape[0], y + h + 10)
                draw_label(frame, (x1, y1, x2, y2), last_label)

            current_time = time.perf_counter()
            dt = current_time - prev_time
            prev_time = current_time
            fps = 1.0 / dt if dt > 0 else 0.0

            if HEADLESS:
                if frame_count % 10 == 0:
                    print(
                        "frames={}, label={}, FPS={:.2f}, inference={:.1f} ms".format(
                            frame_count, last_label or "no-face", fps, last_inference_ms
                        )
                    )
            else:
                cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "Infer: {:.1f} ms".format(last_inference_ms), (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "Frames: {}".format(frame_count), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
