import cv2
import time
import numpy as np
import tensorflow as tf

from collections import deque, Counter


# ============================================================
# 1. CONFIG
# ============================================================

MODEL_PATH = "vision_emotion_model.h5"

# Thứ tự class phải giống lúc train
classes = ["angry", "happy", "normal", "sad"]

IMG_SIZE = 224

# Camera index máy bạn đang nhận là 0
CAMERA_INDEX = 0


# ============================================================
# 2. LOAD MODEL KERAS
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH)

print("Loaded Keras model successfully!")
print("Model input shape:", model.input_shape)
print("Classes:", classes)


# ============================================================
# 3. FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Không load được Haar Cascade.")
    exit()


# ============================================================
# 4. PREDICT FUNCTION
# ============================================================

def predict_emotion(face_bgr):
    # BGR -> RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Resize riêng khuôn mặt về 224x224 để đưa vào model
    # Lưu ý: chỉ resize ảnh input cho model, KHÔNG resize camera hiển thị
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))

    face_array = face_resized.astype("float32") / 255.0
    face_array = np.expand_dims(face_array, axis=0)

    pred_probs = model.predict(face_array, verbose=0)[0]

    pred_idx = int(np.argmax(pred_probs))
    confidence = float(pred_probs[pred_idx])
    emotion = classes[pred_idx]

    return emotion, confidence


# ============================================================
# 5. OPEN CAMERA - KHÔNG ÉP RESOLUTION
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Không mở được camera.")
    print("Hãy tắt app Camera/Zalo/Zoom/Google Meet rồi chạy lại.")
    exit()

print("Camera started. Press 'q' to quit.")


# ============================================================
# 6. WINDOW SETTING
# ============================================================

window_name = "Realtime Facial Emotion Recognition - Keras"

# WINDOW_AUTOSIZE giúp hiển thị đúng kích thước frame camera gốc
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)


# ============================================================
# 7. SLIDING WINDOW
# ============================================================

emotion_window = deque(maxlen=10)

prev_time = time.time()

printed_size = False


# ============================================================
# 8. REALTIME LOOP
# ============================================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không đọc được frame từ camera.")
        break

    if not printed_size:
        print("Original camera frame size:", frame.shape)
        printed_size = True

    # Lật ảnh như gương soi
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # Chỉ lấy khuôn mặt lớn nhất
    if len(faces) > 0:
        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)

        x, y, w, h = faces[0]

        # Margin nhỏ thôi, tránh box quá to
        margin = 10

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face = frame[y1:y2, x1:x2]

        if face.size != 0:
            emotion, confidence = predict_emotion(face)

            emotion_window.append(emotion)

            final_emotion = Counter(emotion_window).most_common(1)[0][0]

            label = f"{final_emotion}: {confidence * 100:.2f}%"

            # Vẽ bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Nền chữ
            label_height = 35
            text_y = max(label_height, y1)

            cv2.rectangle(
                frame,
                (x1, text_y - label_height),
                (x2, text_y),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1 + 5, text_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                2
            )

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Hiển thị đúng frame gốc, không resize
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 9. RELEASE
# ============================================================

cap.release()
cv2.destroyAllWindows()