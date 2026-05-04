import sounddevice as sd
import numpy as np
import librosa
import cv2
import pickle
import json
import os
import warnings
import h5py
from pathlib import Path
from tensorflow.keras.models import load_model, model_from_json

# Tắt các cảnh báo không cần thiết của thư viện để màn hình console sạch sẽ
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. CẤU HÌNH HỆ THỐNG ---
DURATION = 3.0
SAMPLE_RATE = 22050
N_FRAMES = 10
IMG_SIZE = 64
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / "best_model_fold_1.h5"
LABEL_PATH = BASE_DIR / "D:\\Aura clam\\label_mapping.pkl"


def _normalize_keras_config(value):
    if isinstance(value, dict):
        if value.get("class_name") == "DTypePolicy":
            return value.get("config", {}).get("name", "float32")
        return {key: _normalize_keras_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_keras_config(item) for item in value]
    return value


def load_legacy_h5_model(model_path):
    with h5py.File(model_path, "r") as file:
        model_config = file.attrs.get("model_config")

    if model_config is None:
        raise ValueError(f"Model khong co model_config: {model_path}")

    if isinstance(model_config, bytes):
        model_config = model_config.decode("utf-8")

    config_data = _normalize_keras_config(json.loads(model_config))

    for layer in config_data.get("config", {}).get("layers", []):
        layer_config = layer.get("config", {})
        if layer.get("class_name") == "InputLayer" and "batch_shape" in layer_config:
            layer_config["batch_input_shape"] = layer_config.pop("batch_shape")

    model = model_from_json(json.dumps(config_data))
    model.load_weights(model_path)
    return model


def safe_load_model(model_path):
    try:
        return load_model(model_path)
    except ValueError as error:
        if "batch_shape" not in str(error):
            raise
        print("Model cu khong tuong thich truc tiep. Dang thu che do nap tuong thich...")
        return load_legacy_h5_model(model_path)


def load_resources():
    print("Dang tai mo hinh va nhan...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay file nhan: {LABEL_PATH}")

    model = safe_load_model(MODEL_PATH)
    with open(LABEL_PATH, "rb") as file:
        classes = pickle.load(file)

    classes = list(classes)
    output_shape = model.output_shape
    num_outputs = output_shape[-1] if isinstance(output_shape, tuple) else output_shape[0][-1]
    if len(classes) != num_outputs:
        raise ValueError(
            "So luong nhan khong khop voi output cua model: "
            f"{len(classes)} != {num_outputs}"
        )

    print(f"Da tai model thanh cong. Cac lop cam xuc: {classes}")
    return model, classes

# --- 2. HÀM THU ÂM ---
def record_audio():
    print(f"\nBat dau thu am ({DURATION} giay)... Hay noi gi do!")
    try:
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as error:
        raise RuntimeError(f"Khong the thu am tu microphone: {error}") from error

    print("Da thu am xong. Dang phan tich...")
    return audio.flatten()

# --- 3. TIỀN XỬ LÝ ÂM THANH (Dựa trên code fix_data của bạn) ---
def preprocess_audio(y):
    req_len = int(SAMPLE_RATE * DURATION)
    if len(y) < req_len:
        y = np.pad(y, (0, req_len - len(y)))
    else:
        y = y[:req_len]

    chunk_len = len(y) // N_FRAMES
    sequence = []

    for i in range(N_FRAMES):
        chunk = y[i * chunk_len : (i + 1) * chunk_len]
        mel = librosa.feature.melspectrogram(y=chunk, sr=SAMPLE_RATE, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_resized = cv2.resize(mel_db, (IMG_SIZE, IMG_SIZE))

        min_val = mel_resized.min()
        max_val = mel_resized.max()

        denominator = max_val - min_val
        if denominator == 0:
            denominator = 1

        mel_norm = (mel_resized - min_val) / denominator
        sequence.append(mel_norm)

    processed_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=-1)
    return processed_data

# --- 4. HÀM CHẠY CHÍNH ---
def predict_emotion(model, classes):
    while True:
        y_audio = record_audio()
        processed_input = preprocess_audio(y_audio)
        model_input = np.expand_dims(processed_input, axis=0)

        predictions = model.predict(model_input, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)

        emotion_result = classes[predicted_index]
        print("\n" + "=" * 40)
        print(f"CAM XUC DU DOAN: {str(emotion_result).upper()}")
        print(f"DO TU TIN: {confidence:.2f}%")
        print("=" * 40)

        cont = input("\nBan co muon thu lai khong? (y/n): ").strip().lower()
        if cont not in {"y", "yes"}:
            print("Tam biet.")
            break


def main():
    try:
        model, classes = load_resources()
        predict_emotion(model, classes)
    except Exception as error:
        print(f"Loi: {error}")
        raise SystemExit(1) from error

# Kích hoạt chương trình
if __name__ == "__main__":
    main()