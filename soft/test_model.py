import numpy as np
import tensorflow as tf
import librosa
import cv2
import os

# --- CẤU HÌNH ---
MODEL_PATH = "model_hybrid_cam_xuc.h5"  # File model bạn vừa train xong
LABEL_MAP = "label_mapping.pkl"         # File danh sách nhãn (Vui, Buồn...)
FILE_TEST = r"D:\Aura clam\soft\dataset\TESS\OAF_disgust\OAF_back_disgust.wav" # Chọn đại 1 file wav bất kỳ để test

# Load lại từ điển nhãn
import pickle
with open(LABEL_MAP, "rb") as f:
    class_names = pickle.load(f)

# Hàm xử lý 1 file âm thanh (Y hệt lúc train)
def prepare_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=3.0)
    
    # Pad
    req_len = int(22050 * 3.0)
    if len(y) < req_len: y = np.pad(y, (0, req_len - len(y)))
    else: y = y[:req_len]

    # Cắt 10 khúc
    chunk_len = len(y) // 10
    sequence = []
    for i in range(10):
        chunk = y[i*chunk_len : (i+1)*chunk_len]
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_img = cv2.resize(mel_db, (64, 64))
        
        # Fix lỗi NaN (Quan trọng!)
        min_val = mel_img.min()
        max_val = mel_img.max()
        if (max_val - min_val) == 0: denominator = 1
        else: denominator = max_val - min_val
        
        mel_norm = (mel_img - min_val) / denominator
        sequence.append(mel_norm)

    # Thêm chiều batch: (1, 10, 64, 64, 1)
    return np.expand_dims(np.array(sequence), axis=[0, -1])

# --- CHẠY DỰ ĐOÁN ---
print("Đang nạp Model...")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Đang nghe file: {os.path.basename(FILE_TEST)}")
input_data = prepare_audio(FILE_TEST)

# Model đoán
prediction = model.predict(input_data)
# Lấy nhãn có tỉ lệ cao nhất
max_index = np.argmax(prediction)
predicted_label = class_names[max_index]
confidence = prediction[0][max_index] * 100

print(predicted_label
      )
print("="*30)
print(f" ROBOT NGHE THẤY: {predicted_label.upper()}")
print(f" ĐỘ TIN CẬY: {confidence:.2f}%")
print("="*30)