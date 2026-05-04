import librosa
import numpy as np
import os
import cv2  # Thư viện dùng để RESIZE ảnh

# --- CẤU HÌNH ---
INPUT_FOLDER = r"D:\Aura clam\soft\dataset"       # Folder chứa file .wav TESS gốc
OUTPUT_FOLDER = r"D:\Aura clam\soft\kq_hybrid_fix" # Folder chứa kết quả MỚI (đã resize)

# Cấu hình BẮT BUỘC cho Model Hybrid
IMG_SIZE = 64   # Phải resize về số này
N_FRAMES = 10   # Phải cắt thành 10 khúc

def process_and_resize(file_path):
    # 1. Đọc file âm thanh
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=3.0)
    except: return None
    
    # Pad cho đủ 3 giây
    req_len = int(22050 * 3.0)
    if len(y) < req_len: y = np.pad(y, (0, req_len - len(y)))
    else: y = y[:req_len]

    # 2. Cắt thành 10 đoạn
    chunk_len = len(y) // N_FRAMES
    sequence = []
    
    for i in range(N_FRAMES):
        chunk = y[i*chunk_len : (i+1)*chunk_len]
        
        # Tạo Spectrogram
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # --- ĐÂY LÀ LỆNH BẠN ĐANG THIẾU ---
        # Resize về đúng 64x64 pixel
        mel_resized = cv2.resize(mel_db, (IMG_SIZE, IMG_SIZE))
        
        # Chuẩn hóa về 0-1
        mel_norm = (mel_resized - mel_resized.min()) / (mel_resized.max() - mel_resized.min())
        
        sequence.append(mel_norm)

    # Kết quả ra: (10, 64, 64, 1) -> ĐÚNG CHUẨN
    return np.expand_dims(np.array(sequence), axis=-1)

# --- CHẠY VÒNG LẶP ---
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
print("--- ĐANG RESIZE VÀ CẮT DỮ LIỆU ---")

for root, dirs, files in os.walk(INPUT_FOLDER):
    for file in files:
        if file.endswith(".wav"):
            label = os.path.basename(root)
            save_dir = os.path.join(OUTPUT_FOLDER, label)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            
            data = process_and_resize(os.path.join(root, file))
            if data is not None:
                np.save(os.path.join(save_dir, file.replace('.wav','.npy')), data)

print(f"XONG! Dữ liệu đã resize nằm ở: {OUTPUT_FOLDER}")
print("Hãy lấy folder này đi train lại ngay!")