import librosa
import numpy as np
import os
import cv2

# --- CẤU HÌNH ---
# 1. Folder chứa file TESS gốc (.wav)
INPUT_FOLDER = r"D:\Aura clam\data" 

# 2. Folder mới chứa dữ liệu SẠCH (Không bị NaN)
OUTPUT_FOLDER = r"D:\Aura clam\soft\hybrid_final" 

DURATION = 3.0
SAMPLE_RATE = 22050
N_FRAMES = 10
IMG_SIZE = 64

def process_safe_hybrid(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    except: return None
    
    # Pad
    req_len = int(SAMPLE_RATE * DURATION)
    if len(y) < req_len: y = np.pad(y, (0, req_len - len(y)))
    else: y = y[:req_len]

    chunk_len = len(y) // N_FRAMES
    sequence = []
    
    for i in range(N_FRAMES):
        chunk = y[i*chunk_len : (i+1)*chunk_len]
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_resized = cv2.resize(mel_db, (IMG_SIZE, IMG_SIZE))
        
        # --- ĐOẠN SỬA LỖI NaN (QUAN TRỌNG NHẤT) ---
        min_val = mel_resized.min()
        max_val = mel_resized.max()
        
        # Nếu Max bằng Min (ảnh một màu), gán mẫu số = 1 để tránh chia cho 0
        denominator = max_val - min_val
        if denominator == 0:
            denominator = 1 
            
        mel_norm = (mel_resized - min_val) / denominator
        # ------------------------------------------
        
        sequence.append(mel_norm)

    return np.expand_dims(np.array(sequence), axis=-1)

# --- CHẠY ---
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
print("--- ĐANG XỬ LÝ DỮ LIỆU (FIX LỖI NaN) ---")

count = 0
for root, dirs, files in os.walk(INPUT_FOLDER):
    for file in files:
        if file.endswith(".wav"):
            label = os.path.basename(root)
            save_dir = os.path.join(OUTPUT_FOLDER, label)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            
            data = process_safe_hybrid(os.path.join(root, file))
            if data is not None:
                # Kiểm tra lần cuối xem có NaN không
                if np.isnan(data).any():
                    print(f"⚠️ Vẫn còn NaN ở file: {file} -> Bỏ qua")
                else:
                    np.save(os.path.join(save_dir, file.replace('.wav','.npy')), data)
                    count += 1
                    if count % 200 == 0: print(f"Đã xong {count} file...")

print(f"✅ XONG! Dữ liệu sạch nằm tại: {OUTPUT_FOLDER}")