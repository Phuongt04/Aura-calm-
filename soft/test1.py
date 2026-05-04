import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import time

# --- CẤU HÌNH HỆ THỐNG ---
FOLDER_NAME = "dataset_cam_xuc"  # Thư mục chứa dữ liệu
FS = 44100                       # Tần số lấy mẫu (44.1kHz là chuẩn tốt nhất)
SECONDS = 3                      # Độ dài mỗi file ghi âm (giây)

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)
    print(f"[SYSTEM] Đã tạo thư mục mới: {FOLDER_NAME}")
else:
    print(f"[SYSTEM] Dữ liệu sẽ được lưu vào: {FOLDER_NAME}")

print("-" * 50)
print(f"CHƯƠNG TRÌNH THU THẬP DỮ LIỆU GIỌNG NÓI")
print(f"Cấu hình: {SECONDS} giây/file | Tần số: {FS}Hz")
print("-" * 50)

file_count = 0

while True:
    print("\n" + "="*20 + f" FILE THỨ {file_count + 1} " + "="*20)
    
    # 1. Nhập nhãn để bắt đầu ghi âm
    # Nhập 'vui' -> file sẽ tên là vui_ngaygio.wav
    label = input(">> Nhập NHÃN cảm xúc (ví dụ: vui, buon, gian) hoặc 'q' để Thoát: ").strip()
    
    if label.lower() == 'q':
        print("\n[SYSTEM] Đã thoát chương trình. Tạm biệt!")
        break
    
    if label == "":
        label = "khong_xac_dinh" # Xử lý trường hợp lỡ tay bấm Enter mà không nhập gì

    # 2. Chuẩn bị đếm ngược để người dùng chuẩn bị
    print("Chuẩn bị...", end=" ")
    for i in range(3, 0, -1):
        print(i, end="... ", flush=True)
        time.sleep(0.5)
    
    # 3. Tiến hành ghi âm
    print("\n🔴 ĐANG GHI ÂM... NÓI ĐI!")
    myrecording = sd.rec(int(SECONDS * FS), samplerate=FS, channels=1)
    sd.wait()  # Chờ ghi âm xong
    print("🟢 ĐÃ GHI XONG.")

    # 4. Tạo tên file định dạng: nhan_ngaythang_gio.wav
    # Ví dụ: vui_20260111_193005.wav -> Cực kỳ dễ quản lý khi train AI
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{label}_{timestamp}"
    
    wav_path = os.path.join(FOLDER_NAME, f"{filename_base}.wav")
    img_path = os.path.join(FOLDER_NAME, f"{filename_base}.png")

    # 5. Lưu file âm thanh (.wav)
    # Chuyển đổi sang int16 để giảm dung lượng và tương thích tốt
    wav.write(wav_path, FS, (myrecording * 32767).astype(np.int16))
    print(f"   -> Đã lưu Audio: {filename_base}.wav")

    # 6. Xử lý & Lưu ảnh Mel-Spectrogram (Chạy ngầm)
    try:
        # Load lại file vừa ghi để vẽ biểu đồ
        y, sr = librosa.load(wav_path)
        
        plt.figure(figsize=(10, 4))
        
        # Tạo Mel-Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Vẽ
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram: {label}')
        plt.tight_layout()
        
        # Lưu ảnh và đóng plt ngay lập tức để giải phóng RAM
        plt.savefig(img_path)
        plt.close() 
        print(f"   -> Đã lưu Ảnh:   {filename_base}.png")
        
    except Exception as e:
        print(f"   [CẢNH BÁO] Không thể lưu ảnh: {e}")

    file_count += 1