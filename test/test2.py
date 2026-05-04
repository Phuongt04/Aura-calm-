import numpy as np
import librosa
import cv2  # Thư viện OpenCV để chỉnh kích thước ảnh

# CẤU HÌNH
SAMPLE_RATE = 22050
DURATION = 3.0       # Giây
N_FRAMES = 10        # Số lượng khung hình (TimeDistributed yêu cầu)
IMG_SIZE = 64        # Kích thước ảnh (64x64)

def process_audio_for_hybrid(file_path):
    # 1. Đọc file âm thanh
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    
    # Đảm bảo độ dài đúng 3 giây (thêm số 0 nếu thiếu)
    required_length = int(SAMPLE_RATE * DURATION)
    if len(y) < required_length:
        y = np.pad(y, (0, required_length - len(y)))
    else:
        y = y[:required_length]

    # 2. Chia nhỏ thành 10 đoạn (chunks)
    chunk_length = len(y) // N_FRAMES
    chunks = [y[i : i + chunk_length] for i in range(0, len(y), chunk_length)]
    
    # Chỉ lấy đúng 10 đoạn đầu (đề phòng chia dư)
    chunks = chunks[:N_FRAMES] 

    spectrogram_sequence = []

    # 3. Biến từng đoạn thành ảnh Spectrogram
    for chunk in chunks:
        # Tạo Mel-Spectrogram
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Chỉnh kích thước về 64x64 (Resize)
        # Vì ảnh phổ có kích thước lung tung, ta phải ép nó về 64x64
        mel_resized = cv2.resize(mel_db, (IMG_SIZE, IMG_SIZE))
        
        # Chuẩn hóa giá trị về khoảng [0, 1] cho AI dễ học
        # (Bình thường nó là số âm -80dB, phải đổi về 0-1)
        mel_norm = (mel_resized - mel_resized.min()) / (mel_resized.max() - mel_resized.min())

        spectrogram_sequence.append(mel_norm)

    # 4. Đóng gói lại thành mảng Numpy
    # Kết quả sẽ có dạng (10, 64, 64)
    X_data = np.array(spectrogram_sequence)
    
    # Thêm 1 chiều màu (Channel) ở cuối cho đúng chuẩn Keras (10, 64, 64, 1)
    X_data = np.expand_dims(X_data, axis=-1)
    
    return X_data

# --- CHẠY THỬ ---
# Bạn hãy thay đường dẫn này bằng 1 file âm thanh thật của bạn
file_test = "dataset_raw/BUON/BUON_1.wav" 

try:
    ket_qua = process_audio_for_hybrid(file_test)
    print("Xử lý thành công!")
    print(f"Kích thước dữ liệu đầu ra: {ket_qua.shape}")
    print("Đúng chuẩn (10, 64, 64, 1) chưa? Nếu đúng là OK!")
except Exception as e:
    print("Chưa có file âm thanh mẫu. Bạn hãy thu âm 1 file rồi sửa đường dẫn nhé.")
    print(f"Lỗi: {e}")