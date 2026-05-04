import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- PHẦN 1: GHI ÂM ---
fs = 44100  # Tần số lấy mẫu (chuẩn CD)
seconds = 3  # Thời gian ghi âm (giây)

print("--- BẮT ĐẦU GHI ÂM (3 giây) ---")
print("Hãy nói một câu cảm xúc (Ví dụ: 'Tôi đang rất vui!')")

# Lệnh ghi âm
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Chờ ghi âm xong
print("--- GHI ÂM XONG! ---")

# Lưu thành file
filename = 'giong_noi_cua_toi.wav'
# Chuyển đổi định dạng để lưu file (float -> int16)
wav.write(filename, fs, (myrecording * 32767).astype(np.int16))

# --- PHẦN 2: XỬ LÝ & HIỂN THỊ (QUAN TRỌNG CHO AI) ---
# Load file âm thanh vừa ghi bằng thư viện Librosa
y, sr = librosa.load(filename)

plt.figure(figsize=(12, 8))

# 1. Vẽ Sóng âm (Waveform) - Cái tai nghe thấy
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('1. Sóng âm (Waveform) - Biểu hiện độ to nhỏ theo thời gian')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')

# 2. Vẽ Spectrogram - Cái AI (CNN) sẽ nhìn thấy
# Chuyển đổi sang Mel-Spectrogram (Ảnh nhiệt của âm thanh)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.subplot(2, 1, 2)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('2. Mel-Spectrogram - Đầu vào cho mô hình CNN')

plt.tight_layout()
plt.show()