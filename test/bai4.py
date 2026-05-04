import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Dropout

# 1. Khởi tạo mô hình
model = Sequential()

# --- PHẦN 1: CNN (ĐÔI MẮT) ---
# Chúng ta dùng 'TimeDistributed' để bọc CNN lại.
# Nó có nghĩa là: "Hãy áp dụng bộ CNN này cho từng khung hình trong chuỗi 10 khung hình kia".

# Input shape: (10 khung hình, 64 cao, 64 rộng, 1 màu)
model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu'), input_shape=(10, 64, 64, 1)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))

model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))

model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))

# Duỗi từng khung hình ra thành vector đặc trưng
model.add(TimeDistributed(Flatten()))

# --- PHẦN 2: LSTM (TRÍ NHỚ) ---
# Lúc này, CNN đã biến 10 bức ảnh thành 10 vector thông tin.
# LSTM sẽ đọc 10 vector này.

model.add(LSTM(64, return_sequences=False))
# 64: Dung lượng trí nhớ.
# return_sequences=False: Đọc hết 10 tấm rồi mới phán 1 lần cuối cùng.

# --- PHẦN 3: SVM GIẢ LẬP (QUYẾT ĐỊNH) ---
# Lớp Dense này đóng vai trò phân loại (Classifier)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3)) # Quên bớt 30% để tránh học vẹt (Overfitting)

# Đầu ra: 4 cảm xúc
model.add(Dense(4, activation='softmax'))

# --- XEM KẾT QUẢ ---
print("=== CẤU TRÚC MÔ HÌNH HYBRID (TRÙM CUỐI) ===")
model.summary()