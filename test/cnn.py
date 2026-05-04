import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 1. Khởi tạo mô hình
model = Sequential()

# --- BƯỚC 1: QUÉT ẢNH (FEATURE EXTRACTION) ---

# Lớp CNN 1: Dùng 32 cái kính lúp (filters) để soi.
# Kích thước kính lúp là 3x3 pixel.
# input_shape=(64, 64, 3): Giả sử ảnh to 64x64, có 3 màu (RGB).
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))

# Lớp Gộp 1: Giữ lại điểm sáng nhất trong ô 2x2.
# Sau bước này, ảnh sẽ nhỏ đi một nửa (còn 32x32).
model.add(MaxPooling2D(pool_size=(2, 2)))

# Lớp CNN 2: Soi kỹ hơn với 64 cái kính lúp.
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Sau bước này, ảnh còn 16x16.

# --- BƯỚC 2: QUYẾT ĐỊNH (CLASSIFICATION) ---

# Duỗi thẳng tấm ảnh ra thành 1 hàng dọc
model.add(Flatten())

# Đưa vào mạng nơ-ron (giống bài ANN hôm qua)
model.add(Dense(128, activation='relu'))

# Đầu ra: Giả sử phân loại Chó và Mèo (chỉ cần 1 nơ-ron 0 hoặc 1)
model.add(Dense(1, activation='sigmoid'))

# --- XEM KẾT QUẢ ---
# Lệnh này sẽ in ra bảng tóm tắt kiến trúc của con AI
print("CẤU TRÚC MẠNG CNN CỦA BẠN:")
model.summary()