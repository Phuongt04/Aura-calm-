import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# --- CẤU HÌNH ---
DATA_FOLDER = "D:\\Aura clam\\soft\\kq_hybrid_final" 
IMG_SIZE = 64
N_FRAMES = 10 
BATCH_SIZE = 32
EPOCHS = 50

# --- 1. HÀM LOAD DỮ LIỆU (CÓ LỌC 4 LOẠI) ---
def load_data_4_emotions():
    print("⏳ Đang tải dữ liệu và LỌC lấy 4 loại: Happy, Sad, Angry, Neutral...")
    X = []
    y = []
    
    # Từ khóa cần tìm
    keywords = {
        'happy': 'Happy',
        'sad': 'Sad',
        'angry': 'Angry',
        'neutral': 'Neutral'
    }

    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.endswith(".npy"):
                folder_name = os.path.basename(root).lower()
                
                # --- LOGIC LỌC ---
                label = None
                for key, val in keywords.items():
                    if key in folder_name:
                        label = val
                        break
                
                # Nếu không phải 4 loại này thì bỏ qua
                if label is None: continue

                path = os.path.join(root, file)
                try:
                    data = np.load(path)
                    # Kiểm tra kích thước
                    if data.shape == (N_FRAMES, IMG_SIZE, IMG_SIZE, 1):
                        X.append(data)
                        y.append(label)
                except: pass
                
    return np.array(X), np.array(y)

# --- CHƯƠNG TRÌNH CHÍNH ---

# A. Chuẩn bị dữ liệu
X, y_raw = load_data_4_emotions()

if len(X) == 0:
    print("❌ LỖI: Không tìm thấy dữ liệu!")
    exit()

print(f"✅ Tổng số mẫu tìm thấy: {len(X)}")
print(f"   Các nhãn: {np.unique(y_raw)}")

# Mã hóa nhãn (Sẽ tự động ra 0, 1, 2, 3)
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
y_final = to_categorical(y_encoded)

# Lưu lại mapping (Quan trọng cho bước đánh giá sau này)
with open("label_mapping_4_emotions.pkl", "wb") as f:
    pickle.dump(le.classes_, f)
print(f"📝 Đã lưu danh sách nhãn: {le.classes_}")

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)

# B. XÂY DỰNG MODEL MỚI (Output layer sẽ tự động là 4)
print("🏗️ Đang xây dựng Model chuyên biệt 4 cảm xúc...")
model = Sequential()

# CNN
model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu'), input_shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, 1)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))

# LSTM
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))

# Output (Tự động là 4 nhờ len(le.classes_))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax')) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# C. TRAIN
print("🚀 BẮT ĐẦU HUẤN LUYỆN...")
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# D. LƯU
model.save("model_4_emotions.h5")
print("\n✅ THÀNH CÔNG! Đã có file 'model_4_emotions.h5'")