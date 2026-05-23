import numpy as np
import os
import pickle
import json
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, LSTM, TimeDistributed,
                                     BatchNormalization, Bidirectional)
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Seed để reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# --- 1. CẤU HÌNH ---
DATA_FOLDER = "D:\\Aura clam\\soft\\dataset"
IMG_SIZE = 64
N_FRAMES = 10
BATCH_SIZE = 32
MAX_EPOCHS = 100   # EarlyStopping sẽ tự dừng sớm khi không còn cải thiện

# --- 2. TẢI DỮ LIỆU ---
def load_all_data():
    X, y = [], []
    
    if not os.path.isdir(DATA_FOLDER):
        raise FileNotFoundError(f"Folder không tồn tại: {DATA_FOLDER}")
    
    folders = [f for f in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, f))]
    print(f"Đang tải dữ liệu từ {len(folders)} lớp...")
    
    if not folders:
        raise ValueError(f"Không tìm thấy thư mục con nào trong {DATA_FOLDER}")

    for folder in folders:
        folder_path = os.path.join(DATA_FOLDER, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        print(f"  {folder}: {len(files)} mẫu")
        
        for file in files:
            try:
                data = np.load(os.path.join(folder_path, file))
                if data.shape == (N_FRAMES, IMG_SIZE, IMG_SIZE, 1):
                    X.append(data)
                    y.append(folder)
            except Exception as e:
                print(f"    ⚠️  Lỗi tải {file}: {e}")

    X = np.array(X, dtype='float32')
    if len(X) == 0:
        raise ValueError("Không tải được dữ liệu nào!")
    
    if np.max(X) > 1.0:
        X = X / 255.0
    
    print(f"✅ Tải được {len(X)} mẫu\n")
    return X, np.array(y)

X, y_raw = load_all_data()

# Mã hóa nhãn
le = LabelEncoder()
y_int = le.fit_transform(y_raw)
y_final = to_categorical(y_int)

with open("label_mapping_final.pkl", "wb") as f:
    pickle.dump(le.classes_, f)

# Tách validation set 15% để theo dõi thực sự trong quá trình train
X_train, X_val, y_train, y_val = train_test_split(
    X, y_final, test_size=0.15, random_state=42, stratify=y_int
)
print(f"Train: {len(X_train)} mẫu | Val: {len(X_val)} mẫu")

# --- 3. XÂY DỰNG MÔ HÌNH CẢI TIẾN ---
print("\n--- ĐANG KHỞI TẠO MÔ HÌNH CẢI TIẾN ---")
model = Sequential([
    # Khối CNN 1: tăng filter lên 32, thêm BatchNorm để ổn định training
    TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'),
                    input_shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, 1)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),

    # Khối CNN 2: tăng filter lên 64 để học đặc trưng phức tạp hơn
    TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),

    # Khối CNN 3 (mới): học đặc trưng bậc cao hơn
    TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),

    TimeDistributed(Flatten()),

    # Bidirectional LSTM: học ngữ cảnh theo cả 2 chiều thời gian
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.4),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- 4. CALLBACKS ---
callbacks = [
    # Dừng sớm nếu val_accuracy không tăng sau 15 epoch liên tiếp
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    # Lưu model tốt nhất theo val_accuracy
    ModelCheckpoint("best_model_improved.h5", monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    # Giảm learning rate x0.5 nếu val_loss không giảm sau 7 epoch
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
]

# --- 5. HUẤN LUYỆN ---
print(f"\nBắt đầu huấn luyện {len(X_train)} mẫu (val: {len(X_val)}) | tối đa {MAX_EPOCHS} epochs...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# --- 6. LƯU MÔ HÌNH CUỐI ---
model.save("final_production_model.h5")
best_val_acc = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

print(f"\n Xong! Val accuracy tốt nhất: {best_val_acc:.4f} (epoch {best_epoch})")
print("Đã lưu model tốt nhất vào 'best_model_improved.h5' và model cuối vào 'final_production_model.h5'")

# Lưu training history
history_dict = {
    'train_loss': [float(x) for x in history.history['loss']],
    'train_accuracy': [float(x) for x in history.history['accuracy']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'best_val_accuracy': float(best_val_acc),
    'best_epoch': best_epoch,
}
with open("training_history.json", "w") as f:
    json.dump(history_dict, f, indent=2)
print(f"Đã lưu lịch sử training vào 'training_history.json'")