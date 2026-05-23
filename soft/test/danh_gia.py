import numpy as np
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- CẤU HÌNH ---
# Trỏ đúng vào thư mục dữ liệu sạch (Hybrid) của bạn
DATA_FOLDER = "D:\\Aura clam\\soft\\kq_hybrid_fix" 
MODEL_PATH = "model_hybrid_cam_xuc1.h5"
LABEL_MAP = "label_mapping.pkl"
IMG_SIZE = 64
N_FRAMES = 10 

# --- 1. LOAD LẠI DỮ LIỆU (Giống hệt lúc train) ---
def load_data():
    print("Đang tải dữ liệu để chấm điểm...")
    X = []
    y = []
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.endswith(".npy"):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                try:
                    data = np.load(path)
                    if data.shape == (N_FRAMES, IMG_SIZE, IMG_SIZE, 1):
                        X.append(data)
                        y.append(label)
                except: pass
    return np.array(X), np.array(y)

# --- CHƯƠNG TRÌNH CHÍNH ---
# 1. Chuẩn bị dữ liệu
X, y_raw = load_data()

# Load danh sách nhãn cũ để đảm bảo thứ tự đúng (Vui=0, Buồn=1...)
with open(LABEL_MAP, "rb") as f:
    class_names = pickle.load(f)

# Mã hóa nhãn theo đúng thứ tự cũ
le = LabelEncoder()
le.classes_ = class_names # Ép dùng bộ nhãn cũ
y_encoded = le.transform(y_raw)
y_final = to_categorical(y_encoded)

# Chia lại tập Test (phải dùng đúng random_state=42 như lúc train để ra đúng tập test cũ)
X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)

print(f"Số lượng mẫu dùng để test: {len(X_test)}")

# 2. Load Model & Dự đoán
print("Đang nạp Model & Chạy dự đoán...")
model = tf.keras.models.load_model(MODEL_PATH)
y_pred_probs = model.predict(X_test)

# Chuyển từ xác suất (0.9, 0.1...) về nhãn số (0, 1...)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# --- PHẦN 1: CÁC CHỈ SỐ CƠ BẢN ---
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n" + "="*40)
print("BẢNG ĐIỂM TỔNG KẾT (REPORT CARD)")
print("="*40)
print(f"1. Accuracy (Độ chính xác tổng): {acc*100:.2f}%")
print(f"2. Precision (Độ chuẩn xác):     {prec*100:.2f}%")
print(f"3. Recall (Độ bao phủ):          {rec*100:.2f}%")
print(f"4. F1-Score (Trung bình):        {f1*100:.2f}%")
print("="*40)

# In báo cáo chi tiết từng loại cảm xúc
print("\nCHI TIẾT TỪNG CẢM XÚC:")
print(classification_report(y_true, y_pred, target_names=class_names))

# --- PHẦN 2: VẼ CONFUSION MATRIX (QUAN TRỌNG NHẤT) ---
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix (Ma trận nhầm lẫn)')
plt.ylabel('Thực tế (Ground Truth)')
plt.xlabel('Robot đoán (Prediction)')
plt.tight_layout()
plt.savefig("confusion_matrix1.png") # Lưu ảnh để chèn vào báo cáo
plt.show()

print("\n✅ Đã lưu biểu đồ vào file 'confusion_matrix.png'")