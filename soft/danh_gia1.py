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
# 1. Folder dữ liệu (Đã fix lỗi NaN, chuẩn Hybrid)
DATA_FOLDER = "D:\\Aura clam\\soft\\kq_hybrid_fix" 

# 2. File Model (LƯU Ý: Phải là model đã train với 4 nhãn này. 
# Nếu dùng model cũ 14 nhãn sẽ bị lệch kết quả)
MODEL_PATH = "model_4_emotions.h5"
LABEL_MAP_NEW = "label_mapping_4_emotions.pkl" # Trỏ đúng file pkl mới tạo

IMG_SIZE = 64
N_FRAMES = 10 

# --- 1. HÀM LOAD DỮ LIỆU THÔNG MINH (LỌC 4 LOẠI) ---
def load_data_4_emotions():
    print("⏳ Đang quét dữ liệu và lọc lấy 4 cảm xúc: Vui, Buồn, Giận, Bình thường...")
    X = []
    y = []
    
    # Định nghĩa từ khóa để tìm trong tên thư mục
    # Key: Từ khóa tìm kiếm - Value: Nhãn chuẩn hóa
    keywords = {
        'happy': 'Happy',
        'sad': 'Sad',
        'angry': 'Angry',
        'neutral': 'Neutral'
    }

    count_per_label = {'Happy': 0, 'Sad': 0, 'Angry': 0, 'Neutral': 0}

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
                
                # Nếu không phải 4 loại kia thì bỏ qua
                if label is None:
                    continue

                path = os.path.join(root, file)
                try:
                    data = np.load(path)
                    # Kiểm tra kích thước chuẩn Hybrid
                    if data.shape == (N_FRAMES, IMG_SIZE, IMG_SIZE, 1):
                        X.append(data)
                        y.append(label)
                        count_per_label[label] += 1
                except: pass
    
    print("📊 Thống kê dữ liệu tìm thấy:")
    for k, v in count_per_label.items():
        print(f"   - {k}: {v} mẫu")
        
    return np.array(X), np.array(y)

# --- CHƯƠNG TRÌNH CHÍNH ---

# BƯỚC 1: Chuẩn bị dữ liệu
X, y_raw = load_data_4_emotions()

if len(X) == 0:
    print("❌ LỖI: Không tìm thấy dữ liệu nào! Hãy kiểm tra lại đường dẫn DATA_FOLDER.")
    exit()

# BƯỚC 2: Xử lý nhãn (Fix lỗi Unseen Labels)
print("\n🔄 Đang mã hóa nhãn mới...")
le = LabelEncoder()
# Fit trực tiếp trên dữ liệu 4 loại vừa tải (Không dùng file cũ nữa)
y_encoded = le.fit_transform(y_raw)
y_final = to_categorical(y_encoded)

# Lưu danh sách nhãn mới này lại để dùng sau
with open(LABEL_MAP_NEW, "wb") as f:
    pickle.dump(le.classes_, f)
print(f"✅ Đã tạo bộ nhãn mới: {le.classes_}")

# Chia tập test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)
print(f"📌 Số lượng mẫu dùng để test: {len(X_test)}")

# BƯỚC 3: Nạp Model và Dự đoán
print("\n🤖 Đang nạp Model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"❌ Lỗi không đọc được file model: {e}")
    exit()

print("🎯 Đang chạy dự đoán...")
y_pred_probs = model.predict(X_test)

# Chuyển đổi xác suất thành nhãn số (0, 1, 2, 3)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# --- BƯỚC 4: BÁO CÁO KẾT QUẢ ---

# Tính các chỉ số
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\n" + "="*40)
print("BẢNG ĐIỂM TỔNG KẾT (REPORT CARD)")
print("="*40)
print(f"1. Accuracy (Độ chính xác):  {acc*100:.2f}%")
print(f"2. Precision (Độ chuẩn xác): {prec*100:.2f}%")
print(f"3. Recall (Độ bao phủ):      {rec*100:.2f}%")
print(f"4. F1-Score (Trung bình):    {f1*100:.2f}%")
print("="*40)

# Báo cáo chi tiết theo từng lớp
print("\nCHI TIẾT TỪNG CẢM XÚC:")
# Đảm bảo tên nhãn khớp với dữ liệu test
target_names = [le.classes_[i] for i in np.unique(y_true)]
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

# --- BƯỚC 5: VẼ CONFUSION MATRIX ---
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_)
plt.title('Confusion Matrix (4 Emotions)')
plt.ylabel('Thực tế (Ground Truth)')
plt.xlabel('Robot đoán (Prediction)')
plt.tight_layout()
plt.savefig("confusion_matrix_4label.png")
plt.show()

print("\n✅ Đã lưu biểu đồ vào file 'confusion_matrix_4label.png'")