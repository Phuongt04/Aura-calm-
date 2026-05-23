"""
Đánh giá hiệu quả model audio trên dữ liệu test.
Sử dụng các feature đã tiền xử lý (.npy) từ folder kq/
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import h5py
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model, model_from_json


def _normalize_keras_config(value):
    """Chuẩn hóa config Keras cũ."""
    if isinstance(value, dict):
        if value.get("class_name") == "DTypePolicy":
            return value.get("config", {}).get("name", "float32")
        return {k: _normalize_keras_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_keras_config(i) for i in value]
    return value


def _load_legacy_h5(path: Path):
    """Nạp model h5 cũ có batch_shape thay vì batch_input_shape."""
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("model_config")
    if raw is None:
        raise ValueError(f"Không tìm thấy model_config trong {path}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    cfg_data = _normalize_keras_config(json.loads(raw))
    for layer in cfg_data.get("config", {}).get("layers", []):
        lc = layer.get("config", {})
        if layer.get("class_name") == "InputLayer" and "batch_shape" in lc:
            lc["batch_input_shape"] = lc.pop("batch_shape")
    model = model_from_json(json.dumps(cfg_data))
    model.load_weights(str(path))
    return model


def safe_load_keras(path: Path):
    """Nạp model, tự động xử lý lỗi tương thích batch_shape."""
    try:
        return load_model(str(path))
    except ValueError as exc:
        if "batch_shape" not in str(exc):
            raise
        print(f"  [model_loader] Chế độ legacy cho {path.name}...")
        return _load_legacy_h5(path)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

# ─── Cấu hình ───────────────────────────────────────────────────────────────
MODEL_PATH = REPO_ROOT / "best_model_improved.h5"
LABEL_PATH = REPO_ROOT / "label_mapping_final.pkl"
TEST_DATA_DIR = PROJECT_ROOT / "dataset"

# Ánh xạ folder TESS thô → 4 nhãn chuẩn
EMOTION_MAP = {
    "OAF_angry": "gian", "YAF_angry": "gian",
    "OAF_disgust": "gian", "YAF_disgust": "gian",
    "OAF_Fear": "buon", "YAF_fear": "buon",
    "OAF_happy": "vui", "YAF_happy": "vui",
    "OAF_neutral": "binhthuong", "YAF_neutral": "binhthuong",
    "OAF_Pleasant_surprise": "vui", "YAF_pleasant_surprised": "vui",
    "OAF_Sad": "buon", "YAF_sad": "buon",
}

print("=" * 70)
print("   ĐÁNH GIÁ HIỆU QUẢ MODEL ÂM THANH")
print("=" * 70)

# ─── 1. Nạp model và label ───────────────────────────────────────────────────
print("\n[1] Đang nạp model...")
if not MODEL_PATH.exists():
    print(f"❌ Lỗi: Không tìm thấy model tại {MODEL_PATH}")
    exit(1)

model = safe_load_keras(MODEL_PATH)
print(f"  ✅ Model loaded: {model.output_shape}")

print("\n[2] Đang nạp label mapping...")
if not LABEL_PATH.exists():
    # Thử fallback sang label_mapping.pkl
    alt_label_path = BASE_DIR / "label_mapping.pkl"
    if alt_label_path.exists():
        LABEL_PATH = alt_label_path
        print(f"  ⚠️  label_mapping_final.pkl không tìm thấy, dùng {alt_label_path.name}")
    else:
        print(f"❌ Lỗi: Không tìm thấy label mapping tại {LABEL_PATH}")
        exit(1)

with open(LABEL_PATH, "rb") as f:
    labels = list(pickle.load(f))
print(f"  ✅ Labels: {labels}")

# ─── 2. Tải dữ liệu test ─────────────────────────────────────────────────────
print(f"\n[3] Đang tải dữ liệu test từ {TEST_DATA_DIR}...")
if not TEST_DATA_DIR.exists():
    print(f"❌ Lỗi: Không tìm thấy folder test tại {TEST_DATA_DIR}")
    exit(1)

X_test = []
y_test_true = []
y_test_labels = []
file_count = 0

for emotion_dir in sorted(TEST_DATA_DIR.iterdir()):
    if not emotion_dir.is_dir():
        continue
    
    folder_name = emotion_dir.name
    mapped_emotion = EMOTION_MAP.get(folder_name, folder_name)
    
    if mapped_emotion not in labels:
        print(f"  ⚠️  {folder_name} không trong labels, bỏ qua")
        continue
    
    label_idx = labels.index(mapped_emotion)
    
    # Tải tất cả .npy files trong folder
    npy_files = list(emotion_dir.glob("*.npy"))
    for npy_file in npy_files:
        try:
            features = np.load(str(npy_file), allow_pickle=False)
            # Ensure shape is (10, 64, 64, 1) - add channel if needed
            if len(features.shape) == 3:  # (10, 64, 64)
                features = np.expand_dims(features, axis=-1)
            X_test.append(features)
            y_test_true.append(label_idx)
            y_test_labels.append(mapped_emotion)
            file_count += 1
        except Exception as e:
            print(f"  ⚠️  Lỗi khi tải {npy_file}: {e}")

X_test = np.array(X_test, dtype=np.float32)
y_test_true = np.array(y_test_true)
print(f"  ✅ Tải được {file_count} mẫu")
print(f"     Shape: {X_test.shape}, Labels: {np.unique(y_test_true)}")

# ─── 3. Dự đoán ──────────────────────────────────────────────────────────────
print("\n[4] Đang dự đoán...")
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# ─── 4. Tính toán metrics ────────────────────────────────────────────────────
print("\n[5] Tính toán metrics...")
acc = accuracy_score(y_test_true, y_pred)
print(f"\n📊 ACCURACY: {acc*100:.2f}%")

# Per-class metrics
print("\n📋 CHI TIẾT PER-CLASS:")
prec = precision_score(y_test_true, y_pred, average=None, zero_division=0)
rec = recall_score(y_test_true, y_pred, average=None, zero_division=0)
f1 = f1_score(y_test_true, y_pred, average=None, zero_division=0)

for i, label in enumerate(labels):
    print(f"  {label.upper():12} → Prec: {prec[i]*100:6.2f}%  Recall: {rec[i]*100:6.2f}%  F1: {f1[i]*100:6.2f}%")

# Macro averages
print(f"\n  Macro Avg   → Prec: {prec.mean()*100:6.2f}%  Recall: {rec.mean()*100:6.2f}%  F1: {f1.mean()*100:6.2f}%")

# ─── 5. Confusion matrix ─────────────────────────────────────────────────────
print("\n[6] Confusion Matrix:")
cm = confusion_matrix(y_test_true, y_pred)
print(cm)

# ─── 6. Lưu biểu đồ ──────────────────────────────────────────────────────────
print("\n[7] Vẽ biểu đồ...")

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", 
    xticklabels=labels, yticklabels=labels,
    cbar_kws={"label": "Count"},
    ax=axes
)
axes.set_title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%)", fontsize=14, fontweight="bold")
axes.set_ylabel("True Label", fontsize=12)
axes.set_xlabel("Predicted Label", fontsize=12)

output_path = BASE_DIR / "emotion_confusion_matrix.png"
plt.tight_layout()
plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
print(f"  ✅ Lưu biểu đồ: {output_path}")

# ─── 7. Classification report ────────────────────────────────────────────────
print("\n[8] Chi tiết từng class:")
print(classification_report(y_test_true, y_pred, target_names=labels))

# ─── 8. Thống kê ──────────────────────────────────────────────────────────────
print("\n[9] Thống kê:")
print(f"  Tổng mẫu: {len(X_test)}")
print(f"  Mẫu đúng: {(y_pred == y_test_true).sum()}")
print(f"  Mẫu sai: {(y_pred != y_test_true).sum()}")

# Per-class sample count
for i, label in enumerate(labels):
    count = (y_test_true == i).sum()
    correct = ((y_pred == i) & (y_test_true == i)).sum()
    print(f"  {label.upper():12} → {count:4} mẫu ({correct:4} dự đoán đúng)")

print("\n" + "=" * 70)
print(f"✅ Đánh giá hoàn thành! Kết quả được lưu vào {output_path}")
print("=" * 70)