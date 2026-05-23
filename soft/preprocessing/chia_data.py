import os
import random
import shutil

# --- CẤU HÌNH ---
SOURCE_FOLDER = r"D:\Aura clam\data\gian"       # Thư mục chứa data gốc của bạn
OUTPUT_FOLDER = r"D:\Aura clam\soft\dataset_5_folds_a"       # Thư mục đích sẽ chứa 5 phần
NUM_FOLDS = 5                           # Chia làm 5 phần bằng nhau

# Tạo thư mục đích nếu chưa có
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Tạo sẵn 5 thư mục con Fold_1, Fold_2... Fold_5
for i in range(1, NUM_FOLDS + 1):
    fold_path = os.path.join(OUTPUT_FOLDER, f"Fold_{i}")
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)

# 1. Đọc và phân loại file theo nhãn cảm xúc
# Dựa vào quy tắc đặt tên file: "nhan_ngaythang.wav" (VD: vui_20260111.wav)
files_by_emotion = {}

for filename in os.listdir(SOURCE_FOLDER):
    if filename.endswith(".wav"): # Chỉ lấy file âm thanh
        # Cắt chuỗi để lấy nhãn (chữ trước dấu gạch dưới đầu tiên)
        label = filename.split('_')[0] 
        
        if label not in files_by_emotion:
            files_by_emotion[label] = []
        files_by_emotion[label].append(filename)

# 2. Xáo trộn và chia đều
print("BẮT ĐẦU CHIA DỮ LIỆU...")
for label, files in files_by_emotion.items():
    print(f"\n=> Đang xử lý nhãn: [{label.upper()}] - Tổng: {len(files)} file")
    
    # Xáo trộn ngẫu nhiên danh sách file của nhãn này
    random.shuffle(files)
    
    # Tính toán số lượng file mỗi Fold sẽ nhận
    files_per_fold = len(files) // NUM_FOLDS
    
    for i in range(NUM_FOLDS):
        # Xác định điểm đầu và cuối để cắt danh sách
        start_idx = i * files_per_fold
        # Nếu là fold cuối cùng, lấy nốt tất cả các file lẻ còn dư
        end_idx = (i + 1) * files_per_fold if i < (NUM_FOLDS - 1) else len(files)
        
        fold_files = files[start_idx:end_idx]
        
        # Copy file sang thư mục Fold tương ứng
        for f in fold_files:
            src_path = os.path.join(SOURCE_FOLDER, f)
            dest_path = os.path.join(OUTPUT_FOLDER, f"Fold_{i+1}", f)
            shutil.copy2(src_path, dest_path) # Dùng copy để giữ an toàn file gốc
            
        print(f"   - Fold {i+1} nhận: {len(fold_files)} file")

print("\n[THÀNH CÔNG] Đã chia xong 5 Folds! Hãy kiểm tra thư mục:", OUTPUT_FOLDER)