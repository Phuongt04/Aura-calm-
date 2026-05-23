"""
run_chatbot.py
==============
Entry point duy nhất.

Chạy:
    python run_chatbot.py

Để bật nhận diện khuôn mặt:
    1. Đặt FACE_ENABLED = True trong emotion_chat/config.py
    2. Đặt đúng đường dẫn FACE_MODEL_PATH
    3. Điền FACE_RAW_CLASSES theo thứ tự output của model mặt
"""
import os
import sys
from pathlib import Path

# Quiet TensorFlow's native startup logs before any TensorFlow/Keras import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Đảm bảo thư mục soft/ luôn trong sys.path dù chạy từ đâu
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emotion_chat.main import run

if __name__ == "__main__":
    run()
