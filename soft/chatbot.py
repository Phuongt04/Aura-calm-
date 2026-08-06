"""
chatbot.py
==========
Thu âm từ microphone → nhận diện cảm xúc → gọi Ollama (local LLM)
để trò chuyện theo kịch bản phù hợp với tâm trạng người dùng.
Tối ưu cho thiết bị edge như Jetson Nano (4GB RAM).

Cài đặt Ollama trên Jetson Nano:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull tinyllama          # ~600MB – khuyến nghị
    # hoặc: ollama pull llama3.2:1b  # ~800MB

Yêu cầu Python:
    pip install openai sounddevice librosa opencv-python tensorflow
"""

import os
import warnings
import json
import pickle
import numpy as np
import cv2
import librosa
import sounddevice as sd
import h5py
from pathlib import Path
from openai import OpenAI          # openai package dùng để gọi Ollama (tương thích API)
from tensorflow.keras.models import load_model, model_from_json
from emotion_chat import audio as audio_pipeline
from emotion_chat import voice_io

# ── Tắt log thừa ────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── 1. CẤU HÌNH ─────────────────────────────────────────────────────────────
# Ollama chạy local – không cần API key thật
OLLAMA_BASE_URL = "http://localhost:11434/v1"   # đổi IP nếu Ollama chạy trên máy khác
OLLAMA_MODEL    = "tinyllama"                   # hoặc "llama3.2:1b", "phi3:mini", ...

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_PATH = PROJECT_ROOT / "final_model_archive_macro_f1_vui150_lr3e6.h5"
LABEL_PATH = BASE_DIR / "label_mapping_final.pkl"    # sinh ra bởi train.py

DURATION    = 3.0
SAMPLE_RATE = 22050
N_FRAMES    = 10
IMG_SIZE    = 64

# ── 2. SYSTEM PROMPT THEO CẢM XÚC ───────────────────────────────────────────
EMOTION_CONFIG = {
    "vui": {
        "display": "VUI VẺ ",
        "system": (
            "Bạn là một người bạn thân vui tính, nhiệt tình. "
            "Người dùng đang trong tâm trạng vui vẻ, hứng khởi. "
            "Hãy chia sẻ niềm vui đó, đặt câu hỏi gợi mở, kể chuyện hài hước, "
            "và duy trì không khí tích cực trong suốt cuộc trò chuyện. "
            "Trả lời bằng tiếng Việt, ngắn gọn khoảng 2-3 câu."
        ),
    },
    "buon": {
        "display": "BUỒN BÃ ",          
        "system": (
            "Bạn là một người bạn đồng cảm, ấm áp. "
            "Người dùng đang buồn và cần được lắng nghe, an ủi. "
            "Hãy thể hiện sự thấu hiểu, tránh phán xét, đặt câu hỏi nhẹ nhàng "
            "để khuyến khích họ chia sẻ, và đưa ra những lời động viên chân thành. "
            "Trả lời bằng tiếng Việt, giọng điệu nhẹ nhàng, khoảng 2-3 câu."
        ),
    },
    "gian": {
        "display": "TỨC GIẬN ",
        "system": (
            "Bạn là một người bạn bình tĩnh, kiên nhẫn. "
            "Người dùng đang tức giận hoặc khó chịu. "
            "Hãy thừa nhận cảm xúc của họ, KHÔNG tranh luận, KHÔNG phán xét, "
            "giúp họ giải tỏa bằng cách lắng nghe và đặt câu hỏi làm dịu tình hình. "
            "Dùng giọng điệu bình thản, xây dựng. "
            "Trả lời bằng tiếng Việt, khoảng 2-3 câu."
        ),
    },
    "binhthuong": {
        "display": "BÌNH THƯỜNG ",
        "system": (
            "Bạn là một trợ lý thông minh, thân thiện. "
            "Người dùng đang ở tâm trạng bình thường. "
            "Hãy trò chuyện tự nhiên, hữu ích, hỏi thăm xem họ cần gì hoặc "
            "muốn nói về chủ đề gì hôm nay. "
            "Trả lời bằng tiếng Việt, khoảng 2-3 câu."
        ),
    },
}
# Fallback cho nhãn không tìm thấy
DEFAULT_SYSTEM = (
    "Bạn là một trợ lý thân thiện. BẮT BUỘC luôn trả lời bằng tiếng Việt tự nhiên, có dấu. "
    "Không trả lời bằng tiếng Anh trừ khi người dùng yêu cầu dịch hoặc học tiếng Anh."
)


def _say_bot(text: str) -> None:
    """Print and speak a chatbot response."""
    print(f"\n Chatbot: {text}\n")
    voice_io.speak(text)

# ── 3. NẠP MODEL ─────────────────────────────────────────────────────────────
def _normalize_keras_config(value):
    if isinstance(value, dict):
        if value.get("class_name") == "DTypePolicy":
            return value.get("config", {}).get("name", "float32")
        return {k: _normalize_keras_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_keras_config(i) for i in value]
    return value


def _load_legacy_h5(path):
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("model_config")
    if raw is None:
        raise ValueError(f"Không tìm thấy model_config trong {path}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    cfg = _normalize_keras_config(json.loads(raw))
    for layer in cfg.get("config", {}).get("layers", []):
        lc = layer.get("config", {})
        if layer.get("class_name") == "InputLayer" and "batch_shape" in lc:
            lc["batch_input_shape"] = lc.pop("batch_shape")
    model = model_from_json(json.dumps(cfg))
    model.load_weights(path)
    return model


def load_emotion_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file nhãn: {LABEL_PATH}")

    try:
        model = load_model(str(MODEL_PATH))
    except ValueError as e:
        if "batch_shape" not in str(e):
            raise
        print("Đang dùng chế độ nạp model tương thích...")
        model = _load_legacy_h5(str(MODEL_PATH))

    with open(LABEL_PATH, "rb") as f:
        classes = list(pickle.load(f))

    print(f"Model sẵn sàng. Nhãn cảm xúc: {classes}")
    return model, classes

# ── 4. THU ÂM & TIỀN XỬ LÝ ──────────────────────────────────────────────────
def record_audio() -> np.ndarray:
    print(f"\n🎤  Đang thu âm ({DURATION:.0f}s)... Hãy nói gì đó!")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Đã ghi xong. Đang phân tích cảm xúc...")
    return audio.flatten()


def preprocess(y: np.ndarray) -> np.ndarray:
    req_len = int(SAMPLE_RATE * DURATION)
    y = np.pad(y, (0, max(0, req_len - len(y))))[:req_len]
    chunk_len = len(y) // N_FRAMES
    frames = []
    for i in range(N_FRAMES):
        chunk = y[i * chunk_len: (i + 1) * chunk_len]
        mel = librosa.feature.melspectrogram(y=chunk, sr=SAMPLE_RATE, n_mels=IMG_SIZE)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_r = cv2.resize(mel_db, (IMG_SIZE, IMG_SIZE))
        denom = mel_r.max() - mel_r.min() or 1.0
        frames.append((mel_r - mel_r.min()) / denom)
    return np.expand_dims(np.array(frames, dtype=np.float32), axis=-1)  # (10,64,64,1)


def detect_emotion(model, classes) -> tuple[str, float]:
    raw_audio = audio_pipeline.record()
    model_audio = audio_pipeline.prepare_for_model(raw_audio)
    emotion, confidence, window_count = audio_pipeline.predict_emotion_sliding(
        model,
        classes,
        model_audio,
    )
    print(f"Phan tich {window_count} cua so am thanh: {emotion} ({confidence:.1f}%)")
    return emotion, confidence

# ── 5. CHAT VỚI OPENAI ───────────────────────────────────────────────────────
def build_system_prompt(emotion_key: str) -> str:
    key = emotion_key.lower()
    cfg = EMOTION_CONFIG.get(key)
    system = cfg["system"] if cfg else DEFAULT_SYSTEM
    return (
        system
        + " BẮT BUỘC: mọi câu trả lời phải bằng tiếng Việt tự nhiên, có dấu, "
        "ngắn gọn và dễ nghe qua TTS."
    )


def chat_loop(emotion_key: str, confidence: float):
    cfg = EMOTION_CONFIG.get(emotion_key.lower(), {})
    display = cfg.get("display", emotion_key.upper())

    print("\n" + "=" * 50)
    print(f"  CẢM XÚC PHÁT HIỆN: {display}")
    print(f"  ĐỘ TIN CẬY: {confidence:.1f}%")
    print("=" * 50)
    print("Bắt đầu trò chuyện (gõ 'quit' hoặc 'thoát' để kết thúc)\n")

    # Kết nối Ollama qua OpenAI-compatible endpoint (không cần internet)
    client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
    system_prompt = build_system_prompt(emotion_key)

    # Tin nhắn mở đầu theo cảm xúc
    opening_map = {
        "vui":        "Ôi, trông bạn vui vẻ lắm đó! Hôm nay có chuyện gì vui thế?",
        "buon":       "Bạn có vẻ đang buồn... Bạn muốn kể mình nghe không?",
        "gian":       "Mình thấy bạn đang không vui. Điều gì đang xảy ra vậy?",
        "binhthuong": "Chào bạn! Hôm nay bạn cần mình giúp gì không?",
    }
    opening = opening_map.get(emotion_key.lower(), "Mình đây, bạn muốn nói chuyện gì không?")
    _say_bot(opening)

    history = [
        {"role": "system",    "content": system_prompt},
        {"role": "assistant", "content": opening},
    ]

    while True:
        user_input = input("👤 Bạn: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "thoát", "thoat", "exit"}:
            _say_bot("Tạm biệt bạn nhé! Chúc bạn một ngày tốt lành!")
            break

        history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=history,
                temperature=0.8,
                max_tokens=300,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Lỗi Ollama]: {e}")
            print("Kiểm tra Ollama đang chạy chưa: ollama serve")
            break

        history.append({"role": "assistant", "content": reply})
        _say_bot(reply)

# ── 6. MAIN ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("   HỆ THỐNG CHATBOT NHẬN BIẾT CẢM XÚC")
    print("=" * 50)

    model, classes = load_emotion_model()

    while True:
        # Nhận diện cảm xúc từ microphone
        emotion, confidence = detect_emotion(model, classes)

        # Trò chuyện theo cảm xúc
        chat_loop(emotion, confidence)

        again = input("\n Bạn muốn thử lại từ đầu không? (y/n): ").strip().lower()
        if again not in {"y", "yes", "có", "co"}:
            voice_io.speak("Hẹn gặp lại!")
            print("Hẹn gặp lại! ")
            break


if __name__ == "__main__":
    main()
