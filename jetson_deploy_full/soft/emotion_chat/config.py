"""
config.py
=========
TRUNG TÂM CẤU HÌNH — Toàn bộ hành vi hệ thống được điều chỉnh tại đây.
Không cần động vào logic ở các file khác khi muốn thay đổi kịch bản / model.
"""
import os
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent.parent   # .../soft/
PROJECT_ROOT = BASE_DIR.parent                          # .../Aura clam/


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return default if value is None or not value.strip() else Path(value).expanduser()


DEPLOY_TARGET = _env_str("AURA_DEPLOY_TARGET", "desktop")

# ── Ollama (LLM local) ────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = "http://localhost:11434/v1"   # đổi IP nếu Ollama chạy máy khác
OLLAMA_MODEL       = "tinyllama"                   # hoặc "llama3.2:1b", "phi3:mini"
OLLAMA_TEMPERATURE = 0.8
OLLAMA_MAX_TOKENS  = 300
OLLAMA_MODEL       = "llama3.2:latest"

# Deployment overrides for Jetson / embedded setups
CHAT_BACKEND       = _env_str(
    "AURA_CHAT_BACKEND",
    "rule" if DEPLOY_TARGET == "jetson" else "ollama",
)
CHAT_FALLBACK_TO_RULE_ON_ERROR = _env_flag("AURA_CHAT_FALLBACK_TO_RULE_ON_ERROR", True)
VOICE_CHAT_ENABLED = _env_flag("AURA_VOICE_CHAT_ENABLED", True)
VOICE_CHAT_LANGUAGE = _env_str("AURA_VOICE_CHAT_LANGUAGE", "vi-VN")
VOICE_CHAT_SECONDS = _env_float("AURA_VOICE_CHAT_SECONDS", 5.0)
VOICE_CHAT_STT_TIMEOUT = _env_float("AURA_VOICE_CHAT_STT_TIMEOUT", 6.0)
VOICE_CHAT_TTS_ENABLED = _env_flag("AURA_VOICE_CHAT_TTS_ENABLED", True)
VOICE_CHAT_TTS_RATE = _env_int("AURA_VOICE_CHAT_TTS_RATE", 205)
VOICE_CHAT_TTS_VOICE = _env_str(
    "AURA_VOICE_CHAT_TTS_VOICE",
    "vi" if DEPLOY_TARGET == "jetson" else ("Vietnamese" if os.name == "nt" else "vi"),
)
VOICE_CHAT_TTS_BACKEND = _env_str(
    "AURA_VOICE_CHAT_TTS_BACKEND",
    "piper" if DEPLOY_TARGET == "jetson" else ("windows" if os.name == "nt" else "linux"),
)
VOICE_CHAT_TTS_COMMAND = _env_str("AURA_VOICE_CHAT_TTS_COMMAND", "")
VOICE_CHAT_PIPER_BIN = _env_str("AURA_VOICE_CHAT_PIPER_BIN", "piper")
VOICE_CHAT_PIPER_MODEL = _env_path("AURA_VOICE_CHAT_PIPER_MODEL", Path(""))
VOICE_CHAT_PIPER_SPEAKER = _env_str("AURA_VOICE_CHAT_PIPER_SPEAKER", "")
VOICE_CHAT_PIPER_LENGTH_SCALE = _env_float("AURA_VOICE_CHAT_PIPER_LENGTH_SCALE", 1.0)
VOICE_CHAT_PIPER_NOISE_SCALE = _env_float("AURA_VOICE_CHAT_PIPER_NOISE_SCALE", 0.667)
VOICE_CHAT_PIPER_USE_CUDA = _env_flag("AURA_VOICE_CHAT_PIPER_USE_CUDA", False)
VOICE_ASSISTANT_NAME = _env_str("AURA_VOICE_ASSISTANT_NAME", "MONE")
VOICE_CHAT_MAX_EMPTY = _env_int("AURA_VOICE_CHAT_MAX_EMPTY", 2)
VOICE_RESPONSE_SPEED_BY_EMOTION: dict = {
    "vui": _env_int("AURA_VOICE_SPEED_VUI", 220),
    "buon": _env_int("AURA_VOICE_SPEED_BUON", 180),
    "gian": _env_int("AURA_VOICE_SPEED_GIAN", 195),
    "binhthuong": _env_int("AURA_VOICE_SPEED_BINHTHUONG", VOICE_CHAT_TTS_RATE),
}
WAKE_WORD_ENABLED = _env_flag("AURA_WAKE_WORD_ENABLED", True)
WAKE_WORDS = [
    word.strip().lower()
    for word in _env_str("AURA_WAKE_WORDS", "haha").split(",")
    if word.strip()
]
WAKE_LISTEN_SECONDS = _env_float("AURA_WAKE_LISTEN_SECONDS", 3.0)
WAKE_MAX_EMPTY = _env_int("AURA_WAKE_MAX_EMPTY", 0)
VOICE_RECORD_MAX_SECONDS = _env_float("AURA_VOICE_RECORD_MAX_SECONDS", 15.0)
VOICE_RECORD_SILENCE_SECONDS = _env_float("AURA_VOICE_RECORD_SILENCE_SECONDS", 1.1)
VOICE_RECORD_MIN_SECONDS = _env_float("AURA_VOICE_RECORD_MIN_SECONDS", 0.5)
VOICE_RECORD_RMS_THRESHOLD = _env_float("AURA_VOICE_RECORD_RMS_THRESHOLD", 0.012)
MUSIC_DIR = _env_path("AURA_MUSIC_DIR", BASE_DIR / "music")
MUSIC_DEFAULT_QUERY = _env_str("AURA_MUSIC_DEFAULT_QUERY", "nhac thu gian")
OLLAMA_BASE_URL    = _env_str("AURA_OLLAMA_BASE_URL", OLLAMA_BASE_URL)
OLLAMA_MODEL       = _env_str("AURA_OLLAMA_MODEL", OLLAMA_MODEL)
OLLAMA_TEMPERATURE = _env_float("AURA_OLLAMA_TEMPERATURE", OLLAMA_TEMPERATURE)
OLLAMA_MAX_TOKENS  = _env_int("AURA_OLLAMA_MAX_TOKENS", OLLAMA_MAX_TOKENS)
OLLAMA_TIMEOUT_SECONDS = _env_float("AURA_OLLAMA_TIMEOUT_SECONDS", 30.0)

# ── Model âm thanh ────────────────────────────────────────────────────────────
AUDIO_MODEL_PATH  = PROJECT_ROOT / "angry_boost_outputs (1)" / "models" / "cnn_gru_angry_boost_best.keras"
AUDIO_LABEL_PATH  = PROJECT_ROOT / "label_mapping_final.pkl"
AUDIO_FEATURE_MODE = "cnn_gru_logmel_delta"
AUDIO_DURATION    = 4.0
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_FRAMES    = 10
AUDIO_IMG_SIZE    = 64
AUDIO_WINDOW_STRIDE = 1.5
AUDIO_N_FFT       = 512
AUDIO_WIN_LENGTH  = 400
AUDIO_HOP_LENGTH  = 160
AUDIO_N_MELS      = 64
AUDIO_FMIN        = 50
AUDIO_FMAX        = 8000

AUDIO_RAW_CLASSES: list = ["neutral", "happy", "sad", "angry"]
AUDIO_LABEL_MAP: dict = {
    "neutral": "binhthuong",
    "happy": "vui",
    "sad": "buon",
    "angry": "gian",
}

AUDIO_MODEL_PATH  = _env_path("AURA_AUDIO_MODEL_PATH", AUDIO_MODEL_PATH)
AUDIO_LABEL_PATH  = _env_path("AURA_AUDIO_LABEL_PATH", AUDIO_LABEL_PATH)
AUDIO_FEATURE_MODE = _env_str("AURA_AUDIO_FEATURE_MODE", AUDIO_FEATURE_MODE)
AUDIO_DURATION    = _env_float("AURA_AUDIO_DURATION", AUDIO_DURATION)
AUDIO_SAMPLE_RATE = _env_int("AURA_AUDIO_SAMPLE_RATE", AUDIO_SAMPLE_RATE)
AUDIO_N_FRAMES    = _env_int("AURA_AUDIO_N_FRAMES", AUDIO_N_FRAMES)
AUDIO_IMG_SIZE    = _env_int("AURA_AUDIO_IMG_SIZE", AUDIO_IMG_SIZE)
AUDIO_WINDOW_STRIDE = _env_float("AURA_AUDIO_WINDOW_STRIDE", AUDIO_WINDOW_STRIDE)
AUDIO_N_FFT       = _env_int("AURA_AUDIO_N_FFT", AUDIO_N_FFT)
AUDIO_WIN_LENGTH  = _env_int("AURA_AUDIO_WIN_LENGTH", AUDIO_WIN_LENGTH)
AUDIO_HOP_LENGTH  = _env_int("AURA_AUDIO_HOP_LENGTH", AUDIO_HOP_LENGTH)
AUDIO_N_MELS      = _env_int("AURA_AUDIO_N_MELS", AUDIO_N_MELS)
AUDIO_FMIN        = _env_int("AURA_AUDIO_FMIN", AUDIO_FMIN)
AUDIO_FMAX        = _env_int("AURA_AUDIO_FMAX", AUDIO_FMAX)
AUDIO_DEMO_PATH   = _env_path("AURA_DEMO_AUDIO_PATH", Path(""))
AUDIO_MIN_RMS     = _env_float("AURA_AUDIO_MIN_RMS", 0.008)
AUDIO_MIN_PEAK    = _env_float("AURA_AUDIO_MIN_PEAK", 0.03)
AUDIO_MAX_RMS     = _env_float("AURA_AUDIO_MAX_RMS", 0.35)
AUDIO_MIN_CONFIDENCE = _env_float("AURA_AUDIO_MIN_CONFIDENCE", 0.0)
AUDIO_LOW_CONFIDENCE_FALLBACK_ENABLED = _env_flag("AURA_AUDIO_LOW_CONFIDENCE_FALLBACK_ENABLED", False)
AUDIO_LOW_CONFIDENCE_FALLBACK_EMOTION = _env_str("AURA_AUDIO_LOW_CONFIDENCE_FALLBACK_EMOTION", "binhthuong")
AUDIO_UNCERTAIN_MARGIN = _env_float("AURA_AUDIO_UNCERTAIN_MARGIN", 12.0)
AUDIO_UNCERTAIN_AGREEMENT = _env_float("AURA_AUDIO_UNCERTAIN_AGREEMENT", 0.55)
AUDIO_NOISE_FLOOR_RATIO_MAX = _env_float("AURA_AUDIO_NOISE_FLOOR_RATIO_MAX", 0.68)
AUDIO_NOISE_REDUCTION_ENABLED = _env_flag("AURA_AUDIO_NOISE_REDUCTION_ENABLED", True)
AUDIO_PREEMPHASIS_COEF = _env_float("AURA_AUDIO_PREEMPHASIS_COEF", 0.97)
AUDIO_NOISE_REDUCTION_STRENGTH = _env_float("AURA_AUDIO_NOISE_REDUCTION_STRENGTH", 0.65)
AUDIO_TRIM_SILENCE_ENABLED = _env_flag("AURA_AUDIO_TRIM_SILENCE_ENABLED", True)
AUDIO_TRIM_FRAME_MS = _env_float("AURA_AUDIO_TRIM_FRAME_MS", 30.0)
AUDIO_TRIM_HOP_MS = _env_float("AURA_AUDIO_TRIM_HOP_MS", 10.0)
AUDIO_TRIM_THRESHOLD_RATIO = _env_float("AURA_AUDIO_TRIM_THRESHOLD_RATIO", 0.18)
AUDIO_TRIM_PADDING_SECONDS = _env_float("AURA_AUDIO_TRIM_PADDING_SECONDS", 0.15)

# Runtime emotion tracking inside an active chat session.
DYNAMIC_EMOTION_TRACKING_ENABLED = _env_flag("AURA_DYNAMIC_EMOTION_TRACKING_ENABLED", True)
DYNAMIC_EMOTION_MIN_CONFIDENCE = _env_float("AURA_DYNAMIC_EMOTION_MIN_CONFIDENCE", 0.0)
DYNAMIC_FACE_EVERY_N_TURNS = _env_int("AURA_DYNAMIC_FACE_EVERY_N_TURNS", 3)
TEXT_SENTIMENT_ENABLED = _env_flag("AURA_TEXT_SENTIMENT_ENABLED", True)
TEXT_SENTIMENT_STRONG_SCORE = _env_int("AURA_TEXT_SENTIMENT_STRONG_SCORE", 2)
TEXT_SENTIMENT_OVERRIDE_UNCERTAIN = _env_flag("AURA_TEXT_SENTIMENT_OVERRIDE_UNCERTAIN", True)

# ── Model khuôn mặt ───────────────────────────────────────────────────────────
FACE_ENABLED    = False                             # ← đặt True khi có model mặt
FACE_MODEL_PATH = PROJECT_ROOT / "best_model.h5"
FACE_IMG_SIZE   = 48                                # kích thước đầu vào model mặt

# Nhãn gốc của model mặt theo ĐÚNG THỨ TỰ output index
# (thay đổi nếu dùng model khác ngoài FER2013)
FACE_ENABLED    = _env_flag("AURA_FACE_ENABLED", FACE_ENABLED)
FACE_MODEL_PATH = _env_path("AURA_FACE_MODEL_PATH", FACE_MODEL_PATH)
FACE_IMG_SIZE   = _env_int("AURA_FACE_IMG_SIZE", FACE_IMG_SIZE)
FACE_ENABLED    = _env_flag("AURA_FACE_ENABLED", True)
FACE_IMG_SIZE   = _env_int("AURA_FACE_IMG_SIZE", 224)
FACE_CAMERA_INDEX = _env_int("AURA_FACE_CAMERA_INDEX", 0)
FACE_CAPTURE_FRAMES = _env_int("AURA_FACE_CAPTURE_FRAMES", 15)
FACE_CAPTURE_WARMUP_FRAMES = _env_int("AURA_FACE_CAPTURE_WARMUP_FRAMES", 5)
FACE_DETECT_SCALE_FACTOR = _env_float("AURA_FACE_DETECT_SCALE_FACTOR", 1.1)
FACE_DETECT_MIN_NEIGHBORS = _env_int("AURA_FACE_DETECT_MIN_NEIGHBORS", 4)
FACE_DETECT_MIN_SIZE = _env_int("AURA_FACE_DETECT_MIN_SIZE", 60)

FACE_RAW_CLASSES: list = [
    "angry", "happy", "normal", "sad"
]

# Ánh xạ nhãn gốc model mặt → 4 nhãn chuẩn hệ thống
FACE_LABEL_MAP: dict = {
    "angry":    "gian",
    "happy":    "vui",
    "normal":   "binhthuong",
    "neutral":  "binhthuong",
    "sad":      "buon",
}

# ── Mức độ tiêu cực (dùng để ưu tiên khi xung đột) ───────────────────────────
# Số càng cao = càng tiêu cực = càng được ưu tiên chọn làm cảm xúc cuối
NEGATIVITY_WEIGHT: dict = {
    "gian":       3,
    "buon":       2,
    "binhthuong": 1,
    "vui":        0,
}

# ── Cấu hình cảm xúc đơn (không xung đột) ────────────────────────────────────
EMOTION_CONFIG: dict = {
    "vui": {
        "display": "VUI VẺ ",
        "opening": "Nghe bạn có vẻ đang vui đó. Chuyện gì làm hôm nay sáng lên vậy?",
        "system": (
            "Người dùng có tín hiệu đang vui hoặc hào hứng. "
            "Hãy đáp lại như một người bạn thân thiện: vui cùng họ, khuếch đại năng lượng tích cực "
            "nhưng không nói quá lố. "
            "Có thể hỏi thêm một câu mở để họ kể tiếp hoặc giúp họ tận dụng tâm trạng tốt này. "
            "Trả lời bằng tiếng Việt tự nhiên, 1-3 câu."
        ),
    },
    "buon": {
        "display": "BUỒN BÃ ",
        "opening": "Mình nghe có vẻ hôm nay bạn hơi nặng lòng. Bạn muốn kể mình nghe không?",
        "system": (
            "Người dùng có tín hiệu đang buồn, mệt hoặc cần được lắng nghe. "
            "Hãy phản hồi chậm rãi, ấm áp, không phán xét và không vội đưa lời khuyên. "
            "Ưu tiên công nhận cảm xúc của họ trước, sau đó hỏi một câu nhẹ để họ có thể chia sẻ thêm nếu muốn. "
            "Trả lời bằng tiếng Việt tự nhiên, 1-3 câu."
        ),
    },
    "gian": {
        "display": "TỨC GIẬN ",
        "opening": "Mình nghe có vẻ bạn đang rất khó chịu. Chuyện gì làm bạn bực vậy?",
        "system": (
            "Người dùng có tín hiệu đang tức giận, căng thẳng hoặc khó chịu. "
            "Hãy giữ giọng bình tĩnh, không tranh luận, không lên lớp và không bênh/vạch tội ai quá sớm. "
            "Công nhận cảm xúc của họ, giúp họ nói rõ chuyện gì xảy ra, rồi hướng cuộc trò chuyện về cách xử lý an toàn hơn. "
            "Trả lời bằng tiếng Việt tự nhiên, 1-3 câu."
        ),
    },
    "binhthuong": {
        "display": "BÌNH THƯỜNG ",
        "opening": "Mình đây. Hôm nay bạn muốn nói chuyện hay cần mình giúp gì?",
        "system": (
            "Người dùng có tín hiệu cảm xúc ổn định hoặc bình thường. "
            "Hãy trò chuyện như một trợ lý thân thiện, rõ ràng và hữu ích. "
            "Nếu họ hỏi việc cụ thể, trả lời thẳng vào vấn đề; nếu họ chỉ muốn nói chuyện, giữ nhịp tự nhiên và dễ gần. "
            "Trả lời bằng tiếng Việt tự nhiên, 1-3 câu."
        ),
    },
}

# ── Kịch bản xung đột: giọng nói ≠ khuôn mặt ─────────────────────────────────
# Key = "{voice_emotion}_vs_{face_emotion}"
# context_hint sẽ được GHÉP VÀO ĐẦU system prompt của cảm xúc được ưu tiên.
CONFLICT_CONFIG: dict = {
    "vui_vs_buon": {
        "display": "GIỌNG VUI / MẶT BUỒN ",
        "opening": (
            "Bạn có vẻ vui đấy... nhưng mình cảm giác có điều gì đó không ổn. "
            "Bạn có thật sự ổn không?"
        ),
        "context_hint": (
            "QUAN TRỌNG: Giọng nói của người dùng nghe có vẻ vui vẻ, "
            "nhưng khuôn mặt của họ lại biểu lộ sự buồn bã rõ ràng. "
            "Điều này cho thấy họ đang cố che giấu cảm xúc thật. "
            "Hãy tiếp cận nhẹ nhàng, không ép buộc. "
            "Tạo không gian an toàn để họ tự mở lòng nếu muốn. "
        ),
    },
    "vui_vs_gian": {
        "display": "GIỌNG VUI / MẶT TỨC ",
        "opening": (
            "Bạn nghe có vẻ vui, nhưng mình thấy bạn đang có gì đó bực bội. "
            "Chuyện gì vậy, kể mình nghe nhé?"
        ),
        "context_hint": (
            "QUAN TRỌNG: Giọng nói của người dùng tỏ ra vui vẻ "
            "nhưng khuôn mặt biểu lộ sự tức giận hoặc khó chịu rõ ràng. "
            "Họ đang kìm nén hoặc giả vờ ổn. "
            "Hãy thừa nhận cả hai tín hiệu một cách khéo léo, "
            "đặt câu hỏi mở để giúp họ bộc lộ nguyên nhân thật sự. "
            "Không ép, không phán xét. "
        ),
    },
    "binhthuong_vs_buon": {
        "display": "GIỌNG BÌNH THƯỜNG / MẶT BUỒN ",
        "opening": (
            "Mình thấy bạn có vẻ không được vui lắm hôm nay. "
            "Bạn có muốn nói chuyện không?"
        ),
        "context_hint": (
            "QUAN TRỌNG: Giọng nói của người dùng bình thường "
            "nhưng khuôn mặt cho thấy họ đang buồn. "
            "Họ đang cố tỏ ra ổn nhưng thực ra không ổn. "
            "Hãy tiếp cận ấm áp, đặt câu hỏi nhẹ nhàng, "
            "cho họ biết bạn sẵn sàng lắng nghe bất cứ lúc nào. "
        ),
    },
    "binhthuong_vs_gian": {
        "display": "GIỌNG BÌNH THƯỜNG / MẶT TỨC ",
        "opening": (
            "Bạn trông có vẻ đang bực bội điều gì đó. "
            "Muốn kể mình nghe không?"
        ),
        "context_hint": (
            "QUAN TRỌNG: Giọng nói của người dùng bình thản "
            "nhưng khuôn mặt biểu lộ sự tức giận hoặc khó chịu rõ ràng. "
            "Họ đang kìm nén và chưa muốn bộc lộ. "
            "Hãy tạo không gian an toàn, tiếp cận từ từ, không vội. "
            "Dùng câu hỏi mở, không gây áp lực. "
        ),
    },
    "buon_vs_gian": {
        "display": "GIỌNG BUỒN / MẶT TỨC ",
        "opening": (
            "Mình thấy bạn đang có nhiều cảm xúc lẫn lộn. "
            "Cứ nói ra đi, mình ở đây lắng nghe."
        ),
        "context_hint": (
            "QUAN TRỌNG: Người dùng vừa có dấu hiệu buồn bã qua giọng nói, "
            "vừa tức giận qua khuôn mặt — cảm xúc đang rất phức tạp. "
            "Hãy ưu tiên lắng nghe, KHÔNG đưa ra lời khuyên ngay. "
            "Giúp họ đặt tên cho cảm xúc của mình và cảm thấy được thấu hiểu trước. "
        ),
    },
    "gian_vs_buon": {
        "display": "GIỌNG TỨC / MẶT BUỒN ",
        "opening": (
            "Nghe giọng bạn có vẻ bực lắm... "
            "nhưng mình thấy cũng có gì đó đau nữa. Chuyện gì xảy ra vậy?"
        ),
        "context_hint": (
            "QUAN TRỌNG: Giọng nói biểu lộ tức giận "
            "nhưng khuôn mặt cho thấy sự đau buồn ẩn bên trong. "
            "Sự tức giận đôi khi là lớp bảo vệ bên ngoài của nỗi đau sâu hơn. "
            "Hãy tiếp cận với sự đồng cảm sâu sắc, "
            "giúp họ kết nối với cảm xúc gốc rễ một cách từ từ. "
        ),
    },
    "vui_vs_vui": {   # đồng thuận tích cực – không phải xung đột thật, giữ để tham khảo
        "display": "CẢ HAI ĐỀU VUI ",
        "opening": "Ôi, trông bạn vui vẻ lắm đó! Hôm nay có chuyện gì vui thế?",
        "context_hint": "",
    },
}

# Clarify conflict with one short follow-up question before entering the
# main chat loop. This keeps the flow focused on only two channels:
# voice + face, with the system's 4 labels.
CLARIFY_ON_CONFLICT = False

CLARIFY_FALLBACK_QUESTION = (
    "Minh thay ban dang co chut mau thuan trong cam xuc. "
    "Luc nay ban thay minh vui, buon, gian hay chi binh thuong?"
)

CLARIFY_QUESTIONS: dict = {
    "vui_vs_buon": (
        "Minh thay ban dang co gang giu on, nhung ben trong co ve hoi buon. "
        "Luc nay ban thay minh the nao?"
    ),
    "vui_vs_gian": (
        "Nghe thi co ve on, nhung minh doan la ban dang kho chiu ve dieu gi do. "
        "Ban muon noi ro hon khong?"
    ),
    "binhthuong_vs_buon": (
        "Co ve ban dang giu binh tinh, nhung cung hoi xuong suc. "
        "Luc nay ban thay buon hay chi hoi met thoi?"
    ),
    "binhthuong_vs_gian": (
        "Minh thay ban dang rat kiem che. "
        "Luc nay ban dang buc hay van binh thuong?"
    ),
    "buon_vs_gian": (
        "Nghe nhu ban vua met vua kho chiu. "
        "Luc nay cam xuc gan nhat cua ban la buon hay gian?"
    ),
    "gian_vs_buon": (
        "Minh nghe thay co ca buc boi lan met moi. "
        "Neu chon sat nhat, luc nay ban nghieng ve gian hay buon?"
    ),
}

# All phrases are stored in normalized, non-accented form to simplify
# reply parsing for short Vietnamese answers.
CLARIFY_REPLY_KEYWORDS: dict = {
    "vui": [
        "vui",
        "vui ve",
        "hao hung",
        "phan khoi",
        "happy",
        "thoai mai",
        "de chiu",
    ],
    "buon": [
        "buon",
        "met",
        "chan",
        "ap luc",
        "nang long",
        "co don",
        "stress",
        "khong on",
        "muon khoc",
        "te",
    ],
    "gian": [
        "gian",
        "tuc",
        "buc",
        "buc boi",
        "kho chiu",
        "uc che",
        "cau",
        "dien tiet",
        "phat cau",
    ],
    "binhthuong": [
        "binh thuong",
        "khong sao",
        "van on",
        "on ma",
        "tam duoc",
        "khong co gi",
        "cung duoc",
        "khong den muc",
    ],
}

DEFAULT_SYSTEM  = "Bạn là một trợ lý thân thiện. Hãy trò chuyện tự nhiên bằng tiếng Việt."
DEFAULT_OPENING = "Mình đây, bạn muốn nói chuyện gì không?"

# ── Nhận dạng danh tính qua khuôn mặt (Face ID) ─────────────────────────────
# Bật để tự động nhận ra user khi khởi động (không cần gõ tên)
FACE_ID_ENABLED     = _env_flag("AURA_FACE_ID_ENABLED", True)

# Số ảnh chụp khi đăng ký khuôn mặt mới
FACE_ID_SAMPLES_N   = 10

# Ngưỡng LBPH: giá trị THẤP hơn = khớp tốt hơn.
# < 65 → rất chắc chắn  |  65–85 → chấp nhận  |  > 85 → coi là người lạ
FACE_ID_THRESHOLD   = 80

# ── Cá nhân hóa ──────────────────────────────────────────────────────────────
PERSONALIZATION_ENABLED = True

# Thư mục gốc lưu dữ liệu người dùng
USER_DATA_DIR = BASE_DIR / "user_data"

# Số phiên gần nhất đưa vào context khi gọi LLM
HISTORY_CONTEXT_SIZE = 3
SUMMARY_ENABLED = _env_flag("AURA_SUMMARY_ENABLED", True)
SUMMARY_CONTEXT_CHARS = _env_int("AURA_SUMMARY_CONTEXT_CHARS", 1800)
SUMMARY_MAX_TOPICS = _env_int("AURA_SUMMARY_MAX_TOPICS", 12)
LONG_TERM_PERSONALIZATION_ENABLED = _env_flag("AURA_LONG_TERM_PERSONALIZATION_ENABLED", True)
LONG_TERM_CONTEXT_CHARS = _env_int("AURA_LONG_TERM_CONTEXT_CHARS", 1600)
LONG_TERM_RECENT_TOPIC_LIMIT = _env_int("AURA_LONG_TERM_RECENT_TOPIC_LIMIT", 10)
STYLE_FEEDBACK_ENABLED = _env_flag("AURA_STYLE_FEEDBACK_ENABLED", True)
STYLE_FEEDBACK_EVERY_N = _env_int("AURA_STYLE_FEEDBACK_EVERY_N", 3)

# Số lượt tối đa mỗi phiên lưu vào history (mỗi "lượt" = 1 user + 1 assistant)
MAX_HISTORY_TURNS = 6

# Bật tính năng hỏi xác nhận cảm xúc sau mỗi phiên → thu thập feedback re-train
EMOTION_FEEDBACK_ENABLED = False

# ── On-device Incremental Learning ───────────────────────────────────────────
# Bật/tắt toàn bộ tính năng tự học trên thiết bị
INCREMENTAL_ENABLED      = False

# Số mẫu xác nhận cần tích lũy trước khi trigger re-train head
# Nhỏ → cập nhật nhanh nhưng dễ overfit  |  Lớn → ổn định hơn
INCREMENTAL_TRIGGER_N    = 10

# Hyperparameter cho lượt update head (nhẹ, nhanh)
INCREMENTAL_LR           = 1e-4   # learning rate thấp để không quên kiến thức cũ
INCREMENTAL_EPOCHS       = 5
INCREMENTAL_BATCH_SIZE   = 8
