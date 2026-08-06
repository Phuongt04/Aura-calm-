"""
chat.py
=======
Assistant-style conversation loop for an emotion-aware session.
"""
from __future__ import annotations

import os
import random
import unicodedata
import webbrowser
from urllib.parse import quote_plus

from . import audio, config as cfg, face
from . import voice_io
from .emotion_support.manager import EmotionSupportManager
from .emotion_support.safety import detect_risk, safety_response
from .fusion import FusionResult, apply_clarified_emotion, fuse
from .personalization import context as ctx_builder
from .personalization.history import SessionLog
from .personalization.profile import UserProfile
from .text_sentiment import TextSentimentResult, infer_text_sentiment


JOKES = [
    "Co mot con robot di thi. Thay hoi: em co quay bai khong? Robot noi: khong thay, em chi dang backup kien thuc.",
    "May tinh hoi nguoi dung: sao hom nay ban buon? Nguoi dung noi: vi deadline. May tinh dap: toi cung vay, luc nao cung bi chay deadline.",
    "MONE vua hoc nau an. Mon dau tien la mi goi, nhung van can cap nhat firmware de biet do nuoc soi.",
    "Co nguoi hoi AI: ban co biet yeu khong? AI dap: biet chu, nhung hien tai toi yeu cau nguon dien on dinh hon.",
]

VIETNAMESE_RESPONSE_REMINDER = (
    "BẮT BUỘC: luôn trả lời bằng tiếng Việt tự nhiên, có dấu. "
    "Không trả lời bằng tiếng Anh, trừ khi người dùng yêu cầu dịch hoặc học tiếng Anh. "
    "Không giải thích bằng tiếng Anh. Câu trả lời phải ngắn, dễ nghe qua TTS."
)


EXIT_PHRASES = {
    "quit",
    "exit",
    "thoat",
    "thoat chuong trinh",
    "dung",
    "dung lai",
    "tam biet",
    "ket thuc",
    "nghi",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Ä‘", "d").strip()


def _should_exit(text: str) -> bool:
    normalized = _normalize(text)
    return normalized in EXIT_PHRASES or any(
        phrase in normalized for phrase in {"tam biet", "dung lai", "ket thuc"}
    )


def _music_files() -> list:
    if not cfg.MUSIC_DIR.exists() or not cfg.MUSIC_DIR.is_dir():
        return []
    exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    return [path for path in cfg.MUSIC_DIR.iterdir() if path.suffix.lower() in exts]


def _extract_music_query(normalized: str) -> str:
    query = normalized
    for phrase in ("phat nhac", "mo nhac", "bat nhac", "cho nghe nhac"):
        query = query.replace(phrase, "")
    query = query.replace("di", "").replace("giup", "").strip()
    return query or cfg.MUSIC_DEFAULT_QUERY


def _play_music(user_input: str) -> str:
    normalized = _normalize(user_input)
    files = _music_files()
    if files:
        song = random.choice(files)
        os.startfile(song)  # type: ignore[attr-defined]
        return f"MONE mo mot bai trong thu muc nhac cua ban: {song.name}"

    query = _extract_music_query(normalized)
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    webbrowser.open(url)
    return f"MONE dang mo nhac tren YouTube voi tu khoa: {query}"


def _entertainment_reply(user_input: str) -> str | None:
    normalized = _normalize(user_input)

    if any(
        phrase in normalized
        for phrase in {"ke truyen cuoi", "ke chuyen cuoi", "truyen cuoi", "chuyen cuoi"}
    ):
        return random.choice(JOKES)

    if any(
        phrase in normalized
        for phrase in {"phat nhac", "mo nhac", "bat nhac", "cho nghe nhac"}
    ):
        try:
            return _play_music(user_input)
        except Exception as exc:
            return f"MONE chua mo duoc nhac: {exc}"

    return None


def _rule_reply(emotion: str, user_input: str) -> str:
    """Fallback assistant response when the user explicitly chooses rule mode."""
    normalized = _normalize(user_input)
    if "cam on" in normalized:
        return "Không có gì. Mình vẫn ở đây nếu bạn muốn nói thêm."
    if any(word in normalized for word in {"xin chao", "chao", "hello", "alo"}):
        return "Mình nghe đây. Bạn muốn mình hỗ trợ điều gì?"

    replies = {
        "vui": "Nghe bạn có năng lượng tốt đó. Kể mình nghe thêm đi, chuyện gì làm bạn vui hôm nay?",
        "buon": "Mình nghe bạn. Bạn cứ nói chậm thôi, điều gì đang làm bạn nặng lòng nhất lúc này?",
        "gian": "Mình hiểu là chuyện này đang làm bạn khó chịu. Mình ở đây để nghe, chuyện bắt đầu từ đâu vậy?",
        "binhthuong": "Mình đang nghe đây. Hôm nay bạn muốn nói chuyện hay cần mình giúp việc gì?",
    }
    return replies.get(emotion, replies["binhthuong"])


def _build_prompt(
    result: FusionResult,
    profile: UserProfile | None,
    recent_sessions: list | None,
) -> tuple[str, str]:
    """Return the system prompt and the opening assistant message."""
    emotion_cfg = cfg.EMOTION_CONFIG.get(result.final_emotion, {})
    conflict_cfg = cfg.CONFLICT_CONFIG.get(result.conflict_key, {}) if result.is_conflict else {}

    emotion_instruction = emotion_cfg.get("system", cfg.DEFAULT_SYSTEM)
    conflict_hint = conflict_cfg.get("context_hint", "")
    if conflict_hint:
        emotion_instruction = conflict_hint + "\n" + emotion_instruction
    if result.is_uncertain:
        emotion_instruction = (
            "Tín hiệu cảm xúc hiện tại chưa đủ rõ vì top-1/top-2 gần nhau hoặc các cửa sổ âm thanh chưa đồng thuận. "
            "Hãy trò chuyện trung tính, đừng nói như thể đã biết chắc người dùng đang vui, buồn hay giận. "
            "Có thể hỏi nhẹ xem người dùng đang muốn nói chuyện hay cần hỗ trợ gì.\n"
            + emotion_instruction
        )

    assistant_instruction = (
        f"\n\n{VIETNAMESE_RESPONSE_REMINDER} "
        f"Bạn là {cfg.VOICE_ASSISTANT_NAME}, một trợ lý ảo nói tiếng Việt tự nhiên, "
        "ấm áp và biết lắng nghe. "
        "Mỗi lượt chỉ trả lời 1-3 câu, ưu tiên câu ngắn, rõ ý, gần gũi. "
        "Tín hiệu cảm xúc chỉ là gợi ý để chọn giọng điệu, không phải chẩn đoán. "
        f"Cảm xúc ban đầu của người dùng: {result.final_emotion}. "
        "Nếu hệ thống cập nhật cảm xúc trong lúc trò chuyện, hãy điều chỉnh sắc thái trả lời "
        "theo cập nhật mới nhất nhưng không nhắc lại quy trình nhận diện. "
        "Không lặp lại lời mở đầu, không bắt đầu lại phiên, không hỏi dồn dập, "
        "không lặp cùng một câu an ủi hoặc cùng một câu hỏi. "
        "Nếu người dùng chỉ chia sẻ cảm xúc, hãy phản hồi bằng đồng cảm trước rồi hỏi tối đa một câu mở. "
        "Nếu người dùng nhờ làm việc cụ thể, hãy giúp thẳng vào việc nhưng vẫn giữ giọng phù hợp cảm xúc."
    )

    if result.is_uncertain:
        opening = "Mình nghe bạn rồi. Hôm nay bạn muốn nói chuyện hay cần mình giúp gì?"
    elif result.is_conflict and result.conflict_key:
        opening = conflict_cfg.get("opening") or emotion_cfg.get("opening", cfg.DEFAULT_OPENING)
    else:
        opening = emotion_cfg.get("opening", cfg.DEFAULT_OPENING)

    personal_block = ""
    if profile is not None and recent_sessions is not None:
        personal_block = ctx_builder.build(profile, recent_sessions)
        support_context = _build_support_context(profile)
        if support_context:
            personal_block += support_context

    return personal_block + emotion_instruction + assistant_instruction, opening


def _build_support_context(profile: UserProfile) -> str:
    """Add recent non-contact support outcomes to the LLM context."""
    try:
        manager = EmotionSupportManager()
        rows = manager.load_history(profile.user_id, limit=3)
    except Exception:
        return ""
    if not rows:
        return ""

    lines = ["[Recent Emotion Support]"]
    for row in rows:
        lines.append(
            "method={method}; detected={detected}; confirmed={confirmed}; "
            "before={before}; after={after}; completed={completed}; cancelled={cancelled}".format(
                method=row.get("selected_method", ""),
                detected=row.get("detected_emotion", ""),
                confirmed=row.get("confirmed_emotion", ""),
                before=row.get("intensity_before", ""),
                after=row.get("intensity_after", ""),
                completed=row.get("completed", False),
                cancelled=row.get("cancelled", False),
            )
        )
    lines.append(
        "Chỉ dùng thông tin này để điều chỉnh giọng điệu và thứ tự gợi ý. "
        "Không tự quyết định phương pháp hỗ trợ thay người dùng."
    )
    return "\n".join(lines) + "\n"


def _make_llm_client():
    if cfg.CHAT_BACKEND.lower() == "rule":
        return None

    from openai import OpenAI

    return OpenAI(
        api_key="ollama",
        base_url=cfg.OLLAMA_BASE_URL,
        timeout=cfg.OLLAMA_TIMEOUT_SECONDS,
    )


def _respond(client, history: list[dict], emotion: str, user_input: str) -> str:
    if client is None:
        return _rule_reply(emotion, user_input)

    messages = history + [{"role": "system", "content": VIETNAMESE_RESPONSE_REMINDER}]
    response = client.chat.completions.create(
        model=cfg.OLLAMA_MODEL,
        messages=messages,
        temperature=cfg.OLLAMA_TEMPERATURE,
        max_tokens=cfg.OLLAMA_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def _say_bot(text: str) -> None:
    print(f"\n🤖  {text}\n")
    voice_io.speak(text)


def _response_speed(emotion: str | None = None) -> int:
    emotion = (emotion or "binhthuong").lower()
    return int(
        cfg.VOICE_RESPONSE_SPEED_BY_EMOTION.get(
            emotion,
            cfg.VOICE_CHAT_TTS_RATE,
        )
    )


def _say_bot(text: str, emotion: str | None = None) -> int:
    speed = _response_speed(emotion)
    print(f"\nðŸ¤–  {text}\n  [voice] speed={speed}\n")
    voice_io.speak(text, rate=speed)
    return speed


def _add_assistant_turn(
    session: SessionLog | None,
    content: str,
    emotion: str | None = None,
) -> None:
    if session is None:
        return
    session.add_turn(
        "assistant",
        content,
        speed=_response_speed(emotion),
        emotion=emotion or "binhthuong",
    )


def _detect_turn_emotion(
    raw_audio,
    audio_model,
    audio_classes,
    face_model,
    turn_index: int,
) -> FusionResult | None:
    if not cfg.DYNAMIC_EMOTION_TRACKING_ENABLED:
        return None
    if audio_model is None or not audio_classes:
        return None
    if raw_audio is None or len(raw_audio) == 0:
        return None
    if audio.is_too_quiet(raw_audio) or audio.is_too_noisy(raw_audio):
        return None

    model_audio = audio.prepare_for_model(raw_audio)
    detail = audio.predict_emotion_sliding_detail(
        audio_model, audio_classes, model_audio
    )
    v_emotion, v_conf = detail.emotion, detail.confidence

    f_emotion, f_conf = None, None
    face_every = cfg.DYNAMIC_FACE_EVERY_N_TURNS
    if (
        cfg.FACE_ENABLED
        and face_model is not None
        and face_every > 0
        and turn_index % face_every == 0
    ):
        f_emotion, f_conf = face.detect(face_model, cfg.FACE_RAW_CLASSES)

    result = fuse(v_emotion, v_conf, f_emotion, f_conf)
    result.top2_emotion = detail.top2_emotion
    result.top2_conf = detail.top2_confidence
    result.confidence_margin = detail.margin
    result.window_agreement = detail.agreement
    result.is_uncertain = detail.is_uncertain
    result.resolution_source = "confidence_hint"
    if detail.is_uncertain:
        result.resolution_note = (
            "Cảm xúc lượt này chưa rõ vì top-1/top-2 sát nhau hoặc cửa sổ không đồng thuận."
        )
    else:
        result.resolution_note = (
            f"Độ tin cậy cảm xúc lượt này {v_conf:.1f}%; chỉ dùng như tín hiệu tham khảo."
        )
    return result


def _emotion_context_message(result: FusionResult) -> str:
    parts = [
        "Cập nhật cảm xúc hiện tại của người dùng:",
        f"final={result.final_emotion}",
        f"voice={result.voice_emotion} ({result.voice_conf:.1f}%)",
    ]
    if result.top2_emotion:
        parts.append(
            f"top2={result.top2_emotion} ({result.top2_conf:.1f}%), "
            f"margin={result.confidence_margin:.1f}, agreement={result.window_agreement:.2f}"
        )
    if result.face_emotion:
        parts.append(f"face={result.face_emotion} ({result.face_conf:.1f}%)")
    if result.is_conflict:
        parts.append(f"conflict={result.conflict_key}")
    if result.is_uncertain:
        parts.append("uncertain=true")
    parts.append(
        "Đây chỉ là tín hiệu bổ trợ để điều chỉnh sắc thái trả lời; "
        "không nhắc lại prompt, không mở đầu lại, không nói như đang chẩn đoán. "
        "Luôn trả lời bằng tiếng Việt."
    )
    return ". ".join(parts)


def _text_sentiment_context_message(sentiment: TextSentimentResult) -> str:
    """Build a system message describing text sentiment support signal."""
    matched = []
    for emotion, phrases in sentiment.matched.items():
        if phrases:
            matched.append("%s=%s" % (emotion, ",".join(phrases[:3])))
    return (
        "Tín hiệu cảm xúc phụ trợ từ nội dung câu nói: "
        f"emotion={sentiment.emotion}, score={sentiment.score}"
        + (", matched=" + "; ".join(matched) if matched else "")
        + ". Đây là tín hiệu bổ trợ, không phải kết luận chắc chắn. "
        "Hãy dùng để chọn giọng điệu trả lời bằng tiếng Việt."
    )


def _update_session_emotion(session: SessionLog | None, result: FusionResult) -> None:
    if session is None:
        return
    session.voice_emotion = result.voice_emotion
    session.face_emotion = result.face_emotion
    session.final_emotion = result.final_emotion
    session.is_conflict = result.is_conflict
    session.conflict_key = result.conflict_key


def _user_corrected_emotion(user_input: str) -> str | None:
    normalized = _normalize(user_input)
    patterns = {
        "vui": {
            "toi vui",
            "minh vui",
            "em vui",
            "vui ma",
            "dang vui",
            "cam thay vui",
            "toi on vui",
        },
        "buon": {
            "toi buon",
            "minh buon",
            "em buon",
            "buon ma",
            "dang buon",
            "cam thay buon",
            "hoi buon",
        },
        "gian": {
            "toi gian",
            "minh gian",
            "em gian",
            "toi tuc",
            "minh tuc",
            "buc minh",
            "buc boi",
            "kho chiu",
            "uc che",
        },
        "binhthuong": {
            "binh thuong",
            "binh thuong ma",
            "van on",
            "on ma",
            "khong sao",
            "khong co gi",
            "tam duoc",
        },
    }
    for emotion, phrases in patterns.items():
        if any(phrase in normalized for phrase in phrases):
            return emotion
    return None


def _should_apply_text_sentiment(
    sentiment: TextSentimentResult,
    current_result: FusionResult,
    corrected_emotion: str | None,
) -> bool:
    """Decide whether text sentiment should update the active emotion state."""
    if not cfg.TEXT_SENTIMENT_ENABLED or not sentiment.found or corrected_emotion:
        return False
    if cfg.TEXT_SENTIMENT_OVERRIDE_UNCERTAIN and current_result.is_uncertain:
        return True
    return sentiment.score >= cfg.TEXT_SENTIMENT_STRONG_SCORE


def run(
    result: FusionResult,
    profile: UserProfile | None = None,
    session: SessionLog | None = None,
    recent_sessions: list | None = None,
    audio_model=None,
    audio_classes: list | None = None,
    face_model=None,
) -> FusionResult:
    """Run the chat loop until the user exits."""
    system_prompt, opening = _build_prompt(result, profile, recent_sessions)

    print("\n" + "=" * 55)
    print(result.summary())
    if profile:
        print("-" * 55)
        print(profile.summary_str())
    print("=" * 55)

    if cfg.VOICE_CHAT_ENABLED:
        print(f"  Che do tro ly ao: {cfg.VOICE_ASSISTANT_NAME} se noi va nghe bang mic.")
        print("  Noi 'thoat', 'dung lai' hoac 'tam biet' de ket thuc.")
    else:
        print("  Che do ban phim: go 'thoat' hoac 'quit' de ket thuc.")

    if cfg.CHAT_BACKEND.lower() == "rule":
        print("  Chat backend: rule (phan hoi mau, khong phai AI that)")
    else:
        print(f"  Chat backend: {cfg.CHAT_BACKEND} | model: {cfg.OLLAMA_MODEL}")

    _say_bot(opening, result.final_emotion)

    _add_assistant_turn(session, opening, result.final_emotion)

    try:
        client = _make_llm_client()
    except Exception as exc:
        print(f"  [chat] Khong khoi tao duoc LLM: {exc}")
        if cfg.CHAT_BACKEND.lower() != "rule" and not cfg.CHAT_FALLBACK_TO_RULE_ON_ERROR:
            print("  -> Dang yeu cau backend AI that va fallback dang tat.")
            return result
        print("  -> Chuyen tam sang phan hoi rule.")
        client = None

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": opening},
    ]
    empty_count = 0
    turn_index = 0
    current_result = result
    last_context_emotion = result.final_emotion

    while True:
        raw_turn_audio = None
        if cfg.VOICE_CHAT_ENABLED:
            user_input, raw_turn_audio = voice_io.listen_once_with_audio()
            user_input = user_input.strip()
        else:
            user_input = input("👤  Bạn: ").strip()

        if not user_input:
            empty_count += 1
            if cfg.VOICE_CHAT_ENABLED and empty_count >= cfg.VOICE_CHAT_MAX_EMPTY:
                retry = "Mình chưa nghe rõ. Bạn nói lại giúp mình nhé."
                _say_bot(retry, current_result.final_emotion)
                empty_count = 0
            continue
        empty_count = 0

        if _should_exit(user_input):
            goodbye = "Tạm biệt bạn nhé. Khi nào cần thì cứ gọi mình."
            _say_bot(goodbye, current_result.final_emotion)
            if session is not None:
                session.add_turn("user", user_input)
                _add_assistant_turn(session, goodbye, current_result.final_emotion)
            break

        turn_index += 1
        turn_result = _detect_turn_emotion(
            raw_turn_audio,
            audio_model,
            audio_classes,
            face_model,
            turn_index,
        )
        if turn_result is not None:
            current_result = turn_result
            _update_session_emotion(session, current_result)
            print(
                f"  [emotion] hien tai: {current_result.final_emotion.upper()} "
                f"(voice {current_result.voice_conf:.1f}%)"
            )
            if (
                current_result.final_emotion != last_context_emotion
                or current_result.is_conflict
            ):
                history.append(
                    {"role": "system", "content": _emotion_context_message(current_result)}
                )
                last_context_emotion = current_result.final_emotion

        corrected_emotion = _user_corrected_emotion(user_input)
        text_sentiment = infer_text_sentiment(user_input) if cfg.TEXT_SENTIMENT_ENABLED else TextSentimentResult()
        if text_sentiment.found:
            print(
                f"  [text] cam xuc noi dung: {text_sentiment.emotion.upper()} "
                f"(score={text_sentiment.score})"
            )

        if corrected_emotion and corrected_emotion != current_result.final_emotion:
            current_result = apply_clarified_emotion(
                current_result,
                corrected_emotion,
                "nguoi dung tu hieu chinh cam xuc trong hoi thoai",
            )
            _update_session_emotion(session, current_result)
            history.append(
                {"role": "system", "content": _emotion_context_message(current_result)}
            )
            last_context_emotion = current_result.final_emotion
            print(f"  [emotion] nguoi dung hieu chinh: {corrected_emotion.upper()}")
        elif _should_apply_text_sentiment(text_sentiment, current_result, corrected_emotion):
            if text_sentiment.emotion != current_result.final_emotion:
                current_result = apply_clarified_emotion(
                    current_result,
                    text_sentiment.emotion,
                    "suy ra tu noi dung cau noi cua nguoi dung",
                )
                _update_session_emotion(session, current_result)
                last_context_emotion = current_result.final_emotion
                print(f"  [emotion] cap nhat tu noi dung: {text_sentiment.emotion.upper()}")
            history.append(
                {"role": "system", "content": _text_sentiment_context_message(text_sentiment)}
            )

        history.append({"role": "user", "content": user_input})
        if session is not None:
            session.add_turn("user", user_input)

        safety = detect_risk(user_input)
        if safety.risk_detected:
            reply = safety_response()
            history.append({"role": "assistant", "content": reply})
            if session is not None:
                _add_assistant_turn(session, reply, current_result.final_emotion)
            print(f"  [safety] escalation: {safety.category}")
            _say_bot(reply, current_result.final_emotion)
            continue

        entertainment = _entertainment_reply(user_input)
        if entertainment is not None:
            history.append({"role": "assistant", "content": entertainment})
            if session is not None:
                _add_assistant_turn(session, entertainment, current_result.final_emotion)
            _say_bot(entertainment, current_result.final_emotion)
            continue

        try:
            reply = _respond(client, history, current_result.final_emotion, user_input)
        except Exception as exc:
            print(f"\n  [Loi chat backend]: {exc}")
            if cfg.CHAT_BACKEND.lower() != "rule" and not cfg.CHAT_FALLBACK_TO_RULE_ON_ERROR:
                print("  -> Backend AI that dang loi va fallback dang tat.")
                break
            print("  -> Chuyen tam sang phan hoi rule.")
            client = None
            reply = _rule_reply(current_result.final_emotion, user_input)

        history.append({"role": "assistant", "content": reply})
        if session is not None:
            _add_assistant_turn(session, reply, current_result.final_emotion)

        _say_bot(reply, current_result.final_emotion)

    return current_result
