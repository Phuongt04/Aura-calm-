"""
main.py
=======
Main runtime pipeline for MONE:
  1. Nhận dạng người dùng.
  2. Nạp model âm thanh và model khuôn mặt nếu bật.
  3. Thu âm, đọc cảm xúc khuôn mặt nếu có, rồi hợp nhất tín hiệu.
  4. Xây context cá nhân hóa từ profile/history/summary/long-term memory.
  5. Chạy vòng chat.
  6. Lưu lịch sử phiên và cập nhật bộ nhớ cá nhân hóa.

Cá nhân hóa dựa trên ký ức hội thoại, không hỏi người dùng tự gắn nhãn cảm xúc.
"""
from __future__ import annotations

import os
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from . import audio, chat, face, face_id, voice_io
from . import config as cfg
from .emotion_support import EmotionSupportSession
from .emotion_support.manager import EmotionSupportManager
from .fusion import (
    FusionResult,
    apply_clarified_emotion,
    build_clarifying_question,
    fuse,
    infer_emotion_from_reply,
)
from .model_loader import load_audio_model, load_face_model
from .personalization.history import SessionLog, load_recent
from .personalization.long_term import update_long_term_personalization
from .personalization.profile import UserProfile
from .personalization.summary import update_periodic_summaries


def _yes(answer: str) -> bool:
    return answer.strip().lower() in {"y", "yes", "co", "c", "có"}


def _say(text: str) -> None:
    """Speak short user-facing runtime messages."""
    voice_io.speak(text)


def _ask(prompt: str, speak_prompt: str | None = None) -> str:
    if speak_prompt is not None:
        _say(speak_prompt)
    return input(prompt).strip()


def _identify_user() -> UserProfile:
    """Nhận dạng bằng Face ID nếu có, nếu không thì hỏi user id."""
    print("\n" + "-" * 55)

    if cfg.FACE_ID_ENABLED:
        print("[face_id] Đang nhận diện người dùng...")
        uid, conf = face_id.recognize()
        if uid is not None:
            profile = UserProfile.load(uid)
            if profile is not None:
                greeting = f"Xin chào lại, {profile.display_name}!"
                print(f"  {greeting} (confidence={conf:.1f}, {profile.total_sessions} phiên trước)")
                _say(greeting)
                return profile

    user_id = _ask(
        "Nhập tên/ID của bạn (Enter = guest): ",
        "Nhập tên hoặc ID của bạn. Bấm Enter để dùng guest.",
    )
    if not user_id:
        user_id = "guest"
    user_id = user_id.lower().replace(" ", "_")

    profile = UserProfile.load(user_id)
    if profile is None:
        display = _ask(
            f"Chưa có hồ sơ cho '{user_id}'. Tên hiển thị: ",
            f"Chưa có hồ sơ cho {user_id}. Bạn muốn MONE gọi bạn là gì?",
        )
        profile = UserProfile.load_or_create(user_id, display or user_id)
        print("  Đã tạo hồ sơ mới.")
        _say(f"Đã tạo hồ sơ mới cho {profile.display_name}.")

        if cfg.FACE_ID_ENABLED:
            ans = _ask(
                "Đăng ký khuôn mặt để lần sau tự nhận diện? (y/n): ",
                "Bạn có muốn đăng ký khuôn mặt để lần sau MONE tự nhận diện không?",
            )
            if _yes(ans):
                face_id.enroll_user(user_id)
    else:
        greeting = f"Xin chào lại, {profile.display_name}!"
        print(f"  {greeting} ({profile.total_sessions} phiên trước)")
        _say(greeting)

        if cfg.FACE_ID_ENABLED and not face_id.is_enrolled(user_id):
            ans = _ask(
                "Bạn chưa đăng ký khuôn mặt. Đăng ký ngay? (y/n): ",
                "Bạn chưa đăng ký khuôn mặt. Bạn có muốn đăng ký ngay không?",
            )
            if _yes(ans):
                face_id.enroll_user(user_id)

    return profile


def _ask_style_feedback(profile: UserProfile) -> None:
    """Thỉnh thoảng học kiểu phản hồi phù hợp với người dùng."""
    if not cfg.STYLE_FEEDBACK_ENABLED:
        return
    if cfg.STYLE_FEEDBACK_EVERY_N <= 0:
        return
    if profile.total_sessions == 0:
        return
    if profile.total_sessions % cfg.STYLE_FEEDBACK_EVERY_N != 0:
        return

    styles = {
        "1": "ngắn gọn",
        "2": "dịu nhẹ đồng cảm",
        "3": "hỏi nhiều hơn",
        "4": "động viên tích cực",
        "5": "thẳng vào vấn đề",
    }
    print("\n[cá nhân hóa] Cách MONE nói chuyện vừa rồi có hợp với bạn không?")
    print("Chọn kiểu nên dùng thêm: [1]ngắn gọn [2]dịu nhẹ [3]hỏi nhiều [4]động viên [5]thẳng vấn đề")
    _say("Cách MONE nói chuyện vừa rồi có hợp với bạn không? Chọn số nếu bạn muốn MONE ghi nhớ.")
    ans = _ask("Enter=bỏ qua | số=ghi nhớ: ")
    if ans in styles:
        profile.record_style_feedback(preferred=styles[ans])
        print(f"[cá nhân hóa] Đã ghi nhớ: {styles[ans]}")

    avoid = _ask(
        "Có kiểu nào nên giảm không? (1-5 hoặc Enter): ",
        "Có kiểu nào MONE nên giảm không?",
    )
    if avoid in styles:
        profile.record_style_feedback(avoid=styles[avoid])
        print(f"[cá nhân hóa] Sẽ giảm: {styles[avoid]}")


def _clarify_conflict(result: FusionResult) -> tuple[FusionResult, str, str]:
    """Hỏi một câu ngắn khi giọng nói và khuôn mặt bị lệch tín hiệu."""
    if not cfg.CLARIFY_ON_CONFLICT or not result.is_conflict or not result.face_emotion:
        return result, "", ""

    print("\n[Xung đột] MONE hỏi thêm 1 câu ngắn để chốt cảm xúc hiện tại.")
    question = build_clarifying_question(result)
    print(f"MONE: {question}")
    _say(question)
    answer = _ask("Bạn: ")

    if not answer:
        return result, question, ""

    final_emotion, reason = infer_emotion_from_reply(answer, result)
    updated = apply_clarified_emotion(result, final_emotion, reason)
    print(f"-> Tạm chốt: {final_emotion.upper()} ({reason})")
    _say(f"MONE tạm chốt cảm xúc hiện tại là {final_emotion}.")
    return updated, question, answer


def _load_models():
    print("\n[Đang nạp model...]")
    audio_model, audio_classes = load_audio_model()
    face_model = load_face_model() if cfg.FACE_ENABLED else None
    print("\n[Tất cả model đã sẵn sàng!]\n")
    return audio_model, audio_classes, face_model


def _detect_emotion(audio_model, audio_classes, face_model) -> FusionResult | str | None:
    if cfg.WAKE_WORD_ENABLED and not voice_io.wait_for_wake_word():
        return "stop"

    raw_audio = audio.record()
    if len(raw_audio) == 0:
        print("[audio] Chưa phát hiện câu nói. MONE quay lại chế độ chờ.")
        return None

    rms, peak = audio.quality_stats(raw_audio)
    noise_ratio = audio.noise_floor_ratio(raw_audio)
    print(f"[audio] rms={rms:.4f}, peak={peak:.4f}, noise_floor_ratio={noise_ratio:.2f}")
    if audio.is_too_quiet(raw_audio):
        print("[audio] Tín hiệu mic quá nhỏ/không rõ. MONE không phân tích AI.")
        return None

    if audio.is_too_noisy(raw_audio):
        print("[audio] Moi truong qua on/tin hieu bi nhieu. MONE bo qua doan nay.")
        return None

    model_audio = audio.prepare_for_model(raw_audio)
    detail = audio.predict_emotion_sliding_detail(
        audio_model, audio_classes, model_audio
    )
    v_emotion, v_conf, window_count = detail.emotion, detail.confidence, detail.window_count
    print(
        f"[audio] sliding windows={window_count}, "
        f"window={cfg.AUDIO_DURATION:.1f}s, stride={cfg.AUDIO_WINDOW_STRIDE:.1f}s"
    )
    print(f"Giọng nói: {v_emotion.upper()} ({v_conf:.1f}%)")
    print(
        f"[audio] top2={detail.top2_emotion.upper()} ({detail.top2_confidence:.1f}%), "
        f"margin={detail.margin:.1f}, agreement={detail.agreement:.2f}, "
        f"votes={detail.window_votes}"
    )

    f_emotion, f_conf = None, None
    if cfg.FACE_ENABLED and face_model is not None:
        f_emotion, f_conf = face.detect(face_model, cfg.FACE_RAW_CLASSES)
        if f_emotion:
            print(f"Khuôn mặt: {f_emotion.upper()} ({f_conf:.1f}%)")
        else:
            print("Khuôn mặt: không phát hiện, bỏ qua")

    result = fuse(v_emotion, v_conf, f_emotion, f_conf)
    result.resolution_source = "confidence_hint"
    result.top2_emotion = detail.top2_emotion
    result.top2_conf = detail.top2_confidence
    result.confidence_margin = detail.margin
    result.window_agreement = detail.agreement
    result.is_uncertain = detail.is_uncertain
    if detail.is_uncertain:
        result.resolution_note = (
            f"Top-1 và top-2 còn gần nhau hoặc các cửa sổ chưa đồng thuận; "
            f"cảm xúc chỉ là gợi ý mềm."
        )
    else:
        result.resolution_note = (
            f"Độ tin cậy giọng nói {v_conf:.1f}%; tín hiệu cảm xúc dùng như gợi ý."
        )
    result, _, _ = _clarify_conflict(result)
    return result


def _support_emotion_to_runtime(value: str) -> str:
    mapping = {
        "happy": "vui",
        "sad": "buon",
        "angry": "gian",
        "neutral": "binhthuong",
    }
    return mapping.get(value, "")


def _run_emotion_support(result: FusionResult, profile: UserProfile | None) -> FusionResult:
    """Run optional non-contact emotion support before normal chat."""
    try:
        user_id = profile.user_id if profile is not None else "guest"
        profile_sessions = profile.total_sessions if profile is not None else 0
        manager = EmotionSupportManager()
        if result.is_uncertain:
            return result
        if not manager.should_start_support(
            result.final_emotion,
            result.voice_conf,
            user_id=user_id,
            profile_total_sessions=profile_sessions,
        ):
            return result

        support = EmotionSupportSession(manager=manager)
        support_result = support.run(result.final_emotion, result.voice_conf, user_id=user_id)
        confirmed = _support_emotion_to_runtime(support_result.confirmed_emotion)
        if confirmed and confirmed != result.final_emotion:
            return apply_clarified_emotion(
                result,
                confirmed,
                "nguoi dung xac nhan lai trong phien ho tro cam xuc",
            )
    except Exception as exc:
        print(f"[emotion_support] Bo qua phien ho tro do loi: {exc}")
    return result


def run() -> None:
    print("=" * 55)
    print("   HỆ THỐNG CHATBOT NHẬN BIẾT CẢM XÚC ĐA NGUỒN")
    print(
        f"   Chế độ: {'Giọng nói + Khuôn mặt' if cfg.FACE_ENABLED else 'Giọng nói'}"
        f" | Cá nhân hóa: {'Bật' if cfg.PERSONALIZATION_ENABLED else 'Tắt'}"
    )
    print("=" * 55)

    profile = _identify_user() if cfg.PERSONALIZATION_ENABLED else None
    audio_model, audio_classes, face_model = _load_models()

    while True:
        print("\n" + "-" * 55)
        result = _detect_emotion(audio_model, audio_classes, face_model)
        if result == "stop":
            print("Hẹn gặp lại!")
            _say("Hẹn gặp lại!")
            break
        if result is None:
            if cfg.WAKE_WORD_ENABLED:
                continue
            retry = _ask("\nThử lại không? (y/n): ", "Bạn muốn thử lại không?")
            if not _yes(retry):
                print("Hẹn gặp lại!")
                _say("Hẹn gặp lại!")
                break
            continue

        result = _run_emotion_support(result, profile)

        recent_sessions = []
        session = None
        if profile is not None:
            recent_sessions = load_recent(profile.user_id)
            session = SessionLog.from_fusion(result)

        result = chat.run(
            result,
            profile,
            session,
            recent_sessions,
            audio_model=audio_model,
            audio_classes=audio_classes,
            face_model=face_model,
        )

        if profile is not None and session is not None:
            session.save(profile.user_id)
            profile.record_session(result.final_emotion, result.conflict_key)
            profile.save()

            summary_paths = update_periodic_summaries(profile.user_id)
            long_term_paths = update_long_term_personalization(profile.user_id, profile)
            print(f"\nĐã lưu phiên vào hồ sơ '{profile.display_name}'")
            for path in summary_paths:
                print(f"[summary] {path.name}")
            for path in long_term_paths:
                print(f"[long-term] {path.name}")

            _ask_style_feedback(profile)
            update_long_term_personalization(profile.user_id, profile)
            profile.save()

        again = _ask("\nBạn muốn thử lại không? (y/n): ", "Bạn muốn thử lại không?")
        if not _yes(again):
            print("Hẹn gặp lại!")
            _say("Hẹn gặp lại!")
            break


if __name__ == "__main__":
    run()
