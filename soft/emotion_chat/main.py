"""
main.py
=======
Điều phối toàn bộ pipeline:
  1. Nhận dạng người dùng (user_id)
  2. Nạp model âm thanh (+ mặt nếu bật) + nạp head weights cá nhân hóa
  3. Thu âm + chụp ảnh → fusion cảm xúc
  4. Tải lịch sử → xây context cá nhân hóa
  5. Chat theo kịch bản
  6. Lưu lịch sử phiên + cập nhật hồ sơ
  7. Hỏi xác nhận cảm xúc → thêm vào FeedbackBuffer
  8. Khi buffer đủ INCREMENTAL_TRIGGER_N mẫu → update head ON-DEVICE (không cần PC)
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from . import config as cfg
from .model_loader import load_audio_model, load_face_model
from . import audio, face, face_id, chat
from .fusion import (
    FusionResult,
    apply_clarified_emotion,
    build_clarifying_question,
    fuse,
    infer_emotion_from_reply,
)
from .personalization.profile import UserProfile
from .personalization.history import SessionLog, load_recent
from .personalization.export import save_emotion_feedback
from .personalization.incremental import (
    FeedbackBuffer, freeze_backbone, load_user_head, run_incremental_update
)


# ── Nhận dạng người dùng ───────────────────────────────────────────────────
def _identify_user() -> UserProfile:
    """
    Thử nhận dạng qua khuôn mặt trước (nếu FACE_ID_ENABLED).
    Nếu nhận ra → load hồ sơ ngay, không cần gõ gì.
    Nếu không nhận ra / tắt face_id → fallback về nhập tay.
    Người dùng mới được hỏi có muốn đăng ký khuôn mặt không.
    """
    print("\n" + "─" * 55)

    # ── Thử nhận dạng bằng khuôn mặt ────────────────────────────────────
    if cfg.FACE_ID_ENABLED:
        print("📷  Đang nhận diện khuôn mặt...")
        uid, conf = face_id.recognize()
        if uid is not None:
            profile = UserProfile.load(uid)
            if profile is not None:
                print(f"  ✅ Xin chào lại, {profile.display_name}! "
                      f"(confidence={conf:.1f}, {profile.total_sessions} phiên trước)")
                return profile

    # ── Fallback: nhập tay ───────────────────────────────────────────────
    user_id = input("👤  Nhập tên/ID của bạn (Enter = 'guest'): ").strip()
    if not user_id:
        user_id = "guest"
    user_id = user_id.lower().replace(" ", "_")

    profile = UserProfile.load(user_id)
    if profile is None:
        display = input(f"  Chưa có hồ sơ cho '{user_id}'. Tên hiển thị: ").strip()
        profile = UserProfile.load_or_create(user_id, display or user_id)
        print(f"  ✅ Đã tạo hồ sơ mới.")

        # Hỏi đăng ký khuôn mặt ngay khi tạo hồ sơ mới
        if cfg.FACE_ID_ENABLED:
            ans = input("  📷  Bạn muốn đăng ký khuôn mặt để lần sau tự nhận diện không? (y/n): ").strip().lower()
            if ans in {"y", "yes", "có", "co"}:
                face_id.enroll_user(user_id)
    else:
        print(f"  ✅ Xin chào lại, {profile.display_name}! ({profile.total_sessions} phiên trước)")

        # Nếu bật face_id nhưng user chưa đăng ký khuôn mặt
        if cfg.FACE_ID_ENABLED and not face_id.is_enrolled(user_id):
            ans = input("  📷  Bạn chưa đăng ký khuôn mặt. Đăng ký ngay không? (y/n): ").strip().lower()
            if ans in {"y", "yes", "có", "co"}:
                face_id.enroll_user(user_id)

    return profile


# ── Hỏi xác nhận cảm xúc + thêm vào FeedbackBuffer ─────────────────────
def _ask_emotion_feedback(
    profile:           UserProfile,
    predicted_emotion: str,
    audio_features,
    buffer:            FeedbackBuffer | None,
) -> str:
    """
    Hỏi user xem cảm xúc dự đoán có đúng không.
    Trả về nhãn đã xác nhận (dù đúng hay sai).
    """
    if not cfg.EMOTION_FEEDBACK_ENABLED or audio_features is None:
        return predicted_emotion

    all_emotions = list(cfg.EMOTION_CONFIG.keys())
    options      = " / ".join(f"[{i+1}]{e}" for i, e in enumerate(all_emotions))
    print(f"\n  Hệ thống đoán: {predicted_emotion.upper()}")
    ans = input(f"  Đúng không? (Enter=Đúng | {options}): ").strip().lower()

    confirmed_label = predicted_emotion
    if ans not in {"", "y", "yes", "đúng", "dung"}:
        if ans.isdigit():
            idx = int(ans) - 1
            if 0 <= idx < len(all_emotions):
                confirmed_label = all_emotions[idx]
        elif ans in all_emotions:
            confirmed_label = ans
        if confirmed_label != predicted_emotion:
            print(f"  → Nhãn đã sửa: {confirmed_label.upper()} — cảm ơn!")

    # Lưu feedback text
    save_emotion_feedback(
        profile.user_id,
        audio_features,
        predicted_label=predicted_emotion,
        corrected_label=confirmed_label,
        confirmed_by_user=(confirmed_label == predicted_emotion),
    )

    # Thêm vào buffer on-device
    if buffer is not None and cfg.INCREMENTAL_ENABLED:
        count = buffer.add(audio_features, confirmed_label)
        remaining = cfg.INCREMENTAL_TRIGGER_N - count
        if remaining > 0:
            print(f"  📊  Buffer: {count}/{cfg.INCREMENTAL_TRIGGER_N} mẫu "
                  f"(cần thêm {remaining} để cập nhật model)")
        else:
            print(f"  📊  Buffer đủ {count} mẫu — sẵn sàng cập nhật!")

    return confirmed_label


def _clarify_conflict(result: FusionResult) -> tuple[FusionResult, str, str]:
    """
    Ask one short follow-up question when voice and face conflict.
    Returns the updated result plus the clarification turn pair.
    """
    if not cfg.CLARIFY_ON_CONFLICT or not result.is_conflict or not result.face_emotion:
        return result, "", ""

    print("\n  [Xung dot] He thong se hoi them 1 cau ngan de xac nhan cam xuc.")
    question = build_clarifying_question(result)
    print(f"  🤖  {question}")
    answer = input("  👤  ").strip()

    if not answer:
        return result, question, ""

    final_emotion, reason = infer_emotion_from_reply(answer, result)
    updated = apply_clarified_emotion(result, final_emotion, reason)
    print(f"  -> Tam chot: {final_emotion.upper()} ({reason})")
    return updated, question, answer


# ── Pipeline chính ────────────────────────────────────────────────────────
def run() -> None:
    print("=" * 55)
    print("   HỆ THỐNG CHATBOT NHẬN BIẾT CẢM XÚC ĐA NGUỒN")
    print(f"   Chế độ: {'Giọng nói + Khuôn mặt' if cfg.FACE_ENABLED else 'Giọng nói'}"
          f" | Cá nhân hóa: {'Bật' if cfg.PERSONALIZATION_ENABLED else 'Tắt'}")
    print("=" * 55)

    # ── Bước 1: Nhận dạng người dùng ─────────────────────────────────────
    profile = _identify_user() if cfg.PERSONALIZATION_ENABLED else None

    # ── Bước 2: Nạp model ─────────────────────────────────────────────────
    print("\n[Đang nạp model...]")
    audio_model, audio_classes = load_audio_model()
    face_model = None
    if cfg.FACE_ENABLED:
        face_model = load_face_model()

    # Nạp head weights cá nhân hóa + khởi tạo buffer on-device
    feedback_buffer = None
    if profile is not None and cfg.INCREMENTAL_ENABLED:
        freeze_backbone(audio_model)   # đóng băng backbone, chỉ head trainable
        load_user_head(audio_model, profile.user_id)
        feedback_buffer = FeedbackBuffer(profile.user_id, audio_classes)

    print("\n[Tất cả model sẵn sàng!]\n")

    # ── Vòng lặp chính ───────────────────────────────────────────────────
    while True:
        print("\n" + "─" * 55)

        # Bước 3a: Nhận diện giọng nói — trả luôn features để có thể lưu feedback
        raw_audio = audio.record()
        features  = audio.preprocess(raw_audio)   # (N_FRAMES, H, W, 1)

        import numpy as np
        inp    = np.expand_dims(features, axis=0)
        preds  = audio_model.predict(inp, verbose=0)[0]
        v_idx  = int(np.argmax(preds))
        v_emotion = audio_classes[v_idx]
        v_conf    = float(preds[v_idx]) * 100
        print(f"  🎙️  Giọng nói : {v_emotion.upper()} ({v_conf:.1f}%)")

        # Bước 3b: Nhận diện khuôn mặt (nếu bật)
        f_emotion, f_conf = None, None
        if cfg.FACE_ENABLED and face_model is not None:
            f_emotion, f_conf = face.detect(face_model, cfg.FACE_RAW_CLASSES)
            if f_emotion:
                print(f"  📷  Khuôn mặt : {f_emotion.upper()} ({f_conf:.1f}%)")
            else:
                print("  📷  Khuôn mặt : Không phát hiện — bỏ qua")

        # Bước 4: Kết hợp & xử lý xung đột
        result = fuse(v_emotion, v_conf, f_emotion, f_conf)
        result, clarify_question, clarify_answer = _clarify_conflict(result)

        # Bước 5: Tải lịch sử & chuẩn bị session log
        recent_sessions = []
        session         = None
        if profile is not None:
            recent_sessions = load_recent(profile.user_id)
            session         = SessionLog.from_fusion(result)
            if clarify_question:
                session.add_turn("assistant", clarify_question)
            if clarify_answer:
                session.add_turn("user", clarify_answer)

        # Bước 6: Chat theo kịch bản
        chat.run(result, profile, session, recent_sessions)

        # Bước 7: Lưu lịch sử + cập nhật hồ sơ
        if profile is not None and session is not None:
            session.save(profile.user_id)
            profile.record_session(result.final_emotion, result.conflict_key)
            profile.save()
            print(f"\n  💾  Đã lưu phiên vào hồ sơ '{profile.display_name}'")

        # Bước 8: Hỏi feedback + cập nhật buffer on-device
        if profile is not None:
            _ask_emotion_feedback(profile, v_emotion, features, feedback_buffer)

        # Bước 9: Trigger incremental update nếu buffer đủ mẫu
        if (
            feedback_buffer is not None
            and cfg.INCREMENTAL_ENABLED
            and feedback_buffer.ready()
        ):
            audio_model = run_incremental_update(audio_model, feedback_buffer, profile.user_id)

        # Tiếp tục?
        again = input("\n🔄  Bạn muốn thử lại không? (y/n): ").strip().lower()
        if again not in {"y", "yes", "có", "co"}:
            print("Hẹn gặp lại! 👋")
            break


if __name__ == "__main__":
    run()
