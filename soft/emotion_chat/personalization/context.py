"""
context.py
==========
Xây dựng đoạn context cá nhân hóa để GHÉP vào system prompt của LLM.

Đoạn này nằm TRƯỚC phần system prompt cảm xúc, giúp LLM biết:
  - Tên người dùng
  - Xu hướng cảm xúc theo thời gian
  - Những gì đã nói trong các phiên gần đây
  - Pattern xung đột lặp lại (nếu có)
"""
from __future__ import annotations

from .profile import UserProfile
from .history import SessionLog
from .long_term import load_long_term_strategy
from .summary import load_latest_summaries
from .. import config as cfg


def build(profile: UserProfile, recent_sessions: list[SessionLog]) -> str:
    """
    Build a string representing the personalized context for the user.
    This string will be prepended to the system prompt in chat.py.
    """
    if not cfg.PERSONALIZATION_ENABLED:
        return ""

    context_lines = []

    # User is new and has no history
    if profile.total_sessions == 0 and not recent_sessions:
        context_lines.extend([
            "[Người dùng mới]",
            f"Đây là lần đầu {profile.display_name} sử dụng hệ thống.",
            "Hãy giới thiệu bản thân ngắn gọn bằng tiếng Việt và hỏi một câu mở.",
        ])
    else:
        context_lines.append(f"[Thông tin người dùng] - Tên: {profile.display_name}")

        # Overall sentiment pattern
        if profile.total_emotions > 0:
            dominant_emotion = profile.dominant_emotion
            dominant_emotion_count = profile.emotion_counts.get(
                dominant_emotion, 0
            )
            context_lines.append(
                f"Xu hướng cảm xúc chung: {dominant_emotion.upper()} "
                f"({dominant_emotion_count}/{profile.total_emotions} lần)"
            )

        # Common conflict pattern
        if profile.conflict_counts:
            top_conflict = max(profile.conflict_counts, key=profile.conflict_counts.get)
            context_lines.append(
                f"Mẫu xung đột thường gặp: {top_conflict.replace('_vs_', ' ↔ ')} "
                f"({profile.conflict_counts[top_conflict]} lần)"
            )

        preferred_styles = profile.preferred_styles()
        avoided_styles = profile.styles_to_avoid()
        if preferred_styles or avoided_styles:
            context_lines.append("[Phong cách giao tiếp đã học]")
            if preferred_styles:
                context_lines.append("Nên dùng nhiều hơn: " + ", ".join(preferred_styles))
            if avoided_styles:
                context_lines.append("Nên tránh hoặc giảm: " + ", ".join(avoided_styles))

        # Recent session summary
        if recent_sessions:
            context_lines.append(f"Tóm tắt {len(recent_sessions)} phiên gần đây:")
            for i, session in enumerate(recent_sessions, 1):
                conflict_note = ""
                if session.is_conflict:
                    conflict_note = (
                        f" [CONFLICT: voice={session.voice_emotion} vs face={session.face_emotion}]"
                    )
                # Get the last user's turn content to recall the topic
                user_turns = [turn["content"] for turn in session.turns if turn["role"] == "user"]
                last_user = f" — chủ đề: {user_turns[-1][:60]}..." if user_turns else ""
                context_lines.append(
                    f"  Session {i}: {session.final_emotion.upper()}{conflict_note}{last_user}"
                )

        summary_memory = load_latest_summaries(profile.user_id)
        if summary_memory:
            context_lines.append(summary_memory)

        long_term_strategy = load_long_term_strategy(profile.user_id)
        if long_term_strategy:
            context_lines.append(long_term_strategy)

    # Add admin notes
    if profile.notes.strip():
        context_lines.append(f"Ghi chú quản trị: {profile.notes}")

    context_lines.append(
        "\nDựa trên thông tin trên, hãy điều chỉnh phản hồi cho hợp với tính cách và xu hướng cảm xúc của người dùng. Luôn trả lời bằng tiếng Việt tự nhiên."
    )

    return "\n".join(context_lines) + "\n"
