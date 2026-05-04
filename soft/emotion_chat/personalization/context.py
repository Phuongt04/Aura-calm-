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
from .. import config as cfg


def build_context(profile: UserProfile, recent_sessions: list[SessionLog]) -> str:
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
            "[New User]",
            f"This is the first time {profile.display_name} is using the system.",
            "Please introduce yourself and ask an open-ended question.",
        ])
    else:
        context_lines.append(f"[User Information] - Name: {profile.display_name}")

        # Overall sentiment pattern
        if profile.total_emotions > 0:
            dominant_emotion = profile.dominant_emotion
            dominant_emotion_count = profile.emotion_counts.get(
                dominant_emotion, 0
            )
            context_lines.append(
                f"Overall sentiment pattern: {dominant_emotion.upper()} "
                f"({dominant_emotion_count}/{profile.total_emotions} times)"
            )

        # Common conflict pattern
        if profile.conflict_counts:
            top_conflict = max(profile.conflict_counts, key=profile.conflict_counts.get)
            context_lines.append(
                f"Common conflict pattern: {top_conflict.replace('_vs_', ' ↔ ')} "
                f"({profile.conflict_counts[top_conflict]} times)"
            )

        # Recent session summary
        if recent_sessions:
            context_lines.append(f"Recent session summary ({len(recent_sessions)} sessions):")
            for i, session in enumerate(recent_sessions, 1):
                conflict_note = ""
                if session.is_conflict:
                    conflict_note = (
                        f" [CONFLICT: voice={session.voice_emotion} vs face={session.face_emotion}]"
                    )
                # Get the last user's turn content to recall the topic
                user_turns = [turn["content"] for turn in session.turns if turn["role"] == "user"]
                last_user = f" — topic: {user_turns[-1][:60]}..." if user_turns else ""
                context_lines.append(
                    f"  Session {i}: {session.final_emotion.upper()}{conflict_note}{last_user}"
                )

    # Add admin notes
    if profile.notes.strip():
        context_lines.append(f"Admin Notes: {profile.notes}")

    context_lines.append(
        "\nBased on the information above, please adjust your responses to fit the user's personality and sentiment patterns."
    )

    return "\n".join(context_lines) + "\n"
