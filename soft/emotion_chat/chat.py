"""
chat.py
=======
Vòng trò chuyện tương tác với Ollama.
Nhận FusionResult + UserProfile + SessionLog → xây dựng system prompt cá nhân hóa
→ chat loop → lưu từng lượt hội thoại vào SessionLog.

Thứ tự ghép system prompt:
  [1] Personalized context (lịch sử người dùng)
  [2] Conflict context_hint  (nếu có xung đột)
  [3] Emotion base system    (theo cảm xúc cuối)
"""
from __future__ import annotations

from openai import OpenAI
from .fusion import FusionResult
from . import config as cfg
from .personalization.profile import UserProfile
from .personalization.history import SessionLog
from .personalization import context as ctx_builder


def _build_prompt(
    result:          FusionResult,
    profile:         UserProfile  | None,
    recent_sessions: list         | None,
) -> tuple[str, str]:
    """
    Trả về (system_prompt: str, opening_message: str).

    Layer ghép:
      personalized_context  (từ lịch sử user)
    + conflict context_hint (nếu có xung đột)
    + emotion base system
    """
    # ── Layer 1: cảm xúc / xung đột ─────────────────────────────────────
    if result.is_conflict and result.conflict_key:
        cc       = cfg.CONFLICT_CONFIG.get(result.conflict_key)
        base_ec  = cfg.EMOTION_CONFIG.get(result.final_emotion, {})
        base_sys = base_ec.get("system", cfg.DEFAULT_SYSTEM)
        if cc and cc.get("context_hint"):
            emotion_block = cc["context_hint"] + "\n\n" + base_sys
            opening       = cc.get("opening", cfg.DEFAULT_OPENING)
        else:
            emotion_block = (
                f"Lưu ý: Giọng nói biểu lộ '{result.voice_emotion}' "
                f"nhưng khuôn mặt cho thấy '{result.face_emotion}'. "
                "Hãy tiếp cận cẩn thận và đồng cảm.\n\n" + base_sys
            )
            opening = base_ec.get("opening", cfg.DEFAULT_OPENING)
    else:
        ec            = cfg.EMOTION_CONFIG.get(result.final_emotion, {})
        emotion_block = ec.get("system",  cfg.DEFAULT_SYSTEM)
        opening       = ec.get("opening", cfg.DEFAULT_OPENING)

    # ── Layer 2: cá nhân hóa (ghép trước emotion_block) ─────────────────
    personal_block = ""
    if profile is not None and recent_sessions is not None:
        personal_block = ctx_builder.build(profile, recent_sessions)

    system = personal_block + emotion_block
    return system, opening


def run(
    result:          FusionResult,
    profile:         UserProfile  | None = None,
    session:         SessionLog   | None = None,
    recent_sessions: list         | None = None,
) -> None:
    """
    Khởi chạy vòng chat. Blocking cho đến khi user thoát.

    Args:
        result          — kết quả nhận diện cảm xúc (từ fusion.py)
        profile         — hồ sơ người dùng (None = không cá nhân hóa)
        session         — SessionLog của phiên này để ghi lịch sử
        recent_sessions — danh sách phiên gần đây để xây context
    """
    system_prompt, opening = _build_prompt(result, profile, recent_sessions)

    print("\n" + "=" * 55)
    print(result.summary())
    if profile:
        print("─" * 55)
        print(profile.summary_str())
    print("=" * 55)
    print("  (gõ 'thoát' hoặc 'quit' để kết thúc)\n")
    print(f"🤖  {opening}\n")

    # Ghi lượt mở đầu của assistant vào session log
    if session is not None:
        session.add_turn("assistant", opening)

    client  = OpenAI(api_key="ollama", base_url=cfg.OLLAMA_BASE_URL)
    history = [
        {"role": "system",    "content": system_prompt},
        {"role": "assistant", "content": opening},
    ]

    while True:
        user_input = input("👤  Bạn: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "thoát", "thoat", "exit"}:
            print("\n🤖  Tạm biệt bạn nhé! Chúc bạn một ngày tốt lành! 👋")
            if session is not None:
                session.add_turn("user", user_input)
            break

        history.append({"role": "user", "content": user_input})
        if session is not None:
            session.add_turn("user", user_input)

        try:
            response = client.chat.completions.create(
                model=cfg.OLLAMA_MODEL,
                messages=history,
                temperature=cfg.OLLAMA_TEMPERATURE,
                max_tokens=cfg.OLLAMA_MAX_TOKENS,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"\n  [Lỗi Ollama]: {exc}")
            print("  → Kiểm tra Ollama đang chạy chưa: ollama serve")
            break

        history.append({"role": "assistant", "content": reply})
        if session is not None:
            session.add_turn("assistant", reply)
        print(f"\n🤖  {reply}\n")
