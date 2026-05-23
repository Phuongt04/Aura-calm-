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

from . import config as cfg
from . import voice_io
from .fusion import FusionResult
from .personalization import context as ctx_builder
from .personalization.history import SessionLog
from .personalization.profile import UserProfile


JOKES = [
    "Co mot con robot di thi. Thay hoi: em co quay bai khong? Robot noi: khong thay, em chi dang backup kien thuc.",
    "May tinh hoi nguoi dung: sao hom nay ban buon? Nguoi dung noi: vi deadline. May tinh dap: toi cung vay, luc nao cung bi chay deadline.",
    "MONE vua hoc nau an. Mon dau tien la mi goi, nhung van can cap nhat firmware de biet do nuoc soi.",
    "Co nguoi hoi AI: ban co biet yeu khong? AI dap: biet chu, nhung hien tai toi yeu cau nguon dien on dinh hon.",
]


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
    assistant_instruction = (
        f"Ban la {cfg.VOICE_ASSISTANT_NAME}, tro ly ao tieng Viet noi chuyen tu nhien. "
        "Tra loi ngan gon, than thien, moi luot 1-3 cau. "
        "Tin hieu cam xuc ban dau chi la goi y, khong phai ket luan co dinh. "
        f"Nhan cam xuc ban dau: {result.final_emotion}. "
        "Neu noi dung nguoi dung ve sau khac voi nhan ban dau, hay doi cach phan hoi theo noi dung moi. "
        "Khong lap lai cung mot cau hoi hoac cung mot cau an ui."
    )

    if result.is_conflict and result.conflict_key:
        conflict_cfg = cfg.CONFLICT_CONFIG.get(result.conflict_key)
        base_cfg = cfg.EMOTION_CONFIG.get(result.final_emotion, {})
        opening = (conflict_cfg or {}).get("opening") or base_cfg.get("opening", cfg.DEFAULT_OPENING)
    else:
        emotion_cfg = cfg.EMOTION_CONFIG.get(result.final_emotion, {})
        opening = emotion_cfg.get("opening", cfg.DEFAULT_OPENING)

    personal_block = ""
    if profile is not None and recent_sessions is not None:
        personal_block = ctx_builder.build(profile, recent_sessions)

    return personal_block + assistant_instruction, opening


def _make_llm_client():
    if cfg.CHAT_BACKEND.lower() == "rule":
        return None

    from openai import OpenAI

    return OpenAI(api_key="ollama", base_url=cfg.OLLAMA_BASE_URL)


def _respond(client, history: list[dict], emotion: str, user_input: str) -> str:
    if client is None:
        return _rule_reply(emotion, user_input)

    response = client.chat.completions.create(
        model=cfg.OLLAMA_MODEL,
        messages=history,
        temperature=cfg.OLLAMA_TEMPERATURE,
        max_tokens=cfg.OLLAMA_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def run(
    result: FusionResult,
    profile: UserProfile | None = None,
    session: SessionLog | None = None,
    recent_sessions: list | None = None,
) -> None:
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

    print(f"\n🤖  {opening}\n")
    voice_io.speak(opening)

    if session is not None:
        session.add_turn("assistant", opening)

    try:
        client = _make_llm_client()
    except Exception as exc:
        print(f"  [chat] Khong khoi tao duoc LLM: {exc}")
        if cfg.CHAT_BACKEND.lower() != "rule":
            print("  -> Dang yeu cau backend AI that, nen khong tu dong roi ve rule.")
            return
        client = None

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": opening},
    ]
    empty_count = 0

    while True:
        if cfg.VOICE_CHAT_ENABLED:
            user_input = voice_io.listen_once().strip()
        else:
            user_input = input("👤  Bạn: ").strip()

        if not user_input:
            empty_count += 1
            if cfg.VOICE_CHAT_ENABLED and empty_count >= cfg.VOICE_CHAT_MAX_EMPTY:
                retry = "Mình chưa nghe rõ. Bạn nói lại giúp mình nhé."
                print(f"\n🤖  {retry}\n")
                voice_io.speak(retry)
                empty_count = 0
            continue
        empty_count = 0

        if _should_exit(user_input):
            goodbye = "Tạm biệt bạn nhé. Khi nào cần thì cứ gọi mình."
            print(f"\n🤖  {goodbye}\n")
            voice_io.speak(goodbye)
            if session is not None:
                session.add_turn("user", user_input)
                session.add_turn("assistant", goodbye)
            break

        history.append({"role": "user", "content": user_input})
        if session is not None:
            session.add_turn("user", user_input)

        entertainment = _entertainment_reply(user_input)
        if entertainment is not None:
            history.append({"role": "assistant", "content": entertainment})
            if session is not None:
                session.add_turn("assistant", entertainment)
            print(f"\nðŸ¤–  {entertainment}\n")
            voice_io.speak(entertainment)
            continue

        try:
            reply = _respond(client, history, result.final_emotion, user_input)
        except Exception as exc:
            print(f"\n  [Loi chat backend]: {exc}")
            if cfg.CHAT_BACKEND.lower() != "rule":
                print("  -> Backend AI that dang loi, khong tu dong roi ve rule.")
                break
            print("  -> Chuyen tam sang phan hoi rule.")
            client = None
            reply = _rule_reply(result.final_emotion, user_input)

        history.append({"role": "assistant", "content": reply})
        if session is not None:
            session.add_turn("assistant", reply)

        print(f"\n🤖  {reply}\n")
        voice_io.speak(reply)
