"""
history.py
==========
Lưu lịch sử hội thoại theo từng phiên.

Cấu trúc file: user_data/{user_id}/history.jsonl
  Mỗi dòng = 1 phiên (JSON):
  {
    "session_id": "...",
    "timestamp":  "...",
    "voice_emotion": "buon",
    "face_emotion":  "gian",      # "" nếu không có
    "final_emotion": "gian",
    "is_conflict":   true,
    "conflict_key":  "buon_vs_gian",
    "turns": [
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .. import config as cfg
from ..fusion import FusionResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_dir(user_id: str) -> Path:
    d = cfg.USER_DATA_DIR / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class SessionLog:
    session_id:    str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp:     str = field(default_factory=_now_iso)
    voice_emotion: str = ""
    face_emotion:  str = ""
    final_emotion: str = ""
    is_conflict:   bool = False
    conflict_key:  str = ""
    turns:         List[dict] = field(default_factory=list)   # {"role", "content"}

    @classmethod
    def from_fusion(cls, result: FusionResult) -> "SessionLog":
        return cls(
            voice_emotion=result.voice_emotion,
            face_emotion=result.face_emotion,
            final_emotion=result.final_emotion,
            is_conflict=result.is_conflict,
            conflict_key=result.conflict_key,
        )

    def add_turn(self, role: str, content: str, **metadata) -> None:
        """
        Thêm 1 lượt hội thoại.
        Chỉ giữ MAX_HISTORY_TURNS lượt gần nhất để tiết kiệm dung lượng.
        """
        item = {"role": role, "content": content}
        item.update(metadata)
        self.turns.append(item)
        cap = cfg.MAX_HISTORY_TURNS * 2   # mỗi "lượt" = 1 user + 1 assistant
        if len(self.turns) > cap:
            self.turns = self.turns[-cap:]

    def save(self, user_id: str) -> None:
        """Ghi tiếp 1 dòng JSON vào history.jsonl (append-only, không ghi đè)."""
        path = _user_dir(user_id) / "history.jsonl"
        line = json.dumps(asdict(self), ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_recent(user_id: str, n: int = None) -> list[SessionLog]:
    """
    Đọc N phiên gần nhất của người dùng.
    n=None → dùng HISTORY_CONTEXT_SIZE trong config.
    """
    if n is None:
        n = cfg.HISTORY_CONTEXT_SIZE
    path = _user_dir(user_id) / "history.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    recent = lines[-n:] if len(lines) >= n else lines
    sessions = []
    for line in recent:
        try:
            data = json.loads(line)
            sessions.append(SessionLog(**data))
        except Exception:
            pass   # bỏ qua dòng bị lỗi
    return sessions


def load_all(user_id: str) -> list[SessionLog]:
    """Đọc toàn bộ lịch sử — dùng cho export fine-tune."""
    path = _user_dir(user_id) / "history.jsonl"
    if not path.exists():
        return []
    sessions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            sessions.append(SessionLog(**json.loads(line)))
        except Exception:
            pass
    return sessions
