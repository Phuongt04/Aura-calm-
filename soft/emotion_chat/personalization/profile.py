"""
profile.py
==========
Hồ sơ người dùng — lưu thống kê cảm xúc theo thời gian.

Cấu trúc file: user_data/{user_id}/profile.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from .. import config as cfg


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_dir(user_id: str) -> Path:
    d = cfg.USER_DATA_DIR / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class UserProfile:
    user_id:          str
    display_name:     str
    created_at:       str = field(default_factory=_now_iso)
    last_seen:        str = field(default_factory=_now_iso)
    total_sessions:   int = 0

    # Thống kê cảm xúc: {"vui": 5, "buon": 3, ...}
    emotion_counts:   dict = field(default_factory=dict)

    # Thống kê xung đột: {"vui_vs_buon": 2, ...}
    conflict_counts:  dict = field(default_factory=dict)

    # Ghi chú tự do (có thể chỉnh tay để bổ sung ngữ cảnh)
    notes:            str = ""

    # Learned conversation style preferences.
    communication_preferences: dict = field(default_factory=dict)
    avoided_styles:            dict = field(default_factory=dict)

    # ── Thuộc tính tính toán ──────────────────────────────────────────────
    @property
    def dominant_emotion(self) -> str:
        """Cảm xúc xuất hiện nhiều nhất."""
        if not self.emotion_counts:
            return "binhthuong"
        return max(self.emotion_counts, key=self.emotion_counts.get)

    @property
    def total_emotions(self) -> int:
        return sum(self.emotion_counts.values())

    # ── Cập nhật sau mỗi phiên ───────────────────────────────────────────
    def record_session(self, final_emotion: str, conflict_key: str = "") -> None:
        self.total_sessions += 1
        self.last_seen = _now_iso()
        self.emotion_counts[final_emotion] = self.emotion_counts.get(final_emotion, 0) + 1
        if conflict_key:
            self.conflict_counts[conflict_key] = self.conflict_counts.get(conflict_key, 0) + 1

    def record_style_feedback(self, preferred: str = "", avoid: str = "") -> None:
        """Accumulate what response style works or does not work for this user."""
        preferred = preferred.strip()
        avoid = avoid.strip()
        if preferred:
            self.communication_preferences[preferred] = (
                self.communication_preferences.get(preferred, 0) + 1
            )
        if avoid:
            self.avoided_styles[avoid] = self.avoided_styles.get(avoid, 0) + 1

    def preferred_styles(self, limit: int = 3) -> list[str]:
        return [
            name for name, _ in sorted(
                self.communication_preferences.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
        ]

    def styles_to_avoid(self, limit: int = 2) -> list[str]:
        return [
            name for name, _ in sorted(
                self.avoided_styles.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
        ]

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self) -> None:
        path = _user_dir(self.user_id) / "profile.json"
        # Ghi tạm rồi rename để tránh corrupt khi mất điện
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, user_id: str) -> "UserProfile":
        path = _user_dir(user_id) / "profile.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    @classmethod
    def load_or_create(cls, user_id: str, display_name: str = "") -> "UserProfile":
        existing = cls.load(user_id)
        if existing is not None:
            return existing
        profile = cls(user_id=user_id, display_name=display_name or user_id)
        profile.save()
        print(f"  [profile] Tạo hồ sơ mới cho '{display_name or user_id}'")
        return profile

    def summary(self) -> str:
        """Chuỗi tóm tắt ngắn để in ra console."""
        summary = f"  Người dùng : {self.display_name}  ({self.user_id})"
        if self.total_sessions:
            dominant_emotion = self.dominant_emotion
            summary += f"\n  Cảm xúc chủ đạo : {dominant_emotion.upper()}  " \
                              f"({self.emotion_counts.get(dominant_emotion, 0)}/{self.total_emotions} lần)"
            summary += f"\n  Tổng phiên : {self.total_sessions}"
        else:
            summary += "\n  → Người dùng mới, chưa có lịch sử"
        return summary

    def summary_str(self) -> str:
        """Backward-compatible name used by the chat loop."""
        return self.summary()
