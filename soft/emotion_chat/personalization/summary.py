"""
summary.py
==========
Build daily/weekly text memory from saved user conversation history.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from .. import config as cfg
from .history import SessionLog, load_all


def _user_summary_dir(user_id: str) -> Path:
    path = cfg.USER_DATA_DIR / user_id / "summaries"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_local_date(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    return datetime.fromisoformat(text).astimezone()


def _period_key(dt: datetime, period: str) -> str:
    if period == "weekly":
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02d}"
    return dt.strftime("%Y-%m-%d")


def _summary_path(user_id: str, period: str, key: str) -> Path:
    return _user_summary_dir(user_id) / f"{period}_{key}.txt"


def _session_topic(session: SessionLog) -> str:
    user_turns = [
        turn.get("content", "").strip()
        for turn in session.turns
        if turn.get("role") == "user" and turn.get("content", "").strip()
    ]
    if not user_turns:
        return ""
    topic = user_turns[-1].replace("\n", " ")
    return topic[:180]


def _build_summary_text(
    user_id: str,
    period: str,
    key: str,
    sessions: list[SessionLog],
) -> str:
    emotion_counts = Counter(s.final_emotion for s in sessions if s.final_emotion)
    conflict_counts = Counter(s.conflict_key for s in sessions if s.conflict_key)
    topics = [_session_topic(session) for session in sessions]
    topics = [topic for topic in topics if topic][:cfg.SUMMARY_MAX_TOPICS]

    lines = [
        f"User: {user_id}",
        f"Period: {period} {key}",
        f"Sessions: {len(sessions)}",
        "",
        "Emotion counts:",
    ]

    if emotion_counts:
        for label, count in emotion_counts.most_common():
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "Conflict patterns:"])
    if conflict_counts:
        for label, count in conflict_counts.most_common():
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "Recent user topics:"])
    if topics:
        for topic in topics:
            lines.append(f"- {topic}")
    else:
        lines.append("- none")

    dominant = emotion_counts.most_common(1)[0][0] if emotion_counts else "unknown"
    lines.extend([
        "",
        "Personalization notes:",
        f"- Dominant emotion in this period: {dominant}",
        "- Use this as memory, not as a fixed diagnosis.",
    ])

    return "\n".join(lines).strip() + "\n"


def update_periodic_summaries(user_id: str) -> list[Path]:
    """Update today's daily summary and this week's weekly summary."""
    if not cfg.SUMMARY_ENABLED:
        return []

    sessions = load_all(user_id)
    if not sessions:
        return []

    now = datetime.now().astimezone()
    written: list[Path] = []

    for period in ("daily", "weekly"):
        key = _period_key(now, period)
        selected: list[SessionLog] = []
        for session in sessions:
            try:
                session_dt = _parse_local_date(session.timestamp)
            except Exception:
                continue
            if _period_key(session_dt, period) == key:
                selected.append(session)

        if not selected:
            continue

        path = _summary_path(user_id, period, key)
        path.write_text(
            _build_summary_text(user_id, period, key, selected),
            encoding="utf-8",
        )
        written.append(path)

    return written


def load_latest_summaries(user_id: str) -> str:
    """Load latest daily and weekly text summaries for prompt memory."""
    if not cfg.SUMMARY_ENABLED:
        return ""

    summary_dir = _user_summary_dir(user_id)
    parts = []
    for pattern, title in (("daily_*.txt", "Daily Memory"), ("weekly_*.txt", "Weekly Memory")):
        files = sorted(summary_dir.glob(pattern))
        if not files:
            continue
        text = files[-1].read_text(encoding="utf-8").strip()
        if text:
            parts.append(f"[{title}]\n{text}")

    if not parts:
        return ""

    text = "\n\n".join(parts)
    return text[-cfg.SUMMARY_CONTEXT_CHARS:]
