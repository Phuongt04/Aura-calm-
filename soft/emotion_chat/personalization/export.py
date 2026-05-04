"""
export.py
=========
Xuất dữ liệu hội thoại để phục vụ fine-tuning.

Hai loại export:
  1. LLM fine-tuning (LoRA / TinyLlama):
       → user_data/exports/finetune_llm_{user_id}.jsonl
       Định dạng ChatML, mỗi dòng = 1 cặp (user_turn, assistant_turn) với context.

  2. Emotion model feedback (thu thập nhãn cảm xúc được xác nhận bởi user):
       → user_data/{user_id}/emotion_feedback.jsonl
       Định dạng: {"label": "buon", "feature_path": "...", "confirmed_by_user": true}
       Feature (.npy) lưu trong user_data/{user_id}/features/

Cách chạy export:
    python -m emotion_chat.personalization.export --user all
    python -m emotion_chat.personalization.export --user nguyen_van_a
"""
from __future__ import annotations

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from .. import config as cfg
from .history import load_all, SessionLog


def _exports_dir() -> Path:
    d = cfg.USER_DATA_DIR / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── 1. LLM Fine-tuning export ─────────────────────────────────────────────────
def _session_to_chatml(session: SessionLog) -> list[dict]:
    """
    Chuyển 1 phiên thành danh sách mẫu ChatML.
    Mỗi mẫu = {"messages": [system, ..., user, assistant]}
    Trả về list vì 1 phiên có thể tạo nhiều mẫu (sliding window).
    """
    if not session.turns:
        return []

    # Xây dựng system hint từ cảm xúc của phiên
    emotion_hint = cfg.EMOTION_CONFIG.get(session.final_emotion, {}).get(
        "system", cfg.DEFAULT_SYSTEM
    )
    if session.is_conflict:
        cc = cfg.CONFLICT_CONFIG.get(session.conflict_key, {})
        context_hint = cc.get("context_hint", "")
        if context_hint:
            emotion_hint = context_hint + "\n\n" + emotion_hint

    samples = []
    # Sliding window: mỗi lượt user+assistant = 1 training sample
    # Thêm N lượt trước làm context nếu có
    WINDOW = 2   # số lượt trước làm context
    pairs  = [(session.turns[i], session.turns[i + 1])
              for i in range(0, len(session.turns) - 1, 2)
              if session.turns[i]["role"] == "user"]
    for idx, (u, a) in enumerate(pairs):
        context_turns = session.turns[max(0, (idx - WINDOW) * 2): idx * 2]
        messages = [{"role": "system", "content": emotion_hint}]
        messages.extend(context_turns)
        messages.append(u)
        messages.append(a)
        samples.append({"messages": messages})

    return samples


def export_llm_finetune(user_id: str = "all") -> Path:
    """
    Xuất tất cả hội thoại sang ChatML JSONL cho fine-tuning TinyLlama / LoRA.
    Trả về đường dẫn file đã tạo.
    """
    if user_id == "all":
        user_ids = [d.name for d in cfg.USER_DATA_DIR.iterdir() if d.is_dir() and d.name != "exports"]
    else:
        user_ids = [user_id]

    all_samples = []
    for uid in user_ids:
        sessions = load_all(uid)
        for s in sessions:
            all_samples.extend(_session_to_chatml(s))

    tag  = user_id.replace(" ", "_")
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = _exports_dir() / f"finetune_llm_{tag}_{ts}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[export] LLM fine-tune: {len(all_samples)} mẫu → {path}")
    return path


# ── 2. Emotion model feedback ─────────────────────────────────────────────────
def save_emotion_feedback(
    user_id:           str,
    features:          np.ndarray,
    predicted_label:   str,
    corrected_label:   str,
    confirmed_by_user: bool,
) -> None:
    """
    Lưu audio features (.npy) và nhãn đã xác nhận để re-train model cảm xúc.
    Gọi hàm này từ main.py sau khi hỏi user xác nhận cảm xúc.
    """
    feat_dir = cfg.USER_DATA_DIR / user_id / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    feat_path = feat_dir / f"{corrected_label}_{ts}.npy"
    np.save(str(feat_path), features)

    feedback_path = cfg.USER_DATA_DIR / user_id / "emotion_feedback.jsonl"
    record = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "predicted_label":   predicted_label,
        "corrected_label":   corrected_label,
        "confirmed_by_user": confirmed_by_user,
        "feature_path":      str(feat_path),
    }
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_emotion_dataset(output_dir: Path = None) -> Path:
    """
    Gom tất cả features đã xác nhận từ mọi user → thư mục dataset chuẩn.
    Cấu trúc output: output_dir/{label}/{file}.npy
    Dùng để re-train model âm thanh với dữ liệu cá nhân hóa.
    """
    if output_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = _exports_dir() / f"emotion_dataset_{ts}"

    count = 0
    for user_dir in cfg.USER_DATA_DIR.iterdir():
        if not user_dir.is_dir() or user_dir.name == "exports":
            continue
        fb_path = user_dir / "emotion_feedback.jsonl"
        if not fb_path.exists():
            continue
        for line in fb_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if not rec.get("confirmed_by_user"):
                    continue
                label     = rec["corrected_label"]
                src       = Path(rec["feature_path"])
                if not src.exists():
                    continue
                dest_dir  = output_dir / label
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest      = dest_dir / src.name
                dest.write_bytes(src.read_bytes())
                count += 1
            except Exception:
                pass

    print(f"[export] Emotion dataset: {count} mẫu đã xác nhận → {output_dir}")
    return output_dir


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export dữ liệu để fine-tuning")
    parser.add_argument("--user",   default="all", help="user_id hoặc 'all'")
    parser.add_argument("--type",   default="llm", choices=["llm", "emotion", "all"])
    args = parser.parse_args()

    if args.type in ("llm", "all"):
        export_llm_finetune(args.user)
    if args.type in ("emotion", "all"):
        export_emotion_dataset()
