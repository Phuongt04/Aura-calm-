"""
fusion.py
=========
Kết hợp cảm xúc từ 2 nguồn (giọng nói + khuôn mặt).

Quy tắc ưu tiên:
  1. Nếu chỉ có 1 nguồn  → dùng nguồn đó.
  2. Nếu 2 nguồn đồng ý  → dùng cảm xúc đó (không xung đột).
  3. Nếu 2 nguồn mâu thuẫn → chọn cảm xúc có NEGATIVITY_WEIGHT cao hơn.
     (Bằng nhau → ưu tiên giọng nói làm tie-breaker.)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from . import config as cfg


@dataclass
class FusionResult:
    """Kết quả sau khi kết hợp hai nguồn cảm xúc."""
    final_emotion: str            # cảm xúc cuối dùng để chọn kịch bản chat
    voice_emotion: str
    face_emotion:  str            # "" nếu không có dữ liệu mặt
    voice_conf:    float
    face_conf:     float          # 0.0 nếu không có dữ liệu mặt
    is_conflict:   bool = False
    conflict_key:  str  = ""      # key tra CONFLICT_CONFIG; "" nếu không xung đột

    resolution_source: str = "fusion"
    resolution_note:   str = ""
    top2_emotion:      str = ""
    top2_conf:         float = 0.0
    confidence_margin: float = 0.0
    window_agreement:  float = 1.0
    is_uncertain:      bool = False

    def summary(self) -> str:
        """Chuỗi tóm tắt để in ra console."""
        if not self.is_conflict or not self.face_emotion:
            ec = cfg.EMOTION_CONFIG.get(self.final_emotion, {})
            lines = [
                f"  Cảm xúc   : {ec.get('display', self.final_emotion.upper())}",
                f"  Giọng nói : {self.voice_emotion.upper()} ({self.voice_conf:.1f}%)",
            ]
            if self.face_emotion:
                lines.append(
                    f"  Khuôn mặt : {self.face_emotion.upper()} ({self.face_conf:.1f}%)"
                )
            if self.resolution_note:
                lines.append(f"  Ghi chú   : {self.resolution_note}")
            if self.top2_emotion:
                lines.append(
                    f"  Top-2     : {self.top2_emotion.upper()} "
                    f"({self.top2_conf:.1f}%), margin={self.confidence_margin:.1f}, "
                    f"agreement={self.window_agreement:.2f}"
                )
            return "\n".join(lines)

        # Có xung đột
        cc = cfg.CONFLICT_CONFIG.get(self.conflict_key, {})
        if self.resolution_source == "user_clarified":
            note = self.resolution_note or "da hoi lai nguoi dung"
            return "\n".join([
                f"  [XUNG DOT] : {cc.get('display', self.conflict_key)}",
                f"  Giong noi  : {self.voice_emotion.upper()} ({self.voice_conf:.1f}%)",
                f"  Khuon mat  : {self.face_emotion.upper()} ({self.face_conf:.1f}%)",
                f"  -> Sau khi hoi lai: {self.final_emotion.upper()} ({note})",
            ])
        return "\n".join([
            f"  ⚡ XUNG ĐỘT : {cc.get('display', self.conflict_key)}",
            f"  Giọng nói  : {self.voice_emotion.upper()} ({self.voice_conf:.1f}%)",
            f"  Khuôn mặt  : {self.face_emotion.upper()} ({self.face_conf:.1f}%)",
            f"  → Ưu tiên  : {self.final_emotion.upper()}  "
            f"(tiêu cực hơn — weight={cfg.NEGATIVITY_WEIGHT.get(self.final_emotion, '?')})",
        ])


def fuse(
    voice_emotion: str,
    voice_conf:    float,
    face_emotion:  str | None  = None,
    face_conf:     float | None = None,
) -> FusionResult:
    """
    Kết hợp giọng nói và khuôn mặt thành 1 FusionResult duy nhất.

    Tham số:
        voice_emotion  — nhãn từ model âm thanh (ví dụ: "vui", "gian")
        voice_conf     — độ tin cậy % từ model âm thanh
        face_emotion   — nhãn từ model mặt (None nếu không có / không phát hiện mặt)
        face_conf      — độ tin cậy % từ model mặt (None nếu không có)
    """
    v = voice_emotion.lower()

    # ── Trường hợp 1: Chỉ có giọng nói ─────────────────────────────────────
    if face_emotion is None:
        return FusionResult(
            final_emotion=v,
            voice_emotion=v,
            face_emotion="",
            voice_conf=voice_conf,
            face_conf=0.0,
        )

    f = face_emotion.lower()

    # ── Trường hợp 2: Hai nguồn đồng ý ──────────────────────────────────────
    if v == f:
        return FusionResult(
            final_emotion=v,
            voice_emotion=v,
            face_emotion=f,
            voice_conf=voice_conf,
            face_conf=face_conf or 0.0,
        )

    # ── Trường hợp 3: Xung đột → ưu tiên cảm xúc tiêu cực hơn ──────────────
    v_weight = cfg.NEGATIVITY_WEIGHT.get(v, 0)
    f_weight = cfg.NEGATIVITY_WEIGHT.get(f, 0)

    # Bằng nhau → giữ giọng nói (nguồn đáng tin cậy hơn về ngữ nghĩa)
    final = v if v_weight >= f_weight else f

    # Tìm key xung đột (thử cả chiều thuận và chiều ngược)
    key_forward  = f"{v}_vs_{f}"
    key_backward = f"{f}_vs_{v}"
    if key_forward in cfg.CONFLICT_CONFIG:
        conflict_key = key_forward
    elif key_backward in cfg.CONFLICT_CONFIG:
        conflict_key = key_backward
    else:
        conflict_key = key_forward   # fallback: dùng key forward dù chưa có kịch bản

    return FusionResult(
        final_emotion=final,
        voice_emotion=v,
        face_emotion=f,
        voice_conf=voice_conf,
        face_conf=face_conf or 0.0,
        is_conflict=True,
        conflict_key=conflict_key,
    )


def build_clarifying_question(result: FusionResult) -> str:
    """Return a short question used when voice and face conflict."""
    if not result.is_conflict:
        return ""

    if result.conflict_key in cfg.CLARIFY_QUESTIONS:
        return cfg.CLARIFY_QUESTIONS[result.conflict_key]

    reverse_key = f"{result.face_emotion}_vs_{result.voice_emotion}"
    if reverse_key in cfg.CLARIFY_QUESTIONS:
        return cfg.CLARIFY_QUESTIONS[reverse_key]

    return cfg.CLARIFY_FALLBACK_QUESTION


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    normalized = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _score_reply(normalized_reply: str) -> dict[str, int]:
    scores = {emotion: 0 for emotion in cfg.EMOTION_CONFIG.keys()}
    if not normalized_reply:
        return scores

    padded = f" {normalized_reply} "
    for emotion, phrases in cfg.CLARIFY_REPLY_KEYWORDS.items():
        for phrase in phrases:
            if f" {phrase} " in padded:
                scores[emotion] += 2
            elif phrase in normalized_reply:
                scores[emotion] += 1
    return scores


def infer_emotion_from_reply(
    user_reply: str,
    result: FusionResult,
) -> tuple[str, str]:
    """Map a short clarification reply back to one of the 4 labels."""
    normalized = _normalize_text(user_reply)
    scores = _score_reply(normalized)

    if not normalized or max(scores.values(), default=0) <= 0:
        return result.final_emotion, "giu du doan hop nhat ban dau"

    best_score = max(scores.values())
    candidates = [emo for emo, score in scores.items() if score == best_score]

    if len(candidates) == 1:
        return candidates[0], "suy ra tu cau tra loi ngan cua nguoi dung"

    if result.face_conf > result.voice_conf and result.face_emotion in candidates:
        return result.face_emotion, "tra loi mo ho, uu tien kenh mat co do tin cay cao hon"

    if result.voice_emotion in candidates:
        return result.voice_emotion, "tra loi mo ho, giu theo giong noi"

    if result.final_emotion in candidates:
        return result.final_emotion, "tra loi mo ho, giu ket qua hop nhat ban dau"

    return candidates[0], "tra loi co nhieu tin hieu ngang nhau"


def apply_clarified_emotion(
    result: FusionResult,
    final_emotion: str,
    note: str,
) -> FusionResult:
    """Return a new FusionResult after asking the user to clarify."""
    return replace(
        result,
        final_emotion=final_emotion,
        resolution_source="user_clarified",
        resolution_note=note,
    )
