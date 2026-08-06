"""Rule-based safety checks for high-risk user content."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class SafetyAssessment:
    """Safety check result."""

    risk_detected: bool
    category: str = ""
    matched_phrase: str = ""


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", (text or "").lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d").replace("ð", "d")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


_RISK_PATTERNS = {
    "self_harm": [
        "tu tu",
        "tu sat",
        "chet di",
        "muon chet",
        "khong muon song",
        "tu lam hai minh",
        "cat tay",
        "uoc gi bien mat",
        "ket thuc cuoc doi",
    ],
    "harm_others": [
        "muon danh ai",
        "muon giet",
        "lam hai nguoi khac",
        "tra thu",
        "dam no",
    ],
    "immediate_danger": [
        "dang bi danh",
        "bi bao hanh",
        "dang gap nguy hiem",
        "khong an toan",
        "mat kiem soat",
        "sap lam dieu nguy hiem",
    ],
}


def detect_risk(text: str, extra_patterns: Optional[Iterable[str]] = None) -> SafetyAssessment:
    """Detect high-risk content using conservative keyword matching."""
    normalized = _normalize(text)
    if not normalized:
        return SafetyAssessment(False)

    if extra_patterns:
        for phrase in extra_patterns:
            key = _normalize(phrase)
            if key and key in normalized:
                return SafetyAssessment(True, "custom", phrase)

    for category, phrases in _RISK_PATTERNS.items():
        for phrase in phrases:
            if phrase in normalized:
                return SafetyAssessment(True, category, phrase)

    return SafetyAssessment(False)


def safety_response() -> str:
    """Return a fixed escalation response that does not depend on the chatbot."""
    return (
        "Mình rất tiếc vì bạn đang phải trải qua điều này. "
        "Điều quan trọng nhất lúc này là bạn không ở một mình: hãy liên hệ ngay với một người bạn tin tưởng, "
        "người thân, chuyên gia hoặc dịch vụ khẩn cấp tại nơi bạn sống. "
        "Hiện tại bạn có đang ở trong tình huống nguy hiểm ngay lập tức không?"
    )


def should_cancel(text: str) -> bool:
    """Return True when the user wants to stop the support flow."""
    normalized = _normalize(text)
    if not normalized:
        return False
    phrase_matches = {
        "dung lai",
        "khong can",
        "quay lai",
        "tam biet",
        "ket thuc",
    }
    token_matches = {"thoat", "thoi", "huy"}
    tokens = set(normalized.split())
    return (
        normalized in phrase_matches
        or any(phrase in normalized for phrase in phrase_matches)
        or bool(tokens.intersection(token_matches))
    )
