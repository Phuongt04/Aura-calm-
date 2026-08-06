"""Small Vietnamese text-sentiment rules used as a support signal."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TextSentimentResult:
    """Result of a lightweight text sentiment pass."""

    emotion: str = ""
    score: int = 0
    matched: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.emotion and self.score > 0)


def normalize_text(text: str) -> str:
    """Normalize Vietnamese text for simple phrase matching."""
    value = unicodedata.normalize("NFD", (text or "").lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d").replace("ð", "d")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


TEXT_SENTIMENT_KEYWORDS: Dict[str, List[str]] = {
    "vui": [
        "vui",
        "vui qua",
        "rat vui",
        "hanh phuc",
        "thich qua",
        "tuyet voi",
        "hao hung",
        "phan khoi",
        "may qua",
        "tot qua",
        "duoc diem cao",
        "thanh cong",
    ],
    "buon": [
        "buon",
        "chan",
        "met moi",
        "ap luc",
        "that vong",
        "co don",
        "khong on",
        "nang long",
        "muon khoc",
        "te qua",
        "stress",
        "lo lang",
        "so hai",
    ],
    "gian": [
        "gian",
        "tuc",
        "buc",
        "buc minh",
        "buc boi",
        "kho chiu",
        "uc che",
        "dien tiet",
        "noi dien",
        "cay",
        "phat cau",
    ],
    "binhthuong": [
        "binh thuong",
        "khong sao",
        "van on",
        "on ma",
        "tam duoc",
        "cung duoc",
        "khong co gi",
        "hoi met",
        "chi met",
    ],
}

NEGATION_WORDS = {"khong", "chua", "chang", "ko", "k"}


def _phrase_score(normalized: str, phrase: str) -> int:
    padded = " %s " % normalized
    phrase_padded = " %s " % phrase
    if phrase_padded in padded:
        # Multi-word phrases are stronger than isolated emotional words.
        return 2 if " " in phrase else 1
    return 0


def _has_negation_before(normalized: str, phrase: str) -> bool:
    tokens = normalized.split()
    phrase_tokens = phrase.split()
    if not phrase_tokens:
        return False
    for idx in range(0, len(tokens) - len(phrase_tokens) + 1):
        if tokens[idx:idx + len(phrase_tokens)] == phrase_tokens:
            window = tokens[max(0, idx - 3):idx]
            return any(token in NEGATION_WORDS for token in window)
    return False


def infer_text_sentiment(text: str) -> TextSentimentResult:
    """Infer a coarse emotion from Vietnamese user text.

    This is intentionally conservative. It is a support signal for the chatbot,
    not a replacement for voice/face emotion recognition.
    """
    normalized = normalize_text(text)
    if not normalized:
        return TextSentimentResult()

    scores: Dict[str, int] = {emotion: 0 for emotion in TEXT_SENTIMENT_KEYWORDS}
    matched: Dict[str, List[str]] = {}
    for emotion, phrases in TEXT_SENTIMENT_KEYWORDS.items():
        for phrase in phrases:
            score = _phrase_score(normalized, phrase)
            if not score:
                continue
            if _has_negation_before(normalized, phrase):
                if emotion in {"vui", "buon", "gian"}:
                    continue
                score = max(1, score - 1)
            scores[emotion] += score
            matched.setdefault(emotion, []).append(phrase)

    best_score = max(scores.values(), default=0)
    if best_score <= 0:
        return TextSentimentResult()
    candidates = [emotion for emotion, score in scores.items() if score == best_score]

    # Prefer stronger affective labels over neutral when tied.
    for emotion in ("gian", "buon", "vui", "binhthuong"):
        if emotion in candidates:
            return TextSentimentResult(emotion, best_score, matched)
    return TextSentimentResult()
