import re
import unicodedata
from pathlib import Path


LABELS = ["neutral", "happy", "sad", "angry"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}

VI_TO_STANDARD = {
    "binhthuong": "neutral",
    "binh thuong": "neutral",
    "binh_thuong": "neutral",
    "binh-thuong": "neutral",
    "trungtinh": "neutral",
    "trung tinh": "neutral",
    "trung_tinh": "neutral",
    "normal": "neutral",
    "neutral": "neutral",
    "vui": "happy",
    "vui ve": "happy",
    "vui_ve": "happy",
    "happy": "happy",
    "happiness": "happy",
    "buon": "sad",
    "sad": "sad",
    "sadness": "sad",
    "gian": "angry",
    "tucgian": "angry",
    "tuc gian": "angry",
    "tuc_gian": "angry",
    "angry": "angry",
    "anger": "angry",
}

EXCLUDED_LABELS = set([
    "fear",
    "fearful",
    "disgust",
    "surprise",
    "surprised",
    "calm",
    "boredom",
])


def normalize_text(value):
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def standardize_label(raw_label):
    if raw_label is None:
        return None
    norm = normalize_text(raw_label)
    if norm in VI_TO_STANDARD:
        return VI_TO_STANDARD[norm]
    compact = norm.replace(" ", "")
    if compact in VI_TO_STANDARD:
        return VI_TO_STANDARD[compact]
    if compact in EXCLUDED_LABELS or norm in EXCLUDED_LABELS:
        return None
    return None


def infer_label_from_path(path, data_root=None):
    path = Path(path)
    candidates = []
    if data_root is not None:
        try:
            rel_parts = path.relative_to(Path(data_root)).parts
            candidates.extend(rel_parts[:-1])
        except ValueError:
            pass
    candidates.extend(path.parts[:-1])
    candidates.append(path.stem)

    for candidate in reversed(candidates):
        label = standardize_label(candidate)
        if label:
            return label

    stem_norm = normalize_text(path.stem)
    tokens = re.split(r"[\s_.\-()]+", stem_norm)
    for token in tokens:
        label = standardize_label(token)
        if label:
            return label
    return None


def infer_source_from_path(path):
    text = normalize_text(str(path))
    name = Path(path).name.upper()
    if "tess" in text or name.startswith("OAF_") or name.startswith("YAF_"):
        return "tess"
    if "soft archive" in text or "soft/archive" in text or "soft\\archive" in str(path).lower():
        return "vietnamese_kaggle"
    if "kaggle" in text or "vietnamese speech emotion" in text or "vser" in text:
        return "vietnamese_kaggle"
    if "self" in text or "recorded" in text or "tu thu" in text or "tu_thu" in text:
        return "self_recorded"
    return "unknown"


def infer_original_id(path):
    stem = normalize_text(Path(path).stem)
    stem = re.sub(r"\b(aug|augment|augmentation|copy|noise|pitch|shift|stretch|gain)\b.*$", "", stem)
    stem = re.sub(r"(_|-|\s)+(aug|noise|pitch|shift|stretch|gain)\d*$", "", stem)
    stem = re.sub(r"\s+", "_", stem).strip("_")
    return stem or Path(path).stem
