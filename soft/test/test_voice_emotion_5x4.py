"""
Test nhan dien cam xuc giong noi theo bo 4 cam xuc x 5 file.

Mac dinh script se:
  - Lay du lieu tu folder data/<cam_xuc>/*.wav
  - Tu dong chon 5 mau co cung ma/cau giua 4 cam xuc neu co
  - Nap model trong emotion_chat/config.py
  - Xuat bang danh gia CSV, Markdown va JSON vao soft/test/results

Chay:
  python soft/test/test_voice_emotion_5x4.py

Neu co folder test rieng:
  python soft/test/test_voice_emotion_5x4.py --test-root D:\\path\\to\\data
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import librosa
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


THIS_FILE = Path(__file__).resolve()
SOFT_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = SOFT_DIR.parent
sys.path.insert(0, str(SOFT_DIR))

from emotion_chat import audio as audio_pipeline
from emotion_chat import config as cfg
from emotion_chat.model_loader import load_audio_model


EMOTIONS = ["binhthuong", "buon", "gian", "vui"]
EMOTION_DIR_ALIASES = {
    "binhthuong": ["binhthuong", "binh_thuong", "neutral", "normal"],
    "buon": ["buon", "sad"],
    "gian": ["gian", "angry"],
    "vui": ["vui", "happy"],
}
EMOTION_WORDS = {
    "angry",
    "binhthuong",
    "binh thuong",
    "buon",
    "bình thường",
    "buồn",
    "giận",
    "gian",
    "happy",
    "neutral",
    "sad",
    "vui",
}


def normalize_text(value: str) -> str:
    value = value.lower()
    replacements = {
        "á": "a", "à": "a", "ả": "a", "ã": "a", "ạ": "a",
        "ă": "a", "ắ": "a", "ằ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
        "â": "a", "ấ": "a", "ầ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
        "é": "e", "è": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
        "ê": "e", "ế": "e", "ề": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "í": "i", "ì": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
        "ó": "o", "ò": "o", "ỏ": "o", "õ": "o", "ọ": "o",
        "ô": "o", "ố": "o", "ồ": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
        "ơ": "o", "ớ": "o", "ờ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
        "ú": "u", "ù": "u", "ủ": "u", "ũ": "u", "ụ": "u",
        "ư": "u", "ứ": "u", "ừ": "u", "ử": "u", "ữ": "u", "ự": "u",
        "ý": "y", "ỳ": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
        "đ": "d",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def sample_key(path: Path) -> str:
    """Return a stable key to match the same phrase across emotion folders."""
    stem = path.stem
    numeric_prefix = re.match(r"^(\d+)_", stem)
    if numeric_prefix:
        return numeric_prefix.group(1)

    normalized = normalize_text(stem)
    normalized = re.sub(r"\b(20\d{6})\b.*$", "", normalized).strip()
    normalized = re.sub(r"\b\d{6}\b.*$", "", normalized).strip()
    for word in sorted(EMOTION_WORDS, key=len, reverse=True):
        normalized = re.sub(rf"\b{re.escape(normalize_text(word))}\b", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip() or normalize_text(path.stem)


def default_test_root() -> Path:
    archive_test = SOFT_DIR / "archive" / "test"
    if archive_test.exists():
        return archive_test
    return PROJECT_ROOT / "data"


def find_emotion_dir(test_root: Path, emotion: str) -> Path | None:
    for alias in EMOTION_DIR_ALIASES[emotion]:
        candidate = test_root / alias
        if candidate.exists():
            return candidate

    normalized_aliases = {normalize_text(alias) for alias in EMOTION_DIR_ALIASES[emotion]}
    for child in test_root.iterdir() if test_root.exists() else []:
        if child.is_dir() and normalize_text(child.name) in normalized_aliases:
            return child
    return None


def find_wav_files(test_root: Path) -> dict[str, list[Path]]:
    files_by_emotion: dict[str, list[Path]] = {}
    for emotion in EMOTIONS:
        emotion_dir = find_emotion_dir(test_root, emotion)
        files_by_emotion[emotion] = sorted(emotion_dir.glob("*.wav")) if emotion_dir else []
    return files_by_emotion


def choose_files(
    files_by_emotion: dict[str, list[Path]],
    per_emotion: int,
    selection: str,
) -> dict[str, list[Path]]:
    if selection == "first":
        return {emotion: files[:per_emotion] for emotion, files in files_by_emotion.items()}

    by_key: dict[str, dict[str, Path]] = defaultdict(dict)
    for emotion, files in files_by_emotion.items():
        for path in files:
            key = sample_key(path)
            by_key[key].setdefault(emotion, path)

    shared_keys = [
        key for key, emotion_map in by_key.items()
        if all(emotion in emotion_map for emotion in EMOTIONS)
    ]

    def key_sort(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    selected_keys = sorted(shared_keys, key=key_sort)[:per_emotion]
    if len(selected_keys) >= per_emotion:
        return {
            emotion: [by_key[key][emotion] for key in selected_keys]
            for emotion in EMOTIONS
        }

    print(
        f"Khong du {per_emotion} mau chung giua 4 cam xuc "
        f"(chi co {len(selected_keys)}), fallback sang {per_emotion} file dau tien."
    )
    return {emotion: files[:per_emotion] for emotion, files in files_by_emotion.items()}


def load_wav(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=cfg.AUDIO_SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def predict_file(model, classes: list[str], path: Path) -> dict:
    raw_audio = load_wav(path)
    model_audio = audio_pipeline.prepare_for_model(raw_audio)
    pred, confidence, windows = audio_pipeline.predict_emotion_sliding(
        model, classes, model_audio
    )
    rms, peak = audio_pipeline.quality_stats(raw_audio)
    return {
        "predicted": pred,
        "confidence": round(confidence, 2),
        "windows": windows,
        "duration_sec": round(len(raw_audio) / cfg.AUDIO_SAMPLE_RATE, 2),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
    }


def build_rows(model, classes: list[str], selected: dict[str, list[Path]]) -> list[dict]:
    rows = []
    index = 1
    for emotion in EMOTIONS:
        for run_no, path in enumerate(selected[emotion], start=1):
            result = predict_file(model, classes, path)
            passed = result["predicted"] == emotion
            rows.append({
                "stt": index,
                "emotion_true": emotion,
                "run": run_no,
                "file": str(path.relative_to(PROJECT_ROOT)),
                "emotion_pred": result["predicted"],
                "confidence_percent": result["confidence"],
                "windows": result["windows"],
                "duration_sec": result["duration_sec"],
                "rms": result["rms"],
                "peak": result["peak"],
                "result": "Dung" if passed else "Sai",
            })
            status = "OK" if passed else "SAI"
            print(
                f"[{index:02d}] {emotion:<10} lan {run_no}: "
                f"{result['predicted']:<10} {result['confidence']:>6.2f}% {status}"
            )
            index += 1
    return rows


def summary(rows: list[dict]) -> dict:
    correct = sum(1 for row in rows if row["result"] == "Dung")
    total = len(rows)
    per_emotion = {}
    for emotion in EMOTIONS:
        subset = [row for row in rows if row["emotion_true"] == emotion]
        n_ok = sum(1 for row in subset if row["result"] == "Dung")
        per_emotion[emotion] = {
            "total": len(subset),
            "correct": n_ok,
            "accuracy_percent": round((n_ok / len(subset) * 100) if subset else 0.0, 2),
            "predicted_counts": dict(Counter(row["emotion_pred"] for row in subset)),
        }
    return {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy_percent": round((correct / total * 100) if total else 0.0, 2),
        "per_emotion": per_emotion,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], report: dict) -> None:
    headers = [
        "STT", "Cam xuc dung", "Lan", "File", "Du doan",
        "Tin cay (%)", "Ket qua",
    ]
    lines = [
        "# Bang danh gia nhan dien cam xuc giong noi",
        "",
        f"- Tong so mau: {report['total']}",
        f"- Du doan dung: {report['correct']}",
        f"- Du doan sai: {report['wrong']}",
        f"- Accuracy: {report['accuracy_percent']}%",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                str(row["stt"]),
                row["emotion_true"],
                str(row["run"]),
                row["file"],
                row["emotion_pred"],
                f"{row['confidence_percent']:.2f}",
                row["result"],
            ])
            + " |"
        )
    lines.extend(["", "## Tong ket theo cam xuc", ""])
    lines.append("| Cam xuc | Dung/Tong | Accuracy (%) | So lan du doan |")
    lines.append("| --- | --- | --- | --- |")
    for emotion, stats in report["per_emotion"].items():
        lines.append(
            f"| {emotion} | {stats['correct']}/{stats['total']} | "
            f"{stats['accuracy_percent']:.2f} | {stats['predicted_counts']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test model cam xuc giong noi voi 4 cam xuc x 5 file."
    )
    parser.add_argument("--test-root", type=Path, default=default_test_root())
    parser.add_argument("--per-emotion", type=int, default=5)
    parser.add_argument(
        "--selection",
        choices=["shared", "first"],
        default="shared",
        help="shared = chon cac mau chung giua 4 cam xuc, first = lay file dau tien",
    )
    parser.add_argument("--output-dir", type=Path, default=SOFT_DIR / "test" / "results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 72)
    print("TEST NHAN DIEN CAM XUC GIONG NOI - 4 CAM XUC X 5 FILE")
    print("=" * 72)
    print(f"Test root : {args.test_root}")
    print(f"Model     : {cfg.AUDIO_MODEL_PATH}")
    print(f"Labels    : {cfg.AUDIO_LABEL_PATH}")

    files_by_emotion = find_wav_files(args.test_root)
    missing = [emotion for emotion, files in files_by_emotion.items() if len(files) < args.per_emotion]
    if missing:
        raise SystemExit(
            "Khong du file WAV cho: "
            + ", ".join(f"{emotion} ({len(files_by_emotion[emotion])})" for emotion in missing)
        )

    selected = choose_files(files_by_emotion, args.per_emotion, args.selection)
    print("\nCac file se test:")
    for emotion in EMOTIONS:
        for idx, path in enumerate(selected[emotion], start=1):
            print(f"  {emotion:<10} {idx}. {path.name}")

    print("\nDang nap model...")
    model, classes = load_audio_model()
    print("\nDang du doan...")
    rows = build_rows(model, classes, selected)
    report = summary(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"voice_emotion_5x4_{timestamp}.csv"
    md_path = args.output_dir / f"voice_emotion_5x4_{timestamp}.md"
    json_path = args.output_dir / f"voice_emotion_5x4_{timestamp}.json"

    write_csv(csv_path, rows)
    write_markdown(md_path, rows, report)
    json_path.write_text(
        json.dumps({"summary": report, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print(f"Accuracy: {report['accuracy_percent']}% ({report['correct']}/{report['total']})")
    for emotion, stats in report["per_emotion"].items():
        print(
            f"  {emotion:<10}: {stats['accuracy_percent']:>6.2f}% "
            f"({stats['correct']}/{stats['total']})"
        )
    print("\nDa luu bang danh gia:")
    print(f"  CSV : {csv_path}")
    print(f"  MD  : {md_path}")
    print(f"  JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
