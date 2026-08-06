import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_preprocessing import read_audio
from src.voice_emotion_labels import LABELS


REPORT_FIELDS = [
    "filepath", "label", "source", "original_id", "duration", "sample_rate",
    "channels", "is_valid", "issues",
]


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def audio_quality(path, target_sr):
    y, sr = read_audio(path, sample_rate=target_sr, mono=True)
    if y.size == 0:
        return {"empty": True, "rms": 0.0, "peak": 0.0, "clip_ratio": 0.0}
    rms = float(np.sqrt(np.mean(np.square(y))))
    peak = float(np.max(np.abs(y)))
    clip_ratio = float(np.mean(np.abs(y) >= 0.98))
    return {"empty": False, "rms": rms, "peak": peak, "clip_ratio": clip_ratio}


def validate_rows(rows, target_sr=16000, min_duration=0.5):
    seen_paths = set()
    group_counter = Counter()
    for row in rows:
        group_counter[(row.get("original_id", ""), row.get("label", ""))] += 1

    output_rows = []
    summary = {
        "total_rows": len(rows),
        "valid_rows": 0,
        "invalid_rows": 0,
        "issues": Counter(),
        "source_emotion_counts": defaultdict(Counter),
        "class_counts": Counter(),
        "source_counts": Counter(),
    }

    for row in rows:
        issues = []
        path = row.get("filepath", "")
        label = row.get("label", "")
        source = row.get("source", "unknown")
        duration = safe_float(row.get("duration", "0"))
        sample_rate = safe_int(row.get("sample_rate", "0"))
        channels = safe_int(row.get("channels", "0"))

        if path in seen_paths:
            issues.append("duplicate_filepath")
        seen_paths.add(path)

        if label not in LABELS:
            issues.append("invalid_label")
        if duration > 0 and duration < min_duration:
            issues.append("too_short")
        if sample_rate and sample_rate != target_sr:
            issues.append("sample_rate_mismatch")
        if channels and channels != 1:
            issues.append("not_mono")
        if group_counter[(row.get("original_id", ""), label)] > 1:
            issues.append("possible_augmentation_group")

        try:
            q = audio_quality(path, target_sr)
            if q["empty"]:
                issues.append("empty_audio")
            if q["rms"] < 1e-4 or q["peak"] < 1e-3:
                issues.append("near_silence")
            if q["clip_ratio"] > 0.02:
                issues.append("severe_clipping")
        except Exception:
            issues.append("unreadable_audio")

        is_valid = len([i for i in issues if i not in ("sample_rate_mismatch", "not_mono", "possible_augmentation_group")]) == 0
        for issue in issues:
            summary["issues"][issue] += 1
        if is_valid:
            summary["valid_rows"] += 1
        else:
            summary["invalid_rows"] += 1

        summary["source_emotion_counts"][source][label] += 1
        summary["class_counts"][label] += 1
        summary["source_counts"][source] += 1

        out = dict(row)
        out["is_valid"] = "true" if is_valid else "false"
        out["issues"] = ";".join(issues)
        output_rows.append(out)

    summary["issues"] = dict(summary["issues"])
    summary["class_counts"] = dict(summary["class_counts"])
    summary["source_counts"] = dict(summary["source_counts"])
    summary["source_emotion_counts"] = {
        source: dict(counter) for source, counter in summary["source_emotion_counts"].items()
    }
    return output_rows, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Validate audio metadata and report quality issues.")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "audio_metadata.csv")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "audio_dataset_report.csv")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "reports" / "audio_dataset_summary.json")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = read_csv(args.metadata)
    report_rows, summary = validate_rows(rows, args.target_sr, args.min_duration)
    write_csv(args.report, REPORT_FIELDS, report_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Rows:", summary["total_rows"])
    print("Valid:", summary["valid_rows"])
    print("Invalid:", summary["invalid_rows"])
    print("Report:", args.report)
    print("Summary:", args.summary)
