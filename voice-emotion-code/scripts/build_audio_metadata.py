import argparse
import csv
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.voice_emotion_labels import infer_label_from_path, infer_original_id, infer_source_from_path


AUDIO_EXTENSIONS = set([".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"])
FIELDS = ["filepath", "label", "source", "original_id", "duration", "sample_rate", "channels", "is_valid"]


def wav_info(path):
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.getnframes()
        duration = float(frames) / float(sample_rate) if sample_rate else 0.0
    return duration, sample_rate, channels


def librosa_info(path):
    import librosa
    y, sr = librosa.load(str(path), sr=None, mono=False)
    if getattr(y, "ndim", 1) == 1:
        channels = 1
        samples = y.shape[0]
    else:
        channels = y.shape[0]
        samples = y.shape[-1]
    duration = float(samples) / float(sr) if sr else 0.0
    return duration, int(sr), int(channels)


def audio_info(path):
    try:
        if path.suffix.lower() == ".wav":
            return wav_info(path)
        return librosa_info(path)
    except Exception:
        return None, None, None


def iter_audio_files(data_dir):
    for path in sorted(Path(data_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def build_metadata(data_dirs, output_path, rejected_log):
    data_dirs = [Path(item) for item in data_dirs]
    output_path = Path(output_path)
    rejected_log = Path(rejected_log)
    rows = []
    rejected = []

    for data_dir in data_dirs:
        for path in iter_audio_files(data_dir):
            label = infer_label_from_path(path, data_dir)
            source = infer_source_from_path(path)
            original_id = infer_original_id(path)
            duration, sample_rate, channels = audio_info(path)
            readable = duration is not None
            if label is None:
                rejected.append({
                    "filepath": str(path),
                    "reason": "unsupported_or_unknown_label",
                    "source": source,
                })
                continue
            rows.append({
                "filepath": str(path),
                "label": label,
                "source": source,
                "original_id": original_id,
                "duration": "" if duration is None else "%.6f" % duration,
                "sample_rate": "" if sample_rate is None else str(sample_rate),
                "channels": "" if channels is None else str(channels),
                "is_valid": "true" if readable else "false",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    rejected_log.parent.mkdir(parents=True, exist_ok=True)
    with rejected_log.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "reason", "source"])
        writer.writeheader()
        writer.writerows(rejected)

    print("Metadata rows:", len(rows))
    print("Rejected rows:", len(rejected))
    print("Output:", output_path)
    print("Rejected log:", rejected_log)


def parse_args():
    parser = argparse.ArgumentParser(description="Build audio_metadata.csv for speech emotion data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        nargs="+",
        default=[PROJECT_ROOT / "data"],
        help="One or more roots to scan, for example: --data-dir data soft/archive",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "audio_metadata.csv")
    parser.add_argument("--rejected-log", type=Path, default=PROJECT_ROOT / "reports" / "audio_metadata_rejected.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_metadata(args.data_dir, args.output, args.rejected_log)
