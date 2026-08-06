import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_groups(rows, train_ratio, val_ratio, test_ratio, seed):
    groups = defaultdict(list)
    for row in rows:
        if str(row.get("is_valid", "true")).lower() not in ("true", "1", "yes"):
            continue
        key = row.get("original_id") or row.get("filepath")
        stratum = "%s|%s" % (row.get("label", ""), row.get("source", "unknown"))
        groups[(stratum, key)].append(row)

    by_stratum = defaultdict(list)
    for (stratum, key), group_rows in groups.items():
        by_stratum[stratum].append(group_rows)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for stratum, group_list in sorted(by_stratum.items()):
        rng.shuffle(group_list)
        n = len(group_list)
        if n == 1:
            train_n, val_n = 1, 0
        elif n == 2:
            train_n, val_n = 1, 1
        else:
            train_n = max(1, int(round(n * train_ratio)))
            val_n = max(1, int(round(n * val_ratio)))
            if train_n + val_n >= n:
                train_n = max(1, n - 2)
                val_n = 1
        for index, group_rows in enumerate(group_list):
            if index < train_n:
                split = "train"
            elif index < train_n + val_n:
                split = "val"
            else:
                split = "test"
            splits[split].extend(group_rows)
    return splits


def parse_args():
    parser = argparse.ArgumentParser(description="Create reusable train/val/test metadata splits.")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "audio_metadata.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "voice_emotion_config.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    split_cfg = config["split"]
    rows = read_csv(args.metadata)
    splits = split_groups(
        rows,
        float(split_cfg["train_ratio"]),
        float(split_cfg["val_ratio"]),
        float(split_cfg["test_ratio"]),
        int(split_cfg["random_seed"]),
    )
    fieldnames = list(rows[0].keys()) if rows else []
    for name, split_rows in splits.items():
        out_path = args.output_dir / ("%s_metadata.csv" % name)
        write_csv(out_path, split_rows, fieldnames)
        print("%s: %d -> %s" % (name, len(split_rows), out_path))
