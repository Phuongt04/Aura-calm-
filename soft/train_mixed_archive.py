"""
Fine-tune model cam xuc giong noi bang du lieu tron:

Train = soft/dataset + soft/archive/train/phase1 + soft/archive/train/phase2
Val   = soft/archive/val
Test  = soft/archive/test
Start = final_production_model.h5

Chay mac dinh:
  python soft/train_mixed_archive.py

Thu nghiem khi model nham vui nhieu:
  python soft/train_mixed_archive.py --train-source archive-only --learning-rate 5e-6 --epochs 12 --patience 4
  python soft/train_mixed_archive.py --vui-weight 1.4 --learning-rate 5e-6 --epochs 12 --patience 4

Sau khi co ket qua tot, co the thay model production:
  python soft/train_mixed_archive.py --replace-production
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from emotion_chat import audio as audio_pipeline
from emotion_chat import config as cfg
from emotion_chat.model_loader import safe_load_keras


CLASSES = ["binhthuong", "buon", "gian", "vui"]
ARCHIVE_LABEL_MAP = {
    "neutral": "binhthuong",
    "sad": "buon",
    "angry": "gian",
    "happy": "vui",
}


def archive_cache_path(cache_root: Path, wav_path: Path, archive_root: Path) -> Path:
    relative = wav_path.relative_to(archive_root).with_suffix(".npy")
    return cache_root / relative


def wav_to_feature(wav_path: Path) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=cfg.AUDIO_SAMPLE_RATE, mono=True)
    y = audio_pipeline.prepare_for_model(y.astype(np.float32))
    return audio_pipeline.preprocess(y).astype(np.float32)


def cached_wav_feature(wav_path: Path, cache_root: Path, archive_root: Path, refresh: bool) -> np.ndarray:
    cache_path = archive_cache_path(cache_root, wav_path, archive_root)
    if cache_path.exists() and not refresh:
        data = np.load(cache_path, allow_pickle=False)
    else:
        data = wav_to_feature(wav_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, data)

    if data.shape != (cfg.AUDIO_N_FRAMES, cfg.AUDIO_IMG_SIZE, cfg.AUDIO_IMG_SIZE, 1):
        raise ValueError(f"Sai shape {data.shape}: {wav_path}")
    return data.astype(np.float32)


def load_old_dataset(dataset_dir: Path) -> tuple[list[np.ndarray], list[str], list[str]]:
    X, y, source = [], [], []
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in CLASSES:
            continue
        for path in sorted(class_dir.glob("*.npy")):
            data = np.load(path, allow_pickle=False)
            if data.shape == (cfg.AUDIO_N_FRAMES, cfg.AUDIO_IMG_SIZE, cfg.AUDIO_IMG_SIZE, 1):
                X.append(data.astype(np.float32))
                y.append(class_dir.name)
                source.append(str(path.relative_to(PROJECT_ROOT)))
    return X, y, source


def iter_archive_wavs(split_dir: Path) -> list[tuple[Path, str]]:
    items = []
    if not split_dir.exists():
        return items

    for raw_label, mapped_label in ARCHIVE_LABEL_MAP.items():
        label_dir = split_dir / raw_label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.glob("*.wav")):
            items.append((path, mapped_label))
    return items


def iter_archive_train_wavs(archive_root: Path) -> list[tuple[Path, str]]:
    items = []
    for phase in ["phase1", "phase2"]:
        items.extend(iter_archive_wavs(archive_root / "train" / phase))
    return items


def load_archive_features(
    items: list[tuple[Path, str]],
    cache_root: Path,
    archive_root: Path,
    refresh_cache: bool,
    title: str,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    X, y, source = [], [], []
    total = len(items)
    for idx, (path, label) in enumerate(items, start=1):
        if idx == 1 or idx % 500 == 0 or idx == total:
            print(f"  {title}: {idx}/{total}")
        X.append(cached_wav_feature(path, cache_root, archive_root, refresh_cache))
        y.append(label)
        source.append(str(path.relative_to(PROJECT_ROOT)))
    return X, y, source


def encode_labels(labels: list[str]) -> np.ndarray:
    class_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    return np.array([class_to_idx[label] for label in labels], dtype=np.int32)


def limit_triplets(
    X: list[np.ndarray],
    y: list[str],
    source: list[str],
    limit_per_class: int,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    if limit_per_class <= 0:
        return X, y, source

    counts: Counter[str] = Counter()
    out_X, out_y, out_source = [], [], []
    for data, label, src in zip(X, y, source):
        if counts[label] >= limit_per_class:
            continue
        out_X.append(data)
        out_y.append(label)
        out_source.append(src)
        counts[label] += 1
    return out_X, out_y, out_source


def limit_archive_items(
    items: list[tuple[Path, str]],
    limit_per_class: int,
) -> list[tuple[Path, str]]:
    if limit_per_class <= 0:
        return items

    counts: Counter[str] = Counter()
    selected = []
    for path, label in items:
        if counts[label] >= limit_per_class:
            continue
        selected.append((path, label))
        counts[label] += 1
    return selected


def compute_class_weight(y_int: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y_int, minlength=len(CLASSES)).astype(np.float32)
    total = float(np.sum(counts))
    weights = {}
    for idx, count in enumerate(counts):
        weights[idx] = float(total / (len(CLASSES) * count)) if count > 0 else 1.0
    return weights


def apply_class_weight_multipliers(
    class_weight: dict[int, float],
    multipliers: dict[str, float],
) -> dict[int, float]:
    adjusted = dict(class_weight)
    for idx, label in enumerate(CLASSES):
        adjusted[idx] = float(adjusted[idx] * multipliers[label])
    return adjusted


def make_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    training: bool,
    sample_weight: np.ndarray | None = None,
) -> tf.data.Dataset:
    if sample_weight is None:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, y, sample_weight.astype(np.float32)))
    if training:
        dataset = dataset.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class ValidationF1Metrics(tf.keras.callbacks.Callback):
    def __init__(self, X_val: np.ndarray, y_val_int: np.ndarray) -> None:
        super().__init__()
        self.X_val = X_val
        self.y_val_int = y_val_int

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        probs = self.model.predict(self.X_val, verbose=0)
        pred = np.argmax(probs, axis=1)
        per_class_f1 = f1_score(
            self.y_val_int,
            pred,
            labels=list(range(len(CLASSES))),
            average=None,
            zero_division=0,
        )
        logs["val_macro_f1"] = float(np.mean(per_class_f1))
        for idx, label in enumerate(CLASSES):
            logs[f"val_{label}_f1"] = float(per_class_f1[idx])
        print(
            " - val_macro_f1: "
            f"{logs['val_macro_f1']:.4f}"
            f" - val_vui_f1: {logs['val_vui_f1']:.4f}"
        )


def save_labels(label_path: Path) -> None:
    with label_path.open("wb") as f:
        pickle.dump(np.array(CLASSES), f)
    soft_label_path = BASE_DIR / "label_mapping_final.pkl"
    if soft_label_path != label_path:
        with soft_label_path.open("wb") as f:
            pickle.dump(np.array(CLASSES), f)


def evaluate_model(model, X: np.ndarray, y_int: np.ndarray, output_dir: Path, split_name: str) -> dict:
    probs = model.predict(X, verbose=0)
    pred = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_int, pred))
    cm = confusion_matrix(y_int, pred, labels=list(range(len(CLASSES))))
    report_text = classification_report(
        y_int,
        pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
    )
    report_dict = classification_report(
        y_int,
        pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
        output_dict=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{split_name}_classification_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / f"{split_name}_metrics.json").write_text(
        json.dumps(
            {
                "accuracy": acc,
                "classes": CLASSES,
                "confusion_matrix": cm.tolist(),
                "classification_report": report_dict,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / f"{split_name}_confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASSES])
        for label, row in zip(CLASSES, cm.tolist()):
            writer.writerow([label, *row])

    print(f"\n{split_name.upper()} accuracy: {acc * 100:.2f}%")
    print(report_text)
    return {"accuracy": acc, "confusion_matrix": cm.tolist(), "report": report_dict}


def backup_and_replace(new_model_path: Path, production_model_path: Path) -> Path:
    backup_dir = PROJECT_ROOT / "model_backups"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"final_production_model_before_mixed_{timestamp}.h5"
    shutil.copy2(production_model_path, backup_path)
    shutil.copy2(new_model_path, production_model_path)
    return backup_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune audio emotion model on mixed old + archive data.")
    parser.add_argument("--old-dataset", type=Path, default=BASE_DIR / "dataset")
    parser.add_argument("--archive-root", type=Path, default=BASE_DIR / "archive")
    parser.add_argument("--cache-root", type=Path, default=BASE_DIR / "archive_feature_cache")
    parser.add_argument("--start-model", type=Path, default=PROJECT_ROOT / "final_production_model.h5")
    parser.add_argument("--label-path", type=Path, default=PROJECT_ROOT / "label_mapping_final.pkl")
    parser.add_argument("--output-model", type=Path, default=PROJECT_ROOT / "final_production_model_mixed_archive.h5")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "mixed_archive_training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--train-source",
        choices=["mixed", "archive-only", "old-only"],
        default="mixed",
        help="mixed = old + archive, archive-only = chi archive train, old-only = chi soft/dataset.",
    )
    parser.add_argument("--binhthuong-weight", type=float, default=1.0)
    parser.add_argument("--buon-weight", type=float, default=1.0)
    parser.add_argument("--gian-weight", type=float, default=1.0)
    parser.add_argument(
        "--vui-weight",
        type=float,
        default=1.0,
        help="Nhan them class weight cho lop vui. Thu 1.2-1.6 khi recall vui thap.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["val_accuracy", "val_macro_f1", "val_vui_f1"],
        default="val_accuracy",
        help="Metric dung de chon best checkpoint va early stopping.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Debug only. 0 = use all data.",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--replace-production", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FINE-TUNE MODEL CAM XUC - OLD DATASET + ARCHIVE")
    print("=" * 72)
    print(f"Train old     : {args.old_dataset}")
    print(f"Train archive : {args.archive_root / 'train' / 'phase1'}")
    print(f"Train archive : {args.archive_root / 'train' / 'phase2'}")
    print(f"Val           : {args.archive_root / 'val'}")
    print(f"Test          : {args.archive_root / 'test'}")
    print(f"Train source  : {args.train_source}")
    print(f"Selection     : {args.selection_metric}")
    print(f"Start model   : {args.start_model}")
    print(f"Output model  : {args.output_model}")

    print("\n[1] Load old .npy dataset...")
    X_old, y_old, old_source = load_old_dataset(args.old_dataset)
    X_old, y_old, old_source = limit_triplets(X_old, y_old, old_source, args.limit_per_class)
    print(f"  Old samples: {len(X_old)} | {dict(Counter(y_old))}")

    print("\n[2] Load/cache archive train WAV...")
    train_archive_items = limit_archive_items(
        iter_archive_train_wavs(args.archive_root),
        args.limit_per_class,
    )
    X_archive, y_archive, archive_source = load_archive_features(
        train_archive_items,
        args.cache_root,
        args.archive_root,
        args.refresh_cache,
        "archive train",
    )
    print(f"  Archive train samples: {len(X_archive)} | {dict(Counter(y_archive))}")

    print("\n[3] Load/cache archive val WAV...")
    val_items = limit_archive_items(iter_archive_wavs(args.archive_root / "val"), args.limit_per_class)
    X_val_list, y_val, val_source = load_archive_features(
        val_items,
        args.cache_root,
        args.archive_root,
        args.refresh_cache,
        "archive val",
    )
    print(f"  Val samples: {len(X_val_list)} | {dict(Counter(y_val))}")

    print("\n[4] Load/cache archive test WAV...")
    test_items = limit_archive_items(iter_archive_wavs(args.archive_root / "test"), args.limit_per_class)
    X_test_list, y_test, test_source = load_archive_features(
        test_items,
        args.cache_root,
        args.archive_root,
        args.refresh_cache,
        "archive test",
    )
    print(f"  Test samples: {len(X_test_list)} | {dict(Counter(y_test))}")

    if args.train_source == "archive-only":
        train_parts = X_archive
        y_train_labels = y_archive
        train_source_labels = archive_source
    elif args.train_source == "old-only":
        train_parts = X_old
        y_train_labels = y_old
        train_source_labels = old_source
    else:
        train_parts = [*X_old, *X_archive]
        y_train_labels = [*y_old, *y_archive]
        train_source_labels = [*old_source, *archive_source]

    if not train_parts:
        raise ValueError(f"Khong co du lieu train cho train-source={args.train_source}")

    X_train = np.stack(train_parts, axis=0).astype(np.float32)
    y_train_int = encode_labels(y_train_labels)
    y_train = to_categorical(y_train_int, num_classes=len(CLASSES))

    X_val = np.stack(X_val_list, axis=0).astype(np.float32)
    y_val_int = encode_labels(y_val)
    y_val_cat = to_categorical(y_val_int, num_classes=len(CLASSES))

    X_test = np.stack(X_test_list, axis=0).astype(np.float32)
    y_test_int = encode_labels(y_test)

    class_weight = compute_class_weight(y_train_int)
    weight_multipliers = {
        "binhthuong": args.binhthuong_weight,
        "buon": args.buon_weight,
        "gian": args.gian_weight,
        "vui": args.vui_weight,
    }
    class_weight = apply_class_weight_multipliers(class_weight, weight_multipliers)
    sample_weight = np.array([class_weight[int(label)] for label in y_train_int], dtype=np.float32)
    print("\nTrain class counts:", dict(Counter(y_train_labels)))
    print("Class weight multipliers:", {k: round(v, 4) for k, v in weight_multipliers.items()})
    print("Class weights:", {CLASSES[k]: round(v, 4) for k, v in class_weight.items()})

    save_labels(args.label_path)

    print("\n[5] Load start model and fine-tune...")
    model = safe_load_keras(args.start_model)
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_model_path = run_dir / "best_mixed_archive_model.h5"
    callbacks = [
        ValidationF1Metrics(X_val, y_val_int),
        EarlyStopping(
            monitor=args.selection_metric,
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            str(best_model_path),
            monitor=args.selection_metric,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        make_dataset(X_train, y_train, args.batch_size, training=True, sample_weight=sample_weight),
        validation_data=make_dataset(X_val, y_val_cat, args.batch_size, training=False),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output_model))

    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    metadata = {
        "classes": CLASSES,
        "start_model": str(args.start_model),
        "output_model": str(args.output_model),
        "train_old_samples": len(X_old),
        "train_archive_samples": len(X_archive),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_counts": dict(Counter(y_train_labels)),
        "val_counts": dict(Counter(y_val)),
        "test_counts": dict(Counter(y_test)),
        "class_weight": {CLASSES[k]: v for k, v in class_weight.items()},
        "class_weight_multipliers": weight_multipliers,
        "train_source": args.train_source,
        "selection_metric": args.selection_metric,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "history": history_dict,
        "old_source_count": len(old_source),
        "archive_source_count": len(archive_source),
        "selected_train_source_count": len(train_source_labels),
        "val_source_count": len(val_source),
        "test_source_count": len(test_source),
    }
    (run_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[6] Evaluate best/restored model...")
    val_metrics = evaluate_model(model, X_val, y_val_int, run_dir, "val")
    test_metrics = evaluate_model(model, X_test, y_test_int, run_dir, "test")

    metadata["val_metrics"] = val_metrics
    metadata["test_metrics"] = test_metrics
    (run_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.replace_production:
        backup_path = backup_and_replace(args.output_model, PROJECT_ROOT / "final_production_model.h5")
        print(f"\nProduction model replaced. Backup: {backup_path}")

    print("\nDone.")
    print(f"Run dir      : {run_dir}")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Output model : {args.output_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
