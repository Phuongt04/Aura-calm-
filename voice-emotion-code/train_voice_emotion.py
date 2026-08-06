import argparse
import csv
import json
import os
import pickle
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, to_categorical

try:
    from sklearn.metrics import f1_score
except Exception:
    def f1_score(y_true, y_pred, labels=None, average="macro", zero_division=0):
        labels = labels or sorted(set(list(y_true) + list(y_pred)))
        scores = []
        for label in labels:
            tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
            fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
            fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
            precision = tp / float(tp + fp) if (tp + fp) else float(zero_division)
            recall = tp / float(tp + fn) if (tp + fn) else float(zero_division)
            if precision + recall:
                scores.append(2.0 * precision * recall / (precision + recall))
            else:
                scores.append(float(zero_division))
        return float(np.mean(scores)) if average == "macro" else scores

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.create_audio_splits import split_groups, write_csv
from src.audio_augmentation import NoiseLibrary, augment_waveform, spec_augment
from src.audio_preprocessing import (
    expected_time_steps,
    log_mel_delta_features,
    prevent_clipping,
    read_audio,
    symmetric_pad_or_center_crop,
    trim_silence,
)
from src.config_utils import load_config, save_json
from src.models.cnn_gru_ser import build_cnn_gru_ser, print_model_report
from src.voice_emotion_labels import LABELS, LABEL_TO_INDEX


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_metadata(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def ensure_splits(metadata_path, config):
    data_dir = Path(metadata_path).parent
    train_path = data_dir / "train_metadata.csv"
    val_path = data_dir / "val_metadata.csv"
    test_path = data_dir / "test_metadata.csv"
    if train_path.exists() and val_path.exists() and test_path.exists():
        return train_path, val_path, test_path

    rows = read_metadata(metadata_path)
    split_cfg = config["split"]
    splits = split_groups(
        rows,
        float(split_cfg["train_ratio"]),
        float(split_cfg["val_ratio"]),
        float(split_cfg["test_ratio"]),
        int(split_cfg["random_seed"]),
    )
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(train_path, splits["train"], fieldnames)
    write_csv(val_path, splits["val"], fieldnames)
    write_csv(test_path, splits["test"], fieldnames)
    return train_path, val_path, test_path


def filter_valid(rows):
    return [
        row for row in rows
        if str(row.get("is_valid", "true")).lower() in ("true", "1", "yes")
        and row.get("label") in LABEL_TO_INDEX
    ]


def class_weights(rows):
    counts = Counter([row["label"] for row in rows])
    total = float(sum(counts.values()))
    weights = {}
    for label in LABELS:
        count = float(counts.get(label, 0))
        weights[label] = total / (len(LABELS) * count) if count > 0 else 1.0
    return weights


def source_sampling_weights(config):
    cfg = config.get("sampling", {})
    return {
        "tess": float(cfg.get("tess_weight", 0.25)),
        "vietnamese_kaggle": float(cfg.get("vietnamese_kaggle_weight", 0.55)),
        "self_recorded": float(cfg.get("self_recorded_weight", 0.20)),
        "unknown": float(cfg.get("self_recorded_weight", 0.20)),
    }


class AudioEmotionSequence(Sequence):
    def __init__(self, rows, config, batch_size, training=False, seed=42, noise_dir=None, **kwargs):
        super(AudioEmotionSequence, self).__init__(**kwargs)
        self.rows = list(rows)
        self.config = config
        self.batch_size = int(batch_size)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.class_weight = class_weights(self.rows)
        self.by_source = defaultdict(list)
        for row in self.rows:
            self.by_source[row.get("source", "unknown")].append(row)
        self.noise_library = NoiseLibrary(noise_dir, sample_rate=int(config["audio"]["sample_rate"]))
        self.source_weights = source_sampling_weights(config)

    def __len__(self):
        return max(1, int(np.ceil(len(self.rows) / float(self.batch_size))))

    def on_epoch_end(self):
        self.epoch += 1

    def _choose_row(self, rng, index):
        if not self.training:
            return self.rows[index % len(self.rows)]
        sources = [src for src, rows in self.by_source.items() if rows]
        weights = np.array([self.source_weights.get(src, 0.10) for src in sources], dtype=np.float32)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(sources), dtype=np.float32)
        weights = weights / weights.sum()
        source = sources[int(rng.choice(len(sources), p=weights))]
        candidates = self.by_source[source]
        return candidates[int(rng.randint(0, len(candidates)))]

    def _load_feature(self, row, rng):
        audio_cfg = self.config["audio"]
        sample_rate = int(audio_cfg["sample_rate"])
        duration = float(audio_cfg["duration_seconds"])
        num_samples = int(round(sample_rate * duration))
        y, sr = read_audio(row["filepath"], sample_rate=sample_rate, mono=True)
        y = trim_silence(y, sr)
        y = prevent_clipping(y)
        y = symmetric_pad_or_center_crop(y, num_samples)
        if self.training:
            y = augment_waveform(y, rng, self.config, self.noise_library, sample_rate=sample_rate)
        x = log_mel_delta_features(
            y,
            sample_rate=sample_rate,
            n_fft=int(audio_cfg["n_fft"]),
            win_length=int(audio_cfg["win_length"]),
            hop_length=int(audio_cfg["hop_length"]),
            n_mels=int(audio_cfg["n_mels"]),
            fmin=int(audio_cfg["fmin"]),
            fmax=int(audio_cfg["fmax"]),
        )
        if self.training:
            x = spec_augment(x, rng, self.config)
        return x.astype(np.float32)

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed + self.epoch * 100000 + index)
        xs, ys, weights = [], [], []
        for i in range(self.batch_size):
            row = self._choose_row(rng, index * self.batch_size + i)
            xs.append(self._load_feature(row, rng))
            ys.append(LABEL_TO_INDEX[row["label"]])
            weights.append(self.class_weight[row["label"]])
        x = np.stack(xs, axis=0)
        y = to_categorical(np.asarray(ys, dtype=np.int32), num_classes=len(LABELS))
        if self.training:
            return x, y, np.asarray(weights, dtype=np.float32)
        return x, y


class ValidationMacroF1(Callback):
    def __init__(self, val_sequence):
        super(ValidationMacroF1, self).__init__()
        self.val_sequence = val_sequence

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true = []
        y_pred = []
        for batch in range(len(self.val_sequence)):
            x, y = self.val_sequence[batch]
            probs = self.model.predict(x, verbose=0)
            y_true.extend(np.argmax(y, axis=1).tolist())
            y_pred.extend(np.argmax(probs, axis=1).tolist())
        macro = f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)
        logs["val_macro_f1"] = float(macro)
        print(" - val_macro_f1: %.4f" % macro)


def git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def rows_summary(rows):
    return {
        "samples": len(rows),
        "class_counts": dict(Counter(row["label"] for row in rows)),
        "source_counts": dict(Counter(row.get("source", "unknown") for row in rows)),
    }


def save_label_mapping(path):
    with Path(path).open("wb") as handle:
        pickle.dump(np.array(LABELS), handle)


def compile_model(model, learning_rate, label_smoothing):
    try:
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    except TypeError:
        loss = "categorical_crossentropy"
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=loss,
        metrics=["accuracy"],
    )


def train_stage(model, train_rows, val_rows, config, run_dir, model_path, epochs, learning_rate, stage_name, noise_dir=None):
    train_seq = AudioEmotionSequence(
        train_rows,
        config,
        batch_size=int(config["training"]["batch_size"]),
        training=True,
        seed=int(config["split"]["random_seed"]),
        noise_dir=noise_dir,
    )
    val_seq = AudioEmotionSequence(
        val_rows,
        config,
        batch_size=int(config["training"]["batch_size"]),
        training=False,
        seed=int(config["split"]["random_seed"]),
    )
    compile_model(model, learning_rate, float(config["training"]["label_smoothing"]))
    callbacks = [
        ValidationMacroF1(val_seq),
        EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=int(config["training"]["early_stopping_patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(str(model_path), monitor="val_macro_f1", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ]
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=int(epochs),
        callbacks=callbacks,
        verbose=1,
    )
    history_path = run_dir / ("%s_history.json" % stage_name)
    history_path.write_text(
        json.dumps({k: [float(v) for v in values] for k, values in history.history.items()}, indent=2),
        encoding="utf-8",
    )
    return history


def freeze_cnn_blocks(model, trainable_last_block=False):
    for layer in model.layers:
        if layer.name.startswith("cnn_block"):
            if trainable_last_block and layer.name.startswith("cnn_block3"):
                layer.trainable = True
            else:
                layer.trainable = False
        else:
            layer.trainable = True


def calibrate_thresholds(model, val_rows, config):
    seq = AudioEmotionSequence(
        val_rows,
        config,
        batch_size=int(config["training"]["batch_size"]),
        training=False,
        seed=int(config["split"]["random_seed"]),
    )
    y_true, probs_all = [], []
    for batch in range(len(seq)):
        x, y = seq[batch]
        probs = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=1).tolist())
        probs_all.extend(probs.tolist())
    probs_all = np.asarray(probs_all, dtype=np.float32)
    if probs_all.size == 0:
        return {"confidence_threshold": 0.60, "margin_threshold": 0.15, "val_macro_f1": 0.0}

    best = {"confidence_threshold": 0.60, "margin_threshold": 0.15, "val_macro_f1": -1.0}
    for conf in np.arange(0.40, 0.86, 0.05):
        for margin in np.arange(0.05, 0.31, 0.05):
            preds = []
            truth = []
            for idx, prob in enumerate(probs_all):
                order = np.argsort(prob)[::-1]
                if float(prob[order[0]]) >= conf and float(prob[order[0]] - prob[order[1]]) >= margin:
                    preds.append(int(order[0]))
                    truth.append(y_true[idx])
            if not preds:
                score = 0.0
            else:
                score = f1_score(truth, preds, labels=list(range(len(LABELS))), average="macro", zero_division=0)
            if score > best["val_macro_f1"]:
                best = {
                    "confidence_threshold": round(float(conf), 3),
                    "margin_threshold": round(float(margin), 3),
                    "val_macro_f1": float(score),
                    "coverage": float(len(preds)) / float(len(y_true)) if y_true else 0.0,
                }
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN-GRU Vietnamese speech emotion model.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "voice_emotion_config.yaml")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "audio_metadata.csv")
    parser.add_argument("--stage", choices=["stage1", "stage2", "all"], default="all")
    parser.add_argument("--noise-dir", type=Path, default=PROJECT_ROOT / "noise")
    parser.add_argument("--stage1-checkpoint", type=Path, default=PROJECT_ROOT / "models" / "voice_emotion_stage1_best.h5")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["split"]["random_seed"])
    set_seed(seed)

    run_dir = PROJECT_ROOT / "runs" / "voice_emotion" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path, test_path = ensure_splits(args.metadata, config)
    train_rows = filter_valid(read_metadata(train_path))
    val_rows = filter_valid(read_metadata(val_path))
    test_rows = filter_valid(read_metadata(test_path))

    if not train_rows or not val_rows:
        raise ValueError("Need non-empty train and validation metadata.")

    audio_cfg = config["audio"]
    time_steps = expected_time_steps(
        int(audio_cfg["sample_rate"]),
        float(audio_cfg["duration_seconds"]),
        int(audio_cfg["hop_length"]),
    )
    input_shape = (int(audio_cfg["n_mels"]), time_steps, 3)

    model = build_cnn_gru_ser(input_shape=input_shape, num_classes=len(LABELS))
    save_label_mapping(PROJECT_ROOT / "models" / "voice_emotion_label_mapping.pkl")

    with (run_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    metadata = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "random_seed": seed,
        "config": config,
        "train": rows_summary(train_rows),
        "val": rows_summary(val_rows),
        "test": rows_summary(test_rows),
        "input_shape": input_shape,
        "model_name": model.name,
        "parameter_count": int(model.count_params()),
    }
    save_json(run_dir / "training_metadata.json", metadata)

    stage1_path = PROJECT_ROOT / "models" / "voice_emotion_stage1_best.h5"
    final_path = PROJECT_ROOT / "models" / "voice_emotion_vietnamese_finetuned.h5"

    if args.stage in ("stage1", "all"):
        train_stage(
            model,
            train_rows,
            val_rows,
            config,
            run_dir,
            stage1_path,
            int(config["training"]["stage1_epochs"]),
            float(config["training"]["stage1_learning_rate"]),
            "stage1",
            noise_dir=args.noise_dir,
        )

    if args.stage in ("stage2", "all"):
        if args.stage == "stage2":
            model = tf.keras.models.load_model(str(args.stage1_checkpoint))
        stage2_train_rows = [
            row for row in train_rows
            if row.get("source") in ("vietnamese_kaggle", "self_recorded", "unknown")
        ]
        if not stage2_train_rows:
            stage2_train_rows = train_rows
        freeze_cnn_blocks(model, trainable_last_block=False)
        train_stage(
            model,
            stage2_train_rows,
            val_rows,
            config,
            run_dir,
            final_path,
            min(3, int(config["training"]["stage2_epochs"])),
            float(config["training"]["stage2_learning_rate"]),
            "stage2_frozen_cnn",
            noise_dir=args.noise_dir,
        )
        freeze_cnn_blocks(model, trainable_last_block=True)
        remaining_epochs = max(1, int(config["training"]["stage2_epochs"]) - 3)
        train_stage(
            model,
            stage2_train_rows,
            val_rows,
            config,
            run_dir,
            final_path,
            remaining_epochs,
            float(config["training"]["stage2_learning_rate"]),
            "stage2_unfreeze_last_cnn",
            noise_dir=args.noise_dir,
        )
        thresholds = calibrate_thresholds(model, val_rows, config)
        save_json(PROJECT_ROOT / "config" / "voice_emotion_thresholds.json", thresholds)
        save_json(run_dir / "voice_emotion_thresholds.json", thresholds)

    model.save(str(final_path if args.stage != "stage1" else stage1_path))
    print_model_report(model, final_path if args.stage != "stage1" else stage1_path)
    print("Run dir:", run_dir)
    print("Stage1 model:", stage1_path)
    print("Fine-tuned model:", final_path)


if __name__ == "__main__":
    main()
