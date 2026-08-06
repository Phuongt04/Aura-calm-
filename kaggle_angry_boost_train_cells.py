# %% [markdown]
# # Angry-Boost Speech Emotion Training
#
# Copy these cells into a Kaggle notebook, or upload this `.py` file and open it
# as notebook cells. The goal is to improve the `angry` class without adding new
# raw angry recordings:
#
# - label order is fixed as: `neutral=0, happy=1, sad=2, angry=3`
# - angry train samples are oversampled
# - angry samples receive stronger sample weight
# - angry gets light waveform/spec augmentation during training
# - best checkpoint is selected by macro-F1 plus angry recall
# - an optional validation-tuned happy-to-angry postprocess rule is exported

# %%
import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    import librosa
except Exception as exc:
    raise RuntimeError("This notebook needs librosa. Kaggle usually has it preinstalled.") from exc

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

LABELS = ["neutral", "happy", "sad", "angry"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
ANGRY_INDEX = LABEL_TO_INDEX["angry"]
HAPPY_INDEX = LABEL_TO_INDEX["happy"]

SAMPLE_RATE = 16000
DURATION_SECONDS = 4.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 64
TIME_STEPS = 1 + N_SAMPLES // HOP_LENGTH  # 401 when librosa center=True.

WORK_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input")
RUN_DIR = WORK_DIR / f"angry_boost_run_{time.strftime('%Y%m%d_%H%M%S')}"
MODEL_DIR = RUN_DIR / "models"
REPORT_DIR = RUN_DIR / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("TensorFlow:", tf.__version__)
print("Run dir:", RUN_DIR)
print("Label order:", LABELS)

# %% [markdown]
# ## 1. Find metadata and audio files
#
# This cell first looks for `train_metadata.csv`, `val_metadata.csv`, and
# `test_metadata.csv` under `/kaggle/input`. If CSV file paths inside metadata do
# not exist on this Kaggle runtime, the cell remaps them by audio basename.

# %%
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def find_metadata_file(name):
    matches = sorted(INPUT_DIR.rglob(name))
    if not matches:
        return None
    return matches[0]


def index_audio_files(root):
    by_name = defaultdict(list)
    for path in root.rglob("*"):
        if path.suffix.lower() in AUDIO_SUFFIXES:
            by_name[path.name].append(path)
    return by_name


def remap_audio_paths(rows, audio_index):
    fixed = []
    missing = []
    for row in rows:
        row = dict(row)
        raw = row.get("filepath") or row.get("audio_path") or row.get("path") or row.get("file")
        if not raw:
            missing.append(row)
            continue
        path = Path(raw)
        if path.exists():
            row["filepath"] = str(path)
            fixed.append(row)
            continue
        candidates = audio_index.get(path.name, [])
        if candidates:
            row["filepath"] = str(candidates[0])
            fixed.append(row)
        else:
            missing.append(row)
    return fixed, missing


def filter_rows(rows):
    out = []
    for row in rows:
        label = str(row.get("label", "")).strip().lower()
        if label in LABEL_TO_INDEX and str(row.get("is_valid", "true")).lower() in ("true", "1", "yes"):
            row = dict(row)
            row["label"] = label
            out.append(row)
    return out


audio_index = index_audio_files(INPUT_DIR)
print("Audio files indexed:", sum(len(v) for v in audio_index.values()))

train_meta = find_metadata_file("train_metadata.csv")
val_meta = find_metadata_file("val_metadata.csv")
test_meta = find_metadata_file("test_metadata.csv")

if train_meta is None or val_meta is None or test_meta is None:
    raise FileNotFoundError(
        "Could not find train_metadata.csv, val_metadata.csv, and test_metadata.csv under /kaggle/input. "
        "Upload the voice-emotion-data or voice-emotion-code dataset that contains these CSVs."
    )

train_rows, train_missing = remap_audio_paths(filter_rows(read_csv_rows(train_meta)), audio_index)
val_rows, val_missing = remap_audio_paths(filter_rows(read_csv_rows(val_meta)), audio_index)
test_rows, test_missing = remap_audio_paths(filter_rows(read_csv_rows(test_meta)), audio_index)

print("Metadata:")
print(" train:", train_meta)
print(" val:  ", val_meta)
print(" test: ", test_meta)
print("Rows:", len(train_rows), len(val_rows), len(test_rows))
print("Missing paths:", len(train_missing), len(val_missing), len(test_missing))
print("Train counts:", Counter(r["label"] for r in train_rows))
print("Val counts:  ", Counter(r["label"] for r in val_rows))
print("Test counts: ", Counter(r["label"] for r in test_rows))

assert train_rows and val_rows and test_rows, "Need non-empty train, val, and test rows."

# %% [markdown]
# ## 2. Feature extraction and angry-focused augmentation

# %%
def prevent_clipping(y, peak=0.99):
    y = np.asarray(y, dtype=np.float32)
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs > peak:
        y = y / max_abs * peak
    return y.astype(np.float32)


def pad_or_crop(y, n_samples=N_SAMPLES):
    y = np.asarray(y, dtype=np.float32)
    if len(y) < n_samples:
        pad = n_samples - len(y)
        left = pad // 2
        return np.pad(y, (left, pad - left), mode="constant").astype(np.float32)
    if len(y) > n_samples:
        start = (len(y) - n_samples) // 2
        return y[start:start + n_samples].astype(np.float32)
    return y.astype(np.float32)


def load_audio_4s(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    y = prevent_clipping(y)
    return pad_or_crop(y)


def add_noise_snr(y, rng, snr_min=18.0, snr_max=30.0):
    y = np.asarray(y, dtype=np.float32)
    noise = rng.normal(0, 1, size=y.shape).astype(np.float32)
    signal_power = float(np.mean(y * y))
    noise_power = float(np.mean(noise * noise))
    if signal_power <= 1e-10 or noise_power <= 1e-10:
        return y
    snr_db = float(rng.uniform(snr_min, snr_max))
    desired_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return (y + noise * math.sqrt(desired_noise_power / noise_power)).astype(np.float32)


def augment_audio(y, rng, label):
    y = np.asarray(y, dtype=np.float32)
    is_angry = label == "angry"

    # Keep transforms light. Too much augmentation can erase emotional cues.
    gain_min, gain_max = (0.75, 1.30) if is_angry else (0.85, 1.15)
    if rng.rand() < (0.95 if is_angry else 0.70):
        y = y * float(rng.uniform(gain_min, gain_max))

    shift_ms = 250 if is_angry else 150
    if rng.rand() < (0.75 if is_angry else 0.45):
        max_shift = int(SAMPLE_RATE * shift_ms / 1000.0)
        y = np.roll(y, int(rng.randint(-max_shift, max_shift + 1))).astype(np.float32)

    if rng.rand() < (0.45 if is_angry else 0.25):
        rate = float(rng.uniform(0.94 if is_angry else 0.97, 1.06 if is_angry else 1.03))
        y = librosa.effects.time_stretch(y, rate=rate)
        y = pad_or_crop(y)

    if rng.rand() < (0.45 if is_angry else 0.20):
        steps = float(rng.uniform(-1.0 if is_angry else -0.5, 1.0 if is_angry else 0.5))
        y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=steps)

    if rng.rand() < (0.55 if is_angry else 0.35):
        y = add_noise_snr(y, rng)

    return prevent_clipping(pad_or_crop(y))


def log_mel_delta(y):
    mel = librosa.feature.melspectrogram(
        y=np.asarray(y, dtype=np.float32),
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=50,
        fmax=8000,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    delta = librosa.feature.delta(log_mel, width=9).astype(np.float32)
    delta2 = librosa.feature.delta(log_mel, width=9, order=2).astype(np.float32)
    x = np.stack([log_mel, delta, delta2], axis=-1)

    if x.shape[1] < TIME_STEPS:
        pad = TIME_STEPS - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad), (0, 0)), mode="edge")
    elif x.shape[1] > TIME_STEPS:
        x = x[:, :TIME_STEPS, :]

    mean = float(np.mean(x))
    std = float(np.std(x))
    if std > 1e-6:
        x = (x - mean) / std
    else:
        x = x - mean
    return x.astype(np.float32)


def spec_augment(x, rng, label):
    x = np.array(x, copy=True, dtype=np.float32)
    is_angry = label == "angry"
    freq_max = 8 if is_angry else 5
    time_max = 20 if is_angry else 12
    if rng.rand() < (0.65 if is_angry else 0.45):
        width = int(rng.randint(1, min(freq_max, x.shape[0]) + 1))
        start = int(rng.randint(0, x.shape[0] - width + 1))
        x[start:start + width, :, :] = 0.0
    if rng.rand() < (0.65 if is_angry else 0.45):
        width = int(rng.randint(1, min(time_max, x.shape[1]) + 1))
        start = int(rng.randint(0, x.shape[1] - width + 1))
        x[:, start:start + width, :] = 0.0
    return x.astype(np.float32)

# %% [markdown]
# ## 3. Oversampled training sequence and sample weights
#
# Tune these three constants first if angry is still weak:
#
# - `ANGRY_OVERSAMPLE = 3`
# - `ANGRY_SAMPLE_WEIGHT = 2.5`
# - `ANGRY_SCORE_BONUS = 0.35`

# %%
BATCH_SIZE = 16
EPOCHS = 45
ANGRY_OVERSAMPLE = 3
ANGRY_SAMPLE_WEIGHT = 2.5
ANGRY_SCORE_BONUS = 0.35
USE_FOCAL_LOSS = False


def make_sample_weights(rows):
    counts = Counter(row["label"] for row in rows)
    total = float(sum(counts.values()))
    weights = {}
    for label in LABELS:
        base = total / (len(LABELS) * float(max(1, counts[label])))
        weights[label] = float(base)
    weights["angry"] = max(weights["angry"], ANGRY_SAMPLE_WEIGHT)
    return weights


class AngryBoostSequence(keras.utils.Sequence):
    def __init__(self, rows, batch_size, training, seed=42):
        super().__init__()
        self.base_rows = list(rows)
        self.batch_size = int(batch_size)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.sample_weights = make_sample_weights(self.base_rows)
        self.rows = []
        self.on_epoch_end()

    def __len__(self):
        return max(1, int(np.ceil(len(self.rows) / self.batch_size)))

    def on_epoch_end(self):
        rows = list(self.base_rows)
        if self.training:
            angry = [r for r in self.base_rows if r["label"] == "angry"]
            rows.extend(angry * max(0, ANGRY_OVERSAMPLE - 1))
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(rows)
        self.rows = rows
        self.epoch += 1

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed + self.epoch * 100000 + index)
        batch = self.rows[index * self.batch_size:(index + 1) * self.batch_size]
        xs, ys, sw = [], [], []
        for row in batch:
            label = row["label"]
            y = load_audio_4s(row["filepath"])
            if self.training:
                y = augment_audio(y, rng, label)
            x = log_mel_delta(y)
            if self.training:
                x = spec_augment(x, rng, label)
            xs.append(x)
            ys.append(LABEL_TO_INDEX[label])
            sw.append(self.sample_weights[label])
        return (
            np.stack(xs, axis=0).astype(np.float32),
            keras.utils.to_categorical(np.asarray(ys), num_classes=len(LABELS)),
            np.asarray(sw, dtype=np.float32),
        )


train_seq = AngryBoostSequence(train_rows, BATCH_SIZE, training=True, seed=SEED)
val_seq = AngryBoostSequence(val_rows, BATCH_SIZE, training=False, seed=SEED)
test_seq = AngryBoostSequence(test_rows, BATCH_SIZE, training=False, seed=SEED)

print("Base train counts:", Counter(r["label"] for r in train_rows))
print("After angry oversample:", Counter(r["label"] for r in train_seq.rows))
print("Sample weights:", train_seq.sample_weights)

# %% [markdown]
# ## 4. Build or load CNN-GRU model
#
# If a compatible `.h5` / `.keras` model exists under `/kaggle/input`, the cell
# will load it and fine-tune. Otherwise it trains the same CNN-GRU architecture
# from scratch.

# %%
def build_cnn_gru_ser(input_shape=(64, 401, 3), num_classes=4):
    inputs = layers.Input(shape=input_shape, name="logmel_delta_input")
    x = layers.SeparableConv2D(32, (3, 3), padding="same", name="cnn_block1_sepconv")(inputs)
    x = layers.BatchNormalization(name="cnn_block1_bn")(x)
    x = layers.ReLU(name="cnn_block1_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_block1_pool")(x)

    x = layers.SeparableConv2D(64, (3, 3), padding="same", name="cnn_block2_sepconv")(x)
    x = layers.BatchNormalization(name="cnn_block2_bn")(x)
    x = layers.ReLU(name="cnn_block2_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_block2_pool")(x)

    x = layers.SeparableConv2D(128, (3, 3), padding="same", name="cnn_block3_sepconv")(x)
    x = layers.BatchNormalization(name="cnn_block3_bn")(x)
    x = layers.ReLU(name="cnn_block3_relu")(x)

    x = layers.Permute((2, 1, 3), name="time_major")(x)
    x = layers.Reshape((TIME_STEPS // 4, (N_MELS // 4) * 128), name="time_sequence")(x)
    x = layers.GRU(64, name="gru_64")(x)
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    x = layers.Dropout(0.35, name="dropout_035")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_softmax")(x)
    return keras.Model(inputs, outputs, name="CNN_GRU_Vietnamese_SER")


def find_pretrained_model():
    preferred_names = [
        "voice_emotion_stage1_best.h5",
        "voice_emotion_vietnamese_finetuned.h5",
        "cnn_gru_window_encoder_source_lr1e-5.h5",
        "voice_emotion_stage1_best.keras",
    ]
    for name in preferred_names:
        matches = sorted(INPUT_DIR.rglob(name))
        if matches:
            return matches[0]
    matches = sorted([p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in {".h5", ".keras"}])
    return matches[0] if matches else None


pretrained_path = find_pretrained_model()
if pretrained_path:
    print("Loading pretrained model:", pretrained_path)
    try:
        model = keras.models.load_model(pretrained_path, compile=False)
        if tuple(model.input_shape[1:]) != (N_MELS, TIME_STEPS, 3) or tuple(model.output_shape[1:]) != (len(LABELS),):
            print("Pretrained model shape is not CNN-GRU audio shape:", model.input_shape, model.output_shape)
            print("Building CNN-GRU from scratch instead.")
            model = build_cnn_gru_ser((N_MELS, TIME_STEPS, 3), len(LABELS))
    except Exception as exc:
        print("Could not load pretrained model:", repr(exc))
        print("Building CNN-GRU from scratch instead.")
        model = build_cnn_gru_ser((N_MELS, TIME_STEPS, 3), len(LABELS))
else:
    print("No pretrained CNN-GRU found. Training from scratch.")
    model = build_cnn_gru_ser((N_MELS, TIME_STEPS, 3), len(LABELS))

model.summary()

# %% [markdown]
# ## 5. Loss, validation metric, and angry-aware checkpoint

# %%
def categorical_focal_loss(gamma=2.0, alpha=(1.0, 1.0, 1.0, 1.8)):
    alpha = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        focal = tf.pow(1.0 - y_pred, gamma) * ce
        return tf.reduce_sum(alpha * focal, axis=-1)

    return loss_fn


def collect_probs(seq, model):
    y_true = []
    y_prob = []
    for i in range(len(seq)):
        x, y, _ = seq[i]
        probs = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=1).tolist())
        y_prob.extend(probs.tolist())
    return np.asarray(y_true, dtype=np.int32), np.asarray(y_prob, dtype=np.float32)


class AngryAwareCheckpoint(keras.callbacks.Callback):
    def __init__(self, val_sequence, model_path, patience=10):
        super().__init__()
        self.val_sequence = val_sequence
        self.model_path = Path(model_path)
        self.patience = int(patience)
        self.best_score = -1e9
        self.wait = 0
        self.best = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true, y_prob = collect_probs(self.val_sequence, self.model)
        y_pred = np.argmax(y_prob, axis=1)
        macro_f1 = f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)
        recall = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0
        )[1]
        angry_recall = float(recall[ANGRY_INDEX])
        score = float(macro_f1 + ANGRY_SCORE_BONUS * angry_recall)
        logs["val_macro_f1"] = float(macro_f1)
        logs["val_angry_recall"] = angry_recall
        logs["val_angry_score"] = score
        print(f" - val_macro_f1: {macro_f1:.4f} - val_angry_recall: {angry_recall:.4f} - angry_score: {score:.4f}")

        if score > self.best_score:
            self.best_score = score
            self.wait = 0
            self.best = {
                "epoch": int(epoch),
                "val_macro_f1": float(macro_f1),
                "val_angry_recall": angry_recall,
                "val_angry_score": score,
            }
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.model_path)
            (REPORT_DIR / "best_val_metrics.json").write_text(json.dumps(self.best, indent=2), encoding="utf-8")
            print("   saved:", self.model_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping: angry_score did not improve for {self.patience} epochs.")
                self.model.stop_training = True


loss = categorical_focal_loss() if USE_FOCAL_LOSS else keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
    loss=loss,
    metrics=["accuracy"],
)

best_model_path = MODEL_DIR / "cnn_gru_angry_boost_best.keras"
callbacks = [
    AngryAwareCheckpoint(val_seq, best_model_path, patience=10),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
]

# %% [markdown]
# ## 6. Train

# %%
history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

history_df = pd.DataFrame(history.history)
history_df.to_csv(REPORT_DIR / "training_history.csv", index=False)
print("Saved history:", REPORT_DIR / "training_history.csv")

# %% [markdown]
# ## 7. Evaluate best checkpoint on validation and test

# %%
best_model = keras.models.load_model(best_model_path, compile=False)


def evaluate_and_save(seq, rows, split_name, model):
    y_true, y_prob = collect_probs(seq, model)
    y_pred = np.argmax(y_prob, axis=1)
    report = classification_report(y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0
    )
    summary = {
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "angry_recall": float(recall[ANGRY_INDEX]),
        "class_order": LABELS,
        "confusion_matrix": cm.tolist(),
    }

    pred_rows = []
    for row, true_idx, pred_idx, probs in zip(rows, y_true, y_pred, y_prob):
        pred_rows.append({
            "filepath": row["filepath"],
            "true_label": LABELS[int(true_idx)],
            "predicted_label": LABELS[int(pred_idx)],
            "q": float(np.max(probs)),
            "p_neutral": float(probs[LABEL_TO_INDEX["neutral"]]),
            "p_happy": float(probs[LABEL_TO_INDEX["happy"]]),
            "p_sad": float(probs[LABEL_TO_INDEX["sad"]]),
            "p_angry": float(probs[LABEL_TO_INDEX["angry"]]),
            "correct": bool(true_idx == pred_idx),
        })

    pd.DataFrame(pred_rows).to_csv(REPORT_DIR / f"{split_name}_predictions.csv", index=False)
    pd.DataFrame(cm, index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS]).to_csv(
        REPORT_DIR / f"{split_name}_confusion_matrix.csv"
    )
    (REPORT_DIR / f"{split_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / f"{split_name}_classification_report.txt").write_text(report, encoding="utf-8")
    print("\n", split_name.upper())
    print(json.dumps(summary, indent=2))
    print(report)
    return y_true, y_prob, summary


val_true, val_prob, val_summary = evaluate_and_save(val_seq, val_rows, "val", best_model)
test_true, test_prob, test_summary = evaluate_and_save(test_seq, test_rows, "test", best_model)

# %% [markdown]
# ## 8. Optional: tune a happy-to-angry postprocess rule on validation
#
# Use this only if false-negative angry is more harmful than a few extra angry
# predictions. The rule is:
#
# `if predicted == happy and p_angry >= min_p_angry and p_happy - p_angry <= margin: predict angry`

# %%
def apply_angry_rule(probs, min_p_angry, happy_angry_margin):
    preds = np.argmax(probs, axis=1)
    adjusted = preds.copy()
    for i, prob in enumerate(probs):
        if preds[i] == HAPPY_INDEX:
            if float(prob[ANGRY_INDEX]) >= min_p_angry and float(prob[HAPPY_INDEX] - prob[ANGRY_INDEX]) <= happy_angry_margin:
                adjusted[i] = ANGRY_INDEX
    return adjusted


best_rule = None
for min_p in np.arange(0.15, 0.56, 0.05):
    for margin in np.arange(0.00, 0.31, 0.05):
        pred = apply_angry_rule(val_prob, float(min_p), float(margin))
        macro = f1_score(val_true, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)
        recall = precision_recall_fscore_support(val_true, pred, labels=list(range(len(LABELS))), zero_division=0)[1]
        angry_recall = float(recall[ANGRY_INDEX])
        score = float(macro + ANGRY_SCORE_BONUS * angry_recall)
        item = {
            "min_p_angry": round(float(min_p), 3),
            "happy_angry_margin": round(float(margin), 3),
            "val_macro_f1": float(macro),
            "val_angry_recall": angry_recall,
            "val_angry_score": score,
        }
        if best_rule is None or item["val_angry_score"] > best_rule["val_angry_score"]:
            best_rule = item

rule_path = REPORT_DIR / "angry_postprocess_rule.json"
rule_path.write_text(json.dumps(best_rule, indent=2), encoding="utf-8")
print("Best validation rule:", json.dumps(best_rule, indent=2))

test_rule_pred = apply_angry_rule(test_prob, best_rule["min_p_angry"], best_rule["happy_angry_margin"])
test_rule_macro = f1_score(test_true, test_rule_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)
test_rule_recall = precision_recall_fscore_support(test_true, test_rule_pred, labels=list(range(len(LABELS))), zero_division=0)[1]
print("Test with angry rule:")
print(" accuracy:", accuracy_score(test_true, test_rule_pred))
print(" macro_f1:", test_rule_macro)
print(" angry_recall:", float(test_rule_recall[ANGRY_INDEX]))
print(pd.DataFrame(
    confusion_matrix(test_true, test_rule_pred, labels=list(range(len(LABELS)))),
    index=[f"true_{x}" for x in LABELS],
    columns=[f"pred_{x}" for x in LABELS],
))

# %% [markdown]
# ## 9. Zip outputs for download

# %%
import shutil

zip_base = str(WORK_DIR / "angry_boost_outputs")
zip_path = shutil.make_archive(zip_base, "zip", RUN_DIR)
print("Best model:", best_model_path)
print("Reports:", REPORT_DIR)
print("Download zip:", zip_path)
