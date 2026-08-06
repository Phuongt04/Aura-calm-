import gc
import json
import os
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    LSTM,
    MaxPooling2D,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# --- 1. Config ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_FOLDER = BASE_DIR / "dataset"
CV_OUTPUT_FOLDER = PROJECT_ROOT / "cv_audio_runs"
LABEL_PATH = PROJECT_ROOT / "label_mapping_final.pkl"
SOFT_LABEL_PATH = BASE_DIR / "label_mapping_final.pkl"
FINAL_MODEL_PATH = PROJECT_ROOT / "final_production_model.h5"
BEST_CV_MODEL_PATH = PROJECT_ROOT / "best_model_improved.h5"
CV_RESULTS_PATH = PROJECT_ROOT / "cross_validation_results.json"

IMG_SIZE = 64
N_FRAMES = 10
BATCH_SIZE = 32
MAX_EPOCHS = 100
NUM_FOLDS = 5
LEARNING_RATE = 1e-4
AUGMENT_TRAINING = True
AUGMENT_NOISE_MAX_STD = 0.025
AUGMENT_GAIN_MIN = 0.85
AUGMENT_GAIN_MAX = 1.15
AUGMENT_FREQ_MASK_MAX = 8
AUGMENT_TIME_MASK_MAX = 8
AUGMENT_ENGINE_NOISE_PROB = 0.55
AUGMENT_ENGINE_NOISE_MAX = 0.08
AUGMENT_BACKGROUND_NOISE_MAX = 0.035


# --- 2. Load data ---
def load_all_data():
    X, y = [], []

    if not DATA_FOLDER.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_FOLDER}")

    folders = [
        f for f in sorted(os.listdir(DATA_FOLDER))
        if (DATA_FOLDER / f).is_dir()
    ]
    print(f"Loading data from {len(folders)} classes...")

    if not folders:
        raise ValueError(f"No class folders found in {DATA_FOLDER}")

    for folder in folders:
        folder_path = DATA_FOLDER / folder
        files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".npy")]
        print(f"  {folder}: {len(files)} samples")

        for file in files:
            try:
                data = np.load(folder_path / file)
                if data.shape == (N_FRAMES, IMG_SIZE, IMG_SIZE, 1):
                    X.append(data)
                    y.append(folder)
                else:
                    print(f"    Skip {file}: wrong shape {data.shape}")
            except Exception as exc:
                print(f"    Error loading {file}: {exc}")

    X = np.array(X, dtype="float32")
    if len(X) == 0:
        raise ValueError("No valid training samples loaded.")

    if np.max(X) > 1.0:
        X = X / 255.0

    print(f"Loaded {len(X)} samples\n")
    return X, np.array(y)


def save_label_mapping(classes):
    with open(LABEL_PATH, "wb") as f:
        pickle.dump(classes, f)

    # Compatibility for older scripts that read soft/label_mapping_final.pkl.
    if SOFT_LABEL_PATH != LABEL_PATH:
        with open(SOFT_LABEL_PATH, "wb") as f:
            pickle.dump(classes, f)


def encode_labels(y_raw):
    class_names = np.array(sorted(set(y_raw)))
    class_to_index = {label: idx for idx, label in enumerate(class_names)}
    y_int = np.array([class_to_index[label] for label in y_raw], dtype="int32")
    return y_int, class_names


def stratified_kfold_indices(y_indices, n_splits, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]

    for class_id in np.unique(y_indices):
        class_indices = np.where(y_indices == class_id)[0]
        rng.shuffle(class_indices)
        class_parts = np.array_split(class_indices, n_splits)
        for fold_idx, part in enumerate(class_parts):
            folds[fold_idx].extend(part.tolist())

    all_indices = np.arange(len(y_indices))
    for fold in folds:
        val_idx = np.array(sorted(fold), dtype="int32")
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
        yield train_idx, val_idx


# --- 3. Model ---
def build_model(num_classes):
    model = Sequential([
        TimeDistributed(
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            input_shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, 1),
        ),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        TimeDistributed(Flatten()),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.4),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_cv_callbacks(model_path):
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def history_to_dict(history):
    return {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }


def compute_balanced_class_weights(y_indices, num_classes):
    counts = np.bincount(y_indices, minlength=num_classes).astype("float32")
    total = float(np.sum(counts))
    weights = np.ones(num_classes, dtype="float32")

    present = counts > 0
    weights[present] = total / (float(np.sum(present)) * counts[present])
    return weights


def _random_axis_mask(x, axis, max_width):
    width = tf.random.uniform([], minval=1, maxval=max_width + 1, dtype=tf.int32)
    start = tf.random.uniform([], minval=0, maxval=IMG_SIZE - max_width + 1, dtype=tf.int32)
    positions = tf.range(IMG_SIZE)
    keep = tf.logical_or(positions < start, positions >= start + width)
    keep = tf.cast(keep, x.dtype)

    if axis == 1:
        mask = tf.reshape(keep, [1, IMG_SIZE, 1, 1])
    else:
        mask = tf.reshape(keep, [1, 1, IMG_SIZE, 1])
    return x * mask


def _add_engine_like_noise(x):
    """Add low-frequency band noise that mimics motor/fan rumble in mel space."""
    band_height = tf.random.uniform([], minval=4, maxval=18, dtype=tf.int32)
    band_gain = tf.random.uniform([], 0.0, AUGMENT_ENGINE_NOISE_MAX)
    row_positions = tf.range(IMG_SIZE, dtype=tf.float32)
    band = tf.cast(tf.range(IMG_SIZE) < band_height, tf.float32)
    band = band * tf.exp(-row_positions / tf.cast(tf.maximum(band_height, 1), tf.float32))
    band = tf.reshape(band, [1, IMG_SIZE, 1, 1])

    time_variation = tf.random.uniform([N_FRAMES, 1, IMG_SIZE, 1], 0.65, 1.35)
    return x + band_gain * band * time_variation


def augment_spectrogram(x):
    x = tf.cast(x, tf.float32)

    gain = tf.random.uniform([], AUGMENT_GAIN_MIN, AUGMENT_GAIN_MAX)
    x = x * gain

    noise_std = tf.random.uniform([], 0.0, AUGMENT_NOISE_MAX_STD)
    x = x + tf.random.normal(tf.shape(x), stddev=noise_std)

    background_noise = tf.random.uniform([], 0.0, AUGMENT_BACKGROUND_NOISE_MAX)
    x = x + background_noise * tf.random.uniform(tf.shape(x), minval=0.0, maxval=1.0)

    x = tf.cond(
        tf.random.uniform([]) < AUGMENT_ENGINE_NOISE_PROB,
        lambda: _add_engine_like_noise(x),
        lambda: x,
    )

    frame_shift = tf.random.uniform([], minval=-1, maxval=2, dtype=tf.int32)
    x = tf.roll(x, shift=frame_shift, axis=0)

    x = tf.cond(
        tf.random.uniform([]) < 0.6,
        lambda: _random_axis_mask(x, axis=1, max_width=AUGMENT_FREQ_MASK_MAX),
        lambda: x,
    )
    x = tf.cond(
        tf.random.uniform([]) < 0.6,
        lambda: _random_axis_mask(x, axis=2, max_width=AUGMENT_TIME_MASK_MAX),
        lambda: x,
    )

    return tf.clip_by_value(x, 0.0, 1.0)


def augment_dataset_item(x, y, sample_weight):
    return augment_spectrogram(x), y, sample_weight


def make_dataset(X_values, y_values, training=False, class_weights=None, y_indices=None):
    if training:
        if class_weights is None or y_indices is None:
            raise ValueError("Training dataset needs class_weights and y_indices.")
        sample_weights = class_weights[y_indices].astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((X_values, y_values, sample_weights))
        dataset = dataset.shuffle(
            buffer_size=len(X_values),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        if AUGMENT_TRAINING:
            dataset = dataset.map(
                augment_dataset_item,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X_values, y_values))

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def train_one_fold(fold_idx, train_idx, val_idx, X, y_final, y_int, class_names):
    tf.keras.backend.clear_session()

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_final[train_idx], y_final[val_idx]
    y_train_int = y_int[train_idx]
    class_weights = compute_balanced_class_weights(y_train_int, len(class_names))
    train_dataset = make_dataset(
        X_train,
        y_train,
        training=True,
        class_weights=class_weights,
        y_indices=y_train_int,
    )
    val_dataset = make_dataset(X_val, y_val, training=False)

    fold_dir = CV_OUTPUT_FOLDER / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = fold_dir / f"best_model_fold_{fold_idx}.h5"
    history_path = fold_dir / "training_history.json"

    print("\n" + "=" * 72)
    print(f"Fold {fold_idx}: train={len(X_train)} | val={len(X_val)}")
    print(
        "Class weights:",
        {str(class_names[i]): round(float(class_weights[i]), 3) for i in range(len(class_names))},
    )
    print("=" * 72)

    model = build_model(num_classes=len(class_names))
    if fold_idx == 1:
        model.summary()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        callbacks=make_cv_callbacks(best_model_path),
        verbose=1,
    )

    best_val_acc = max(history.history["val_accuracy"])
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1
    best_val_loss = min(history.history["val_loss"])

    history_dict = history_to_dict(history)
    history_dict.update({
        "fold": fold_idx,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "augmentation_enabled": bool(AUGMENT_TRAINING),
        "class_weights": {
            str(class_names[i]): float(class_weights[i]) for i in range(len(class_names))
        },
        "classes": [str(c) for c in class_names],
    })
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2, ensure_ascii=False)

    # Keep a root-level copy for older scripts that point to best_model_fold_1.h5.
    if fold_idx == 1:
        shutil.copy2(best_model_path, PROJECT_ROOT / "best_model_fold_1.h5")

    result = {
        "fold": fold_idx,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "best_model_path": str(best_model_path),
        "augmentation_enabled": bool(AUGMENT_TRAINING),
        "class_weights": {
            str(class_names[i]): float(class_weights[i]) for i in range(len(class_names))
        },
    }

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return result


def train_final_model(X, y_final, y_int, class_names, final_epochs):
    tf.keras.backend.clear_session()
    class_weights = compute_balanced_class_weights(y_int, len(class_names))
    train_dataset = make_dataset(
        X,
        y_final,
        training=True,
        class_weights=class_weights,
        y_indices=y_int,
    )

    print("\n" + "=" * 72)
    print(f"Training final production model on all {len(X)} samples")
    print(f"Epochs: {final_epochs} (median best epoch from CV)")
    print(
        "Class weights:",
        {str(class_names[i]): round(float(class_weights[i]), 3) for i in range(len(class_names))},
    )
    print("=" * 72)

    model = build_model(num_classes=len(class_names))
    history = model.fit(
        train_dataset,
        epochs=final_epochs,
        verbose=1,
    )

    model.save(str(FINAL_MODEL_PATH))

    history_dict = history_to_dict(history)
    history_dict.update({
        "train_samples": int(len(X)),
        "epochs": int(final_epochs),
        "augmentation_enabled": bool(AUGMENT_TRAINING),
        "class_weights": {
            str(class_names[i]): float(class_weights[i]) for i in range(len(class_names))
        },
        "classes": [str(c) for c in class_names],
    })
    with open(PROJECT_ROOT / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2, ensure_ascii=False)

    del model
    tf.keras.backend.clear_session()
    gc.collect()


def main():
    X, y_raw = load_all_data()

    y_int, class_names = encode_labels(y_raw)
    y_final = to_categorical(y_int)
    save_label_mapping(class_names)

    counts = np.bincount(y_int)
    class_counts = {
        str(class_names[i]): int(counts[i])
        for i in range(len(class_names))
    }
    print("Class counts:", class_counts)

    min_class_count = int(counts.min())
    n_splits = min(NUM_FOLDS, min_class_count)
    if n_splits < 2:
        raise ValueError(
            "Cross-validation needs at least 2 samples in every class. "
            f"Current class counts: {class_counts}"
        )
    if n_splits != NUM_FOLDS:
        print(
            f"Only {min_class_count} samples in the smallest class; "
            f"using {n_splits}-fold instead of {NUM_FOLDS}-fold."
        )

    CV_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    fold_results = []
    split_indices = stratified_kfold_indices(y_int, n_splits=n_splits, seed=SEED)
    for fold_idx, (train_idx, val_idx) in enumerate(split_indices, start=1):
        result = train_one_fold(
            fold_idx=fold_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            X=X,
            y_final=y_final,
            y_int=y_int,
            class_names=class_names,
        )
        fold_results.append(result)

    val_accuracies = [r["best_val_accuracy"] for r in fold_results]
    val_losses = [r["best_val_loss"] for r in fold_results]
    best_epochs = [r["best_epoch"] for r in fold_results]
    final_epochs = max(1, int(np.median(best_epochs)))
    best_fold = max(fold_results, key=lambda r: r["best_val_accuracy"])
    shutil.copy2(best_fold["best_model_path"], BEST_CV_MODEL_PATH)

    cv_summary = {
        "dataset_folder": str(DATA_FOLDER),
        "num_samples": int(len(X)),
        "num_classes": int(len(class_names)),
        "classes": [str(c) for c in class_names],
        "class_counts": class_counts,
        "num_folds": int(n_splits),
        "folds": fold_results,
        "mean_val_accuracy": float(np.mean(val_accuracies)),
        "std_val_accuracy": float(np.std(val_accuracies)),
        "mean_val_loss": float(np.mean(val_losses)),
        "std_val_loss": float(np.std(val_losses)),
        "median_best_epoch": int(final_epochs),
        "best_cv_model_path": str(BEST_CV_MODEL_PATH),
        "best_cv_fold": int(best_fold["fold"]),
        "augmentation_enabled": bool(AUGMENT_TRAINING),
        "augmentation": {
            "noise_max_std": float(AUGMENT_NOISE_MAX_STD),
            "gain_min": float(AUGMENT_GAIN_MIN),
            "gain_max": float(AUGMENT_GAIN_MAX),
            "freq_mask_max": int(AUGMENT_FREQ_MASK_MAX),
            "time_mask_max": int(AUGMENT_TIME_MASK_MAX),
            "engine_noise_prob": float(AUGMENT_ENGINE_NOISE_PROB),
            "engine_noise_max": float(AUGMENT_ENGINE_NOISE_MAX),
            "background_noise_max": float(AUGMENT_BACKGROUND_NOISE_MAX),
        },
    }

    with open(CV_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("Cross-validation summary")
    print("=" * 72)
    for result in fold_results:
        print(
            f"Fold {result['fold']}: "
            f"best val acc={result['best_val_accuracy']:.4f}, "
            f"best val loss={result['best_val_loss']:.4f}, "
            f"best epoch={result['best_epoch']}"
        )
    print(
        f"Mean val acc: {cv_summary['mean_val_accuracy']:.4f} "
        f"+/- {cv_summary['std_val_accuracy']:.4f}"
    )
    print(f"Best CV model: {BEST_CV_MODEL_PATH} (fold {best_fold['fold']})")
    print(f"Results saved to: {CV_RESULTS_PATH}")

    train_final_model(X, y_final, y_int, class_names, final_epochs)

    print("\nDone.")
    print(f"Final model: {FINAL_MODEL_PATH}")
    print(f"Label mapping: {LABEL_PATH}")


if __name__ == "__main__":
    main()
