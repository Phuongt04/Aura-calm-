import argparse
import csv
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import tensorflow as tf
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except Exception:
    def confusion_matrix(y_true, y_pred, labels=None):
        labels = labels or sorted(set(list(y_true) + list(y_pred)))
        index = {label: idx for idx, label in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for true, pred in zip(y_true, y_pred):
            if true in index and pred in index:
                cm[index[true], index[pred]] += 1
        return cm

    def accuracy_score(y_true, y_pred):
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    def precision_recall_fscore_support(y_true, y_pred, labels=None, zero_division=0):
        labels = labels or sorted(set(list(y_true) + list(y_pred)))
        precision, recall, f1, support = [], [], [], []
        for label in labels:
            tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
            fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
            fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
            sup = sum(1 for a in y_true if a == label)
            p = tp / float(tp + fp) if (tp + fp) else float(zero_division)
            r = tp / float(tp + fn) if (tp + fn) else float(zero_division)
            score = 2.0 * p * r / (p + r) if (p + r) else float(zero_division)
            precision.append(p)
            recall.append(r)
            f1.append(score)
            support.append(sup)
        return np.asarray(precision), np.asarray(recall), np.asarray(f1), np.asarray(support)

    def balanced_accuracy_score(y_true, y_pred):
        labels = sorted(set(list(y_true) + list(y_pred)))
        _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        return float(np.mean(recall)) if len(recall) else 0.0

    def classification_report(y_true, y_pred, labels=None, target_names=None, output_dict=True, zero_division=0):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=zero_division
        )
        names = target_names or [str(label) for label in labels]
        report = {}
        for idx, name in enumerate(names):
            report[name] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1-score": float(f1[idx]),
                "support": int(support[idx]),
            }
        report["macro avg"] = {
            "precision": float(np.mean(precision)) if len(precision) else 0.0,
            "recall": float(np.mean(recall)) if len(recall) else 0.0,
            "f1-score": float(np.mean(f1)) if len(f1) else 0.0,
            "support": int(np.sum(support)) if len(support) else 0,
        }
        return report

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_preprocessing import preprocess_file
from src.config_utils import load_config
from src.voice_emotion_labels import LABELS, LABEL_TO_INDEX


def read_metadata(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row.get("label") in LABEL_TO_INDEX
            and str(row.get("is_valid", "true")).lower() in ("true", "1", "yes")
        ]


def batches(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def predict_rows(model, rows, config, batch_size):
    y_true = []
    y_prob = []
    preprocessing_ms = []
    inference_ms = []
    for chunk in batches(rows, batch_size):
        xs = []
        for row in chunk:
            t0 = time.time()
            xs.append(preprocess_file(row["filepath"], config))
            preprocessing_ms.append((time.time() - t0) * 1000.0)
            y_true.append(LABEL_TO_INDEX[row["label"]])
        x = np.stack(xs, axis=0).astype(np.float32)
        t1 = time.time()
        probs = model.predict(x, verbose=0)
        batch_ms = (time.time() - t1) * 1000.0
        inference_ms.extend([batch_ms / float(len(chunk))] * len(chunk))
        y_prob.extend(probs.tolist())
    return np.asarray(y_true, dtype=np.int32), np.asarray(y_prob, dtype=np.float32), preprocessing_ms, inference_ms


def compute_metrics(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def plot_confusion_matrix(cm, output_path, title):
    if plt is None:
        write_simple_png_confusion(cm, output_path)
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def _png_chunk(chunk_type, data):
    return (
        struct.pack("!I", len(data))
        + chunk_type
        + data
        + struct.pack("!I", zlib.crc32(chunk_type + data) & 0xffffffff)
    )


def write_simple_png_confusion(cm, output_path):
    cm = np.asarray(cm, dtype=np.float32)
    cell = 48
    n = cm.shape[0]
    width = max(1, n * cell)
    height = max(1, n * cell)
    max_value = float(np.max(cm)) if cm.size else 1.0
    if max_value <= 0:
        max_value = 1.0
    rows = []
    for y in range(height):
        row = bytearray()
        i = min(n - 1, y // cell)
        for x in range(width):
            j = min(n - 1, x // cell)
            intensity = int(255.0 - 205.0 * float(cm[i, j]) / max_value)
            border = (x % cell == 0) or (y % cell == 0)
            if border:
                row.extend([30, 30, 30])
            else:
                row.extend([intensity, intensity, 255])
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw))
    png += _png_chunk(b"IEND", b"")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png)


def write_report_csv(path, metrics_by_scope):
    rows = []
    for scope, metrics in metrics_by_scope.items():
        rows.append({
            "scope": scope,
            "class": "macro",
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["macro_precision"],
            "recall": metrics["macro_recall"],
            "f1": metrics["macro_f1"],
            "support": "",
        })
        for label, values in metrics["per_class"].items():
            rows.append({
                "scope": scope,
                "class": label,
                "accuracy": "",
                "balanced_accuracy": "",
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1"],
                "support": values["support"],
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(label, model_path, rows, config, batch_size):
    model = tf.keras.models.load_model(str(model_path), compile=False)
    y_true, y_prob, prep_ms, infer_ms = predict_rows(model, rows, config, batch_size)
    metrics = compute_metrics(y_true, y_prob)
    metrics["model_label"] = label
    metrics["model_path"] = str(model_path)
    metrics["avg_preprocessing_time_ms"] = float(np.mean(prep_ms)) if prep_ms else 0.0
    metrics["avg_inference_time_ms"] = float(np.mean(infer_ms)) if infer_ms else 0.0
    metrics["avg_total_time_ms"] = metrics["avg_preprocessing_time_ms"] + metrics["avg_inference_time_ms"]
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate voice emotion model.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "voice_emotion_vietnamese_finetuned.h5")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "test_metadata.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "voice_emotion_config.yaml")
    parser.add_argument("--baseline-model", type=Path, default=PROJECT_ROOT / "final_production_model.h5")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    rows = read_metadata(args.metadata)
    if not rows:
        raise ValueError("No valid rows in metadata: %s" % args.metadata)

    metrics_by_scope = {}
    new_metrics = evaluate_model("CNN_GRU_Vietnamese_SER", args.model, rows, config, args.batch_size)
    metrics_by_scope["all"] = new_metrics
    for source in ["tess", "vietnamese_kaggle", "self_recorded", "unknown"]:
        source_rows = [row for row in rows if row.get("source") == source]
        if source_rows:
            metrics_by_scope[source] = evaluate_model("CNN_GRU_Vietnamese_SER", args.model, source_rows, config, args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classification_report.json").write_text(
        json.dumps(metrics_by_scope, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report_csv(args.output_dir / "classification_report.csv", metrics_by_scope)
    plot_confusion_matrix(new_metrics["confusion_matrix"], args.output_dir / "confusion_matrix.png", "CNN-GRU test confusion matrix")

    comparison_rows = [{
        "model": "CNN_GRU_Vietnamese_SER",
        "model_path": str(args.model),
        "accuracy": new_metrics["accuracy"],
        "balanced_accuracy": new_metrics["balanced_accuracy"],
        "macro_f1": new_metrics["macro_f1"],
        "avg_preprocessing_time_ms": new_metrics["avg_preprocessing_time_ms"],
        "avg_inference_time_ms": new_metrics["avg_inference_time_ms"],
        "note": "",
    }]
    if args.baseline_model.exists():
        try:
            baseline = evaluate_model("Baseline CNN+LSTM", args.baseline_model, rows, config, args.batch_size)
            comparison_rows.append({
                "model": "Baseline CNN+LSTM",
                "model_path": str(args.baseline_model),
                "accuracy": baseline["accuracy"],
                "balanced_accuracy": baseline["balanced_accuracy"],
                "macro_f1": baseline["macro_f1"],
                "avg_preprocessing_time_ms": baseline["avg_preprocessing_time_ms"],
                "avg_inference_time_ms": baseline["avg_inference_time_ms"],
                "note": "Uses new preprocessing; compare only if baseline input shape is compatible.",
            })
        except Exception as exc:
            comparison_rows.append({
                "model": "Baseline CNN+LSTM",
                "model_path": str(args.baseline_model),
                "accuracy": "",
                "balanced_accuracy": "",
                "macro_f1": "",
                "avg_preprocessing_time_ms": "",
                "avg_inference_time_ms": "",
                "note": "Could not evaluate baseline with new preprocessing: %s" % exc,
            })

    with (args.output_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("Macro F1:", "%.4f" % new_metrics["macro_f1"])
    print("Balanced accuracy:", "%.4f" % new_metrics["balanced_accuracy"])
    print("Report dir:", args.output_dir)


if __name__ == "__main__":
    main()
