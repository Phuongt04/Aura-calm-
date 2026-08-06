import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def file_size_mb(path):
    path = Path(path)
    if path.is_dir():
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024.0 * 1024.0)
    if path.exists():
        return path.stat().st_size / (1024.0 * 1024.0)
    return 0.0


def benchmark(model, repeats=20):
    shape = [1]
    for dim in model.input_shape[1:]:
        shape.append(1 if dim is None else int(dim))
    dummy = np.zeros(shape, dtype=np.float32)
    for _ in range(3):
        model.predict(dummy, batch_size=1, verbose=0)
    start = time.time()
    for _ in range(repeats):
        model.predict(dummy, batch_size=1, verbose=0)
    elapsed = (time.time() - start) * 1000.0 / float(repeats)
    return elapsed


def export_saved_model(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(output_dir))
    return output_dir


def export_tflite_fp16(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    return output_path


def export_onnx(model_path, output_path):
    try:
        import tf2onnx
    except Exception as exc:
        raise RuntimeError("tf2onnx is not installed or not compatible: %s" % exc)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--keras",
        str(model_path),
        "--output",
        str(output_path),
        "--opset",
        "13",
    ]
    import subprocess
    subprocess.check_call(cmd)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Export voice emotion model for deployment.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "voice_emotion_vietnamese_finetuned.h5")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "exports")
    parser.add_argument("--saved-model", action="store_true", default=True)
    parser.add_argument("--tflite-fp16", action="store_true")
    parser.add_argument("--onnx", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = tf.keras.models.load_model(str(args.model), compile=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(args.model),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "parameter_count": int(model.count_params()),
        "h5_size_mb": file_size_mb(args.model),
        "inference_latency_ms_batch1": benchmark(model),
        "exports": {},
    }
    if args.saved_model:
        saved = export_saved_model(model, args.output_dir / "saved_model")
        report["exports"]["saved_model"] = {"path": str(saved), "size_mb": file_size_mb(saved)}
    if args.tflite_fp16:
        try:
            tflite = export_tflite_fp16(model, args.output_dir / "voice_emotion_fp16.tflite")
            report["exports"]["tflite_fp16"] = {"path": str(tflite), "size_mb": file_size_mb(tflite)}
        except Exception as exc:
            report["exports"]["tflite_fp16"] = {"error": str(exc)}
    if args.onnx:
        try:
            onnx = export_onnx(args.model, args.output_dir / "voice_emotion.onnx")
            report["exports"]["onnx"] = {"path": str(onnx), "size_mb": file_size_mb(onnx)}
        except Exception as exc:
            report["exports"]["onnx"] = {"error": str(exc)}

    report_path = args.output_dir / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Export report:", report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
