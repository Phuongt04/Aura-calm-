from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np


DEFAULT_REMOVE_KEYS = {"optional", "quantization_config"}
DEFAULT_INPUT_LAYER_KEYS = {"batch_shape"}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return str(value)


def inspect_h5(path: Path) -> None:
    print(f"[H5] Inspecting: {path}")
    with h5py.File(path, "r") as h5:
        print(f"[H5] Root keys: {list(h5.keys())}")
        print(f"[H5] Root attrs: {list(h5.attrs.keys())}")
        model_config = h5.attrs.get("model_config")
        if model_config is None:
            raise ValueError("Missing required HDF5 root attribute: model_config")
        print(f"[H5] model_config type: {type(model_config).__name__}")
        print(f"[H5] model_config length: {len(decode_attr(model_config))}")
        print(f"[H5] training_config exists: {'training_config' in h5.attrs}")


def layer_class_name(node: Dict[str, Any]) -> str:
    return str(node.get("class_name", ""))


def clean_config_node(
    node: Any,
    path: str,
    removed: List[Tuple[str, str, Any]],
    converted: List[Tuple[str, Any]],
) -> Any:
    if isinstance(node, list):
        return [
            clean_config_node(item, f"{path}[{index}]", removed, converted)
            for index, item in enumerate(node)
        ]

    if not isinstance(node, dict):
        return node

    current_class = layer_class_name(node)
    config = node.get("config")
    is_input_layer = current_class == "InputLayer" and isinstance(config, dict)

    keys_to_delete = []
    for key, value in node.items():
        if key in DEFAULT_REMOVE_KEYS and value is None:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        removed.append((path, key, node[key]))
        del node[key]

    if is_input_layer and "batch_shape" in config and "batch_input_shape" not in config:
        config["batch_input_shape"] = config.pop("batch_shape")
        converted.append((f"{path}.config", config["batch_input_shape"]))

    for key, value in list(node.items()):
        child_path = f"{path}.{key}"
        node[key] = clean_config_node(value, child_path, removed, converted)

    return node


def remove_training_config(clean_path: Path) -> bool:
    with h5py.File(clean_path, "r+") as h5:
        if "training_config" in h5.attrs:
            del h5.attrs["training_config"]
            return True
    return False


def write_clean_model_config(clean_path: Path) -> Tuple[List[Tuple[str, str, Any]], List[Tuple[str, Any]]]:
    removed: List[Tuple[str, str, Any]] = []
    converted: List[Tuple[str, Any]] = []

    with h5py.File(clean_path, "r+") as h5:
        raw_config = h5.attrs.get("model_config")
        if raw_config is None:
            raise ValueError("Missing required HDF5 root attribute: model_config")

        config = json.loads(decode_attr(raw_config))
        clean_config = clean_config_node(config, "$", removed, converted)
        clean_json = json.dumps(clean_config, separators=(",", ":"), ensure_ascii=True)
        h5.attrs.modify("model_config", clean_json)

    return removed, converted


def print_change_log(
    removed: List[Tuple[str, str, Any]],
    converted: List[Tuple[str, Any]],
    removed_training_config: bool,
) -> None:
    print("[CLEAN] Removed metadata keys:")
    if not removed:
        print("  none")
    else:
        counts = Counter(key for _, key, _ in removed)
        for key, count in sorted(counts.items()):
            print(f"  {key}: {count}")
        for path, key, value in removed:
            print(f"  - {path}.{key} = {value!r}")

    print("[CLEAN] InputLayer batch_shape conversions:")
    if not converted:
        print("  none")
    else:
        for path, value in converted:
            print(f"  - {path}.batch_shape -> batch_input_shape = {value!r}")

    print(f"[CLEAN] Removed training_config: {removed_training_config}")


def load_and_validate(clean_path: Path) -> Any:
    import keras

    print(f"[CHECK] Keras version: {keras.__version__}")
    print(f"[CHECK] Loading: {clean_path}")
    model = keras.models.load_model(str(clean_path), compile=False)
    print(f"[CHECK] Input shape: {model.input_shape}")
    print(f"[CHECK] Output shape: {model.output_shape}")
    print(f"[CHECK] Total params: {model.count_params()}")
    print("[CHECK] Model summary:")
    model.summary()

    x = np.zeros((1, 224, 224, 3), dtype=np.float32)
    y = model.predict(x, verbose=0)
    print(f"[CHECK] Predict output shape: {y.shape}")
    print(f"[CHECK] Predict output: {y}")

    if y.shape != (1, 4):
        raise ValueError(f"Unexpected prediction shape: {y.shape}, expected (1, 4)")
    if not np.all(np.isfinite(y)):
        raise ValueError("Prediction contains NaN or Inf")

    print("[CHECK] Inference validation passed.")
    return model


def save_exports(model: Any, keras_path: Path, compatible_h5_path: Path) -> None:
    print(f"[EXPORT] Saving native Keras file: {keras_path}")
    model.save(str(keras_path))
    print(f"[EXPORT] Saving compatible H5 file: {compatible_h5_path}")
    model.save(str(compatible_h5_path), include_optimizer=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean incompatible Keras metadata from a legacy HDF5 model without touching weights."
    )
    parser.add_argument("--input", default="vision_emotion_model.h5", help="Original H5 model path.")
    parser.add_argument("--output", default="vision_emotion_model_clean.h5", help="Cleaned H5 model path.")
    parser.add_argument("--keras-output", default="vision_emotion_model_clean.keras", help="Native Keras export path.")
    parser.add_argument(
        "--compatible-output",
        default="vision_emotion_model_compatible.h5",
        help="H5 export saved by the current Keras version without optimizer.",
    )
    parser.add_argument("--backup-dir", default="model_backups", help="Directory for timestamped backups.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.input).resolve()
    clean = Path(args.output).resolve()
    keras_output = Path(args.keras_output).resolve()
    compatible_output = Path(args.compatible_output).resolve()
    backup_dir = Path(args.backup_dir).resolve()

    try:
        if not src.exists():
            raise FileNotFoundError(f"Input model not found: {src}")
        if src == clean:
            raise ValueError("Output path must be different from input path; refusing to overwrite the original model.")

        backup_dir.mkdir(parents=True, exist_ok=True)
        original_backup = backup_dir / f"{src.stem}_original_{timestamp()}{src.suffix}"
        print(f"[BACKUP] Copying original to: {original_backup}")
        shutil.copy2(src, original_backup)

        if clean.exists():
            clean_backup = backup_dir / f"{clean.stem}_before_rewrite_{timestamp()}{clean.suffix}"
            print(f"[BACKUP] Existing output found. Backing it up to: {clean_backup}")
            shutil.copy2(clean, clean_backup)

        print(f"[COPY] Creating working copy: {clean}")
        shutil.copy2(src, clean)

        inspect_h5(clean)
        removed, converted = write_clean_model_config(clean)
        removed_train = remove_training_config(clean)
        print_change_log(removed, converted, removed_train)

        model = load_and_validate(clean)
        save_exports(model, keras_output, compatible_output)
        print("[DONE] Clean model, native Keras export, and compatible H5 export were created successfully.")
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
