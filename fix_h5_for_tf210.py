from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_DIR / "vision_emotion_model_compatible.h5"
DEFAULT_OUTPUT = PROJECT_DIR / "soft" / "vision_emotion_model_tf210.h5"
DEFAULT_REBUILT = PROJECT_DIR / "soft" / "vision_emotion_model_tf210_rebuilt.h5"
DEFAULT_OLD_PYTHON = Path(r"C:\Users\ttphu\AppData\Local\Programs\Python\Python310\python.exe")
DEFAULT_NEW_PYTHON = PROJECT_DIR / ".venv_face_convert" / "Scripts" / "python.exe"
EXPECTED_INPUT_SHAPE = "(None, 224, 224, 3)"
EXPECTED_OUTPUT_SHAPE = "(None, 4)"
EXPECTED_PARAMS = 2422468


Removal = Tuple[str, str, Any]
Conversion = Tuple[str, Any]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def decode_h5_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return str(value)


def json_path(parent: str, key: str) -> str:
    if key.isidentifier():
        return f"{parent}.{key}"
    return f"{parent}[{key!r}]"


def collect_keras_histories(value: Any, histories: List[List[Any]]) -> None:
    if isinstance(value, dict):
        if value.get("class_name") == "__keras_tensor__":
            config = value.get("config")
            if isinstance(config, dict) and isinstance(config.get("keras_history"), list):
                history = config["keras_history"]
                if len(history) == 3:
                    histories.append(history)
            return
        for child in value.values():
            collect_keras_histories(child, histories)
    elif isinstance(value, list):
        for child in value:
            collect_keras_histories(child, histories)


def convert_inbound_nodes_if_needed(node: Dict[str, Any], path: str, converted: List[Conversion]) -> None:
    inbound_nodes = node.get("inbound_nodes")
    if not isinstance(inbound_nodes, list) or not inbound_nodes:
        return
    if not all(isinstance(item, dict) and "args" in item for item in inbound_nodes):
        return

    legacy_nodes = []
    for inbound_node in inbound_nodes:
        histories: List[List[Any]] = []
        collect_keras_histories(inbound_node.get("args", []), histories)
        if not histories:
            continue
        legacy_nodes.append([[history[0], history[1], history[2], {}] for history in histories])

    if len(legacy_nodes) != len(inbound_nodes):
        raise ValueError(f"Could not convert all inbound_nodes at {path}")

    node["inbound_nodes"] = legacy_nodes
    converted.append((f"{path}.inbound_nodes Keras3 -> Keras2 legacy", inbound_nodes))


def clean_metadata_node(
    node: Any,
    path: str,
    removed: List[Removal],
    converted: List[Conversion],
) -> Any:
    if isinstance(node, list):
        return [
            clean_metadata_node(item, f"{path}[{index}]", removed, converted)
            for index, item in enumerate(node)
        ]

    if not isinstance(node, dict):
        return node

    class_name = node.get("class_name")
    config = node.get("config")

    keys_to_delete = []
    for key, value in node.items():
        if key == "optional":
            keys_to_delete.append(key)
        elif key == "quantization_config" and value is None:
            keys_to_delete.append(key)
        elif key == "synchronized" and value is False:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        removed.append((path, key, node[key]))
        del node[key]

    dtype_value = node.get("dtype")
    if isinstance(dtype_value, dict) and dtype_value.get("class_name") == "DTypePolicy":
        dtype_config = dtype_value.get("config")
        if isinstance(dtype_config, dict) and dtype_config.get("name") == "float32":
            node["dtype"] = "float32"
            converted.append((f"{path}.dtype DTypePolicy -> string", dtype_value))

    if class_name == "InputLayer" and isinstance(config, dict):
        if "batch_shape" in config and "batch_input_shape" not in config:
            old_value = config.pop("batch_shape")
            config["batch_input_shape"] = old_value
            converted.append((f"{path}.config.batch_shape -> batch_input_shape", old_value))

    convert_inbound_nodes_if_needed(node, path, converted)

    for key, value in list(node.items()):
        node[key] = clean_metadata_node(value, json_path(path, key), removed, converted)

    return node


def inspect_h5(path: Path) -> None:
    print(f"[H5] File: {path}")
    with h5py.File(path, "r") as h5:
        print(f"[H5] Root groups: {list(h5.keys())}")
        print(f"[H5] Root attrs: {list(h5.attrs.keys())}")
        if "model_config" not in h5.attrs:
            raise ValueError("Missing HDF5 root attribute: model_config")
        model_config = decode_h5_attr(h5.attrs["model_config"])
        print(f"[H5] model_config length: {len(model_config)}")
        print(f"[H5] training_config exists: {'training_config' in h5.attrs}")


def sanitize_h5_metadata(path: Path) -> Tuple[List[Removal], List[Conversion], bool]:
    removed: List[Removal] = []
    converted: List[Conversion] = []
    removed_training_config = False

    with h5py.File(path, "r+") as h5:
        raw_model_config = h5.attrs.get("model_config")
        if raw_model_config is None:
            raise ValueError("Missing HDF5 root attribute: model_config")

        model_config = json.loads(decode_h5_attr(raw_model_config))
        clean_config = clean_metadata_node(model_config, "$", removed, converted)
        clean_json = json.dumps(clean_config, separators=(",", ":"), ensure_ascii=True)
        h5.attrs.modify("model_config", clean_json)

        if "training_config" in h5.attrs:
            del h5.attrs["training_config"]
            removed_training_config = True

    return removed, converted, removed_training_config


def print_clean_log(removed: List[Removal], converted: List[Conversion], removed_training_config: bool) -> None:
    optional_items = [item for item in removed if item[1] == "optional"]
    print(f"[CLEAN] optional keys removed: {len(optional_items)}")
    for path, key, old_value in optional_items:
        print(f"  - {json_path(path, key)} old_value={old_value!r}")

    other_items = [item for item in removed if item[1] != "optional"]
    if other_items:
        print("[CLEAN] Other metadata keys removed:")
        for path, key, old_value in other_items:
            print(f"  - {json_path(path, key)} old_value={old_value!r}")
    else:
        print("[CLEAN] Other metadata keys removed: none")

    if converted:
        print("[CLEAN] Metadata conversions:")
        for path, old_value in converted:
            print(f"  - {path} old_value={old_value!r}")
    else:
        print("[CLEAN] Metadata conversions: none")

    print(f"[CLEAN] Removed training_config: {removed_training_config}")


def run_subprocess(command: List[str], env: Dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("[RUN]", " ".join(f'"{part}"' if " " in part else part for part in command))
    result = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(result.stdout)
    return result


def validate_with_tf210(model_path: Path, old_python: Path) -> bool:
    if not old_python.exists():
        raise FileNotFoundError(f"TensorFlow 2.10 Python not found: {old_python}")

    validation_code = rf"""
import sys
import numpy as np
import tensorflow as tf

model_path = r"{model_path}"
print("TensorFlow:", tf.__version__)
model = tf.keras.models.load_model(model_path, compile=False)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Params:", model.count_params())

x = np.zeros((1, 224, 224, 3), dtype=np.float32)
y = model.predict(x)
print("Predict output shape:", y.shape)
print("Predict output:", y)

if tf.__version__ != "2.10.1":
    raise SystemExit("Unexpected TensorFlow version: " + tf.__version__)
if str(model.input_shape) != "{EXPECTED_INPUT_SHAPE}":
    raise SystemExit("Unexpected input shape: " + str(model.input_shape))
if str(model.output_shape) != "{EXPECTED_OUTPUT_SHAPE}":
    raise SystemExit("Unexpected output shape: " + str(model.output_shape))
if model.count_params() != {EXPECTED_PARAMS}:
    raise SystemExit("Unexpected param count: " + str(model.count_params()))
if y.shape != (1, 4):
    raise SystemExit("Unexpected predict shape: " + str(y.shape))
if not np.all(np.isfinite(y)):
    raise SystemExit("Predict output contains NaN or Inf")

print("TF210 validation passed.")
"""
    result = run_subprocess([str(old_python), "-c", validation_code])
    return result.returncode == 0


def create_sanitized_copy(source: Path, output: Path, backup_dir: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source model not found: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("Output path must be different from source path.")

    backup_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_backup = backup_dir / f"{source.stem}_source_{timestamp()}{source.suffix}"
    print(f"[BACKUP] Source backup: {source_backup}")
    shutil.copy2(source, source_backup)

    if output.exists():
        output_backup = backup_dir / f"{output.stem}_before_rewrite_{timestamp()}{output.suffix}"
        print(f"[BACKUP] Existing output backup: {output_backup}")
        shutil.copy2(output, output_backup)

    print(f"[COPY] {source} -> {output}")
    shutil.copy2(source, output)


def fallback_rebuild_with_keras3(
    source: Path,
    rebuilt_output: Path,
    new_python: Path,
    old_python: Path,
    backup_dir: Path,
) -> bool:
    if not new_python.exists():
        print(f"[FALLBACK] Keras 3 Python not found: {new_python}")
        return False

    if rebuilt_output.exists():
        rebuilt_backup = backup_dir / f"{rebuilt_output.stem}_before_rewrite_{timestamp()}{rebuilt_output.suffix}"
        print(f"[BACKUP] Existing rebuilt output backup: {rebuilt_backup}")
        shutil.copy2(rebuilt_output, rebuilt_backup)

    with tempfile.TemporaryDirectory(prefix="tf210_rebuild_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        weights_npz = tmp_dir / "weights_by_layer.npz"
        architecture_json = tmp_dir / "architecture_tf210.json"

        export_code = rf"""
import json
import numpy as np
import keras

source = r"{source}"
weights_npz = r"{weights_npz}"
architecture_json = r"{architecture_json}"

model = keras.models.load_model(source, compile=False)
config = json.loads(model.to_json())

def clean(node):
    if isinstance(node, list):
        return [clean(item) for item in node]
    if not isinstance(node, dict):
        return node

    def collect_histories(value, histories):
        if isinstance(value, dict):
            if value.get("class_name") == "__keras_tensor__":
                config = value.get("config")
                if isinstance(config, dict) and isinstance(config.get("keras_history"), list):
                    history = config["keras_history"]
                    if len(history) == 3:
                        histories.append(history)
                return
            for child in value.values():
                collect_histories(child, histories)
        elif isinstance(value, list):
            for child in value:
                collect_histories(child, histories)

    dtype_value = node.get("dtype")
    if isinstance(dtype_value, dict) and dtype_value.get("class_name") == "DTypePolicy":
        dtype_config = dtype_value.get("config")
        if isinstance(dtype_config, dict) and dtype_config.get("name") == "float32":
            node["dtype"] = "float32"
    if node.get("class_name") == "InputLayer" and isinstance(node.get("config"), dict):
        cfg = node["config"]
        if "batch_shape" in cfg and "batch_input_shape" not in cfg:
            cfg["batch_input_shape"] = cfg.pop("batch_shape")
    inbound_nodes = node.get("inbound_nodes")
    if isinstance(inbound_nodes, list) and inbound_nodes and all(isinstance(item, dict) and "args" in item for item in inbound_nodes):
        legacy_nodes = []
        for inbound_node in inbound_nodes:
            histories = []
            collect_histories(inbound_node.get("args", []), histories)
            if histories:
                legacy_nodes.append([[history[0], history[1], history[2], {{}}] for history in histories])
        if len(legacy_nodes) == len(inbound_nodes):
            node["inbound_nodes"] = legacy_nodes
    for key in list(node.keys()):
        value = node[key]
        if key == "optional":
            del node[key]
        elif key == "quantization_config" and value is None:
            del node[key]
        elif key == "synchronized" and value is False:
            del node[key]
        else:
            node[key] = clean(value)
    return node

config = clean(config)
with open(architecture_json, "w", encoding="utf-8") as f:
    json.dump(config, f, separators=(",", ":"), ensure_ascii=True)

arrays = {{}}
for layer in model.layers:
    for index, weight in enumerate(layer.get_weights()):
        arrays[f"{{layer.name}}::{{index}}"] = weight
np.savez(weights_npz, **arrays)

x = np.zeros((1, 224, 224, 3), dtype=np.float32)
y = model.predict(x, verbose=0)
np.save(r"{tmp_dir / 'keras3_output.npy'}", y)
print("Keras3 export complete")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Params:", model.count_params())
"""
        export_result = run_subprocess([str(new_python), "-c", export_code])
        if export_result.returncode != 0:
            return False

        rebuild_code = rf"""
import json
import numpy as np
import tensorflow as tf

architecture_json = r"{architecture_json}"
weights_npz = r"{weights_npz}"
rebuilt_output = r"{rebuilt_output}"
keras3_output_path = r"{tmp_dir / 'keras3_output.npy'}"

with open(architecture_json, "r", encoding="utf-8") as f:
    model = tf.keras.models.model_from_json(f.read())

weights = np.load(weights_npz)
for layer in model.layers:
    current_weights = layer.get_weights()
    if not current_weights:
        continue
    incoming = []
    for index, current in enumerate(current_weights):
        key = f"{{layer.name}}::{{index}}"
        if key not in weights:
            raise SystemExit("Missing weight: " + key)
        value = weights[key]
        if value.shape != current.shape:
            raise SystemExit("Shape mismatch for " + key + ": " + str(value.shape) + " != " + str(current.shape))
        incoming.append(value)
    layer.set_weights(incoming)

print("TensorFlow:", tf.__version__)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Params:", model.count_params())

x = np.zeros((1, 224, 224, 3), dtype=np.float32)
y = model.predict(x)
old_y = np.load(keras3_output_path)
print("Predict output shape:", y.shape)
print("Predict output:", y)
print("Keras3 output:", old_y)
print("allclose:", np.allclose(y, old_y, rtol=1e-5, atol=1e-6))

if str(model.input_shape) != "{EXPECTED_INPUT_SHAPE}":
    raise SystemExit("Unexpected input shape: " + str(model.input_shape))
if str(model.output_shape) != "{EXPECTED_OUTPUT_SHAPE}":
    raise SystemExit("Unexpected output shape: " + str(model.output_shape))
if model.count_params() != {EXPECTED_PARAMS}:
    raise SystemExit("Unexpected param count: " + str(model.count_params()))
if y.shape != (1, 4):
    raise SystemExit("Unexpected predict shape: " + str(y.shape))
if not np.all(np.isfinite(y)):
    raise SystemExit("Predict output contains NaN or Inf")
if not np.allclose(y, old_y, rtol=1e-5, atol=1e-6):
    raise SystemExit("Rebuilt model output is not close to Keras3 output")

model.save(rebuilt_output, include_optimizer=False)
print("Saved rebuilt model:", rebuilt_output)
"""
        rebuild_result = run_subprocess([str(old_python), "-c", rebuild_code])
        if rebuild_result.returncode != 0:
            return False

    return validate_with_tf210(rebuilt_output, old_python)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix a Keras H5 model for TensorFlow 2.10.1 inference.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Input compatible H5 model. It is never modified.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Sanitized TF 2.10 H5 output.")
    parser.add_argument("--rebuilt-output", default=str(DEFAULT_REBUILT), help="Fallback rebuilt TF 2.10 H5 output.")
    parser.add_argument("--old-python", default=str(DEFAULT_OLD_PYTHON), help="Python executable with TensorFlow 2.10.1.")
    parser.add_argument("--new-python", default=str(DEFAULT_NEW_PYTHON), help="Python executable with Keras 3.")
    parser.add_argument("--backup-dir", default=str(PROJECT_DIR / "model_backups"), help="Backup directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    rebuilt_output = Path(args.rebuilt_output).resolve()
    old_python = Path(args.old_python).resolve()
    new_python = Path(args.new_python).resolve()
    backup_dir = Path(args.backup_dir).resolve()

    try:
        create_sanitized_copy(source, output, backup_dir)
        inspect_h5(output)
        removed, converted, removed_training_config = sanitize_h5_metadata(output)
        print_clean_log(removed, converted, removed_training_config)

        print("[VALIDATE] Testing sanitized H5 with TensorFlow 2.10.1.")
        if validate_with_tf210(output, old_python):
            print("[DONE] TensorFlow 2.10.1 validation passed for sanitized H5.")
            return 0

        print("[FALLBACK] Sanitized metadata file did not pass TF 2.10.1 validation.")
        print("[FALLBACK] Rebuilding architecture under TensorFlow 2.10.1 and copying weights from Keras 3.")
        if fallback_rebuild_with_keras3(source, rebuilt_output, new_python, old_python, backup_dir):
            print("[DONE] TensorFlow 2.10.1 validation passed for rebuilt H5.")
            return 0

        print("[ERROR] Metadata fix and fallback rebuild both failed.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
