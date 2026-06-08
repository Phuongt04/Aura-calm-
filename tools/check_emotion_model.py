from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and validate the emotion model for inference.")
    parser.add_argument("--model", default="vision_emotion_model_clean.h5", help="Model path to validate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).resolve()

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        import keras

        print(f"Keras version: {keras.__version__}")
        model = keras.models.load_model(str(model_path), compile=False)
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Total params: {model.count_params()}")
        print("Model summary:")
        model.summary()

        x = np.zeros((1, 224, 224, 3), dtype=np.float32)
        y = model.predict(x, verbose=0)
        print(f"Predict output shape: {y.shape}")
        print(f"Predict output: {y}")

        if y.shape != (1, 4):
            raise ValueError(f"Unexpected output shape: {y.shape}, expected (1, 4)")
        if not np.all(np.isfinite(y)):
            raise ValueError("Prediction contains NaN or Inf")

        print("Validation passed.")
        return 0
    except Exception as exc:
        print(f"Validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
