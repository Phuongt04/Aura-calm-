import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


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

    # Convert (mel, time, channels) to a time sequence. Each time step sees all
    # remaining mel bins and feature channels.
    x = layers.Permute((2, 1, 3), name="time_major")(x)
    time_steps = int(input_shape[1] // 4) if input_shape[1] is not None else -1
    freq_bins = int(input_shape[0] // 4)
    if time_steps > 0:
        x = layers.Reshape((time_steps, freq_bins * 128), name="time_sequence")(x)
    else:
        x = layers.TimeDistributed(layers.Flatten(), name="time_sequence")(x)

    x = layers.GRU(64, name="gru_64")(x)
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    x = layers.Dropout(0.35, name="dropout_035")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="CNN_GRU_Vietnamese_SER")


def model_size_mb(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    return os.path.getsize(str(path)) / (1024.0 * 1024.0)


def print_model_report(model, saved_path=None):
    print("Model name:", model.name)
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Total parameters:", model.count_params())
    if saved_path:
        print("Model size MB:", round(model_size_mb(saved_path), 3))


if __name__ == "__main__":
    model = build_cnn_gru_ser()
    model.summary()
    print_model_report(model)
