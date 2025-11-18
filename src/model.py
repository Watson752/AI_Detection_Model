# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

from .config import IMG_SIZE, CHANNELS


def build_cnn() -> tf.keras.Model:
    """
    Build a simple 2D CNN for binary classification (glioma vs healthy).
    """
    input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS)

    inputs = layers.Input(shape=input_shape, name="input_image")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="tumor_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="BrainTumorCNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
