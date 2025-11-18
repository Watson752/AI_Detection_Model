# src/train.py
import os
import numpy as np

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import DATA_PROCESSED, MODELS_DIR
from .model import build_cnn


def main():
    X_path = os.path.join(DATA_PROCESSED, "X.npy")
    y_path = os.path.join(DATA_PROCESSED, "y.npy")

    if not (os.path.isfile(X_path) and os.path.isfile(y_path)):
        raise FileNotFoundError(
            f"Processed dataset not found. Expected {X_path} and {y_path}. "
            f"Run `python -m src.make_dataset` first."
        )

    X = np.load(X_path)
    y = np.load(y_path)

    print(f"Loaded X with shape {X.shape}, y with shape {y.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Save test set for later evaluation
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    np.save(os.path.join(DATA_PROCESSED, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_PROCESSED, "y_test.npy"), y_test)

    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(X_train, y_train, batch_size=16, shuffle=True)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=16, shuffle=False)

    model = build_cnn()
    model.summary()

    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, "best_model.h5")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        epochs=40,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # Save final model
    final_model_path = os.path.join(MODELS_DIR, "final_model.h5")
    model.save(final_model_path)
    print(f"Training complete. Best model: {checkpoint_path}, final model: {final_model_path}")


if __name__ == "__main__":
    main()
