# src/grad_cam.py
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from .config import DATA_PROCESSED, MODELS_DIR


def get_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Return the name of the last Conv2D layer in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(
    img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a single image batch.

    Parameters
    ----------
    img_array : np.ndarray
        Shape (1, H, W, C), normalized.
    model : tf.keras.Model
    last_conv_layer_name : str

    Returns
    -------
    heatmap : np.ndarray
        2D array in [0,1]
    """
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # for binary classification

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (Hc, Wc, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (Hc, Wc, 1)
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(
    heatmap: np.ndarray, image: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay heatmap on top of original grayscale image.

    image is assumed to be (H, W) in [0, 1].
    """
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert original image to uint8
    if image.ndim == 2:
        base = (image * 255).astype("uint8")
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        base = (image * 255).astype("uint8")

    overlayed = cv2.addWeighted(heatmap_color, alpha, base, 1 - alpha, 0)
    return overlayed


def demo_example(index: int = 0):
    """
    Demo Grad-CAM on one test sample by index.
    """
    X_test_path = os.path.join(DATA_PROCESSED, "X_test.npy")
    if not os.path.isfile(X_test_path):
        raise FileNotFoundError(
            f"{X_test_path} not found. Run `python -m src.train` first."
        )

    X_test = np.load(X_test_path)
    model_path = os.path.join(MODELS_DIR, "best_model.h5")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Train the model first."
        )

    model = tf.keras.models.load_model(model_path)

    if index < 0 or index >= X_test.shape[0]:
        raise IndexError(f"Index {index} out of range for X_test of size {X_test.shape[0]}")

    img = X_test[index]  # (H, W, 1)
    img_batch = np.expand_dims(img, axis=0)

    last_conv_layer_name = get_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name)
    overlayed = overlay_heatmap(heatmap, img[..., 0])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img[..., 0], cmap="gray")
    plt.title("Original MRI slice")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlayed)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_example(index=0)
