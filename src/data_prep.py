# src/data_prep.py
import os
from typing import Tuple

import cv2
import numpy as np
import pydicom
from tqdm import tqdm

from .config import IMG_SIZE


def load_dicom_series(series_path: str) -> np.ndarray:
    """
    Load a folder of DICOM slices as a 3D volume (Z, H, W).

    series_path/
      slice1.dcm
      slice2.dcm
      ...

    Returns
    -------
    volume : np.ndarray
        3D array of shape (Z, H, W).
    """
    files = [
        f for f in os.listdir(series_path)
        if f.lower().endswith(".dcm")
    ]
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {series_path}")

    slices = []
    for f in files:
        d = pydicom.dcmread(os.path.join(series_path, f))
        slices.append(d)

    # Try to sort by InstanceNumber, fallback to filename if missing.
    try:
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    except Exception:
        slices.sort(key=lambda s: s.SOPInstanceUID)

    volume = np.stack([s.pixel_array for s in slices], axis=0)  # (Z, H, W)
    return volume


def preprocess_slice(slice_2d: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D slice to [0, 1] and resize to (IMG_SIZE, IMG_SIZE).

    Returns
    -------
    img : np.ndarray
        2D array float32 in [0, 1] with shape (IMG_SIZE, IMG_SIZE).
    """
    slice_2d = slice_2d.astype(np.float32)

    # Normalize to [0,1]
    slice_2d -= slice_2d.min()
    max_val = slice_2d.max()
    if max_val > 0:
        slice_2d /= max_val

    # Resize
    img = cv2.resize(slice_2d, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def build_dataset_from_root(root_path: str, label: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dataset from a root directory.

    Expected layout:
      root_path/
        patient_001/
          series1/
          series2/ (optional)
        patient_002/
          series1/
        ...

    Parameters
    ----------
    root_path : str
        Path containing patient subfolders.
    label : int
        Class label (1 for glioma, 0 for healthy).

    Returns
    -------
    X : np.ndarray
        Shape (N, IMG_SIZE, IMG_SIZE, 1)
    y : np.ndarray
        Shape (N,)
    """
    X, y = [], []
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"Root path does not exist: {root_path}")

    patient_dirs = [
        os.path.join(root_path, d)
        for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    ]

    for patient_dir in tqdm(patient_dirs, desc=f"Building dataset from {root_path}"):
        series_dirs = [
            os.path.join(patient_dir, s)
            for s in os.listdir(patient_dir)
            if os.path.isdir(os.path.join(patient_dir, s))
        ]
        if not series_dirs:
            # If patient folder directly contains .dcm files, treat it as one series
            dicom_files = [
                f for f in os.listdir(patient_dir)
                if f.lower().endswith(".dcm")
            ]
            if dicom_files:
                series_dirs = [patient_dir]
            else:
                print(f"No series found for {patient_dir}, skipping.")
                continue

        # For now, only use the first series (e.g., T1).
        series_path = series_dirs[0]

        try:
            volume = load_dicom_series(series_path)
        except Exception as e:
            print(f"Skipping {patient_dir} due to error: {e}")
            continue

        # Pick the middle slice for simplicity.
        mid_idx = volume.shape[0] // 2
        slice_2d = volume[mid_idx]

        img = preprocess_slice(slice_2d)
        X.append(img)
        y.append(label)

    if not X:
        raise RuntimeError(f"No samples found in {root_path}.")

    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (N, H, W, 1)
    y = np.array(y, dtype=np.int32)
    return X, y
