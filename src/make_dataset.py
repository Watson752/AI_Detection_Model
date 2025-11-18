# src/make_dataset.py
import os
import numpy as np

from .data_prep import build_dataset_from_root
from .config import DATA_PROCESSED, UCSF_PDGM_DICOM, HEALTHY_DICOM


def main():
    print("=== Building glioma dataset from UCSF-PDGM ===")
    X_glioma, y_glioma = build_dataset_from_root(UCSF_PDGM_DICOM, label=1)

    print("=== Building healthy dataset ===")
    X_healthy, y_healthy = build_dataset_from_root(HEALTHY_DICOM, label=0)

    X = np.concatenate([X_glioma, X_healthy], axis=0)
    y = np.concatenate([y_glioma, y_healthy], axis=0)

    os.makedirs(DATA_PROCESSED, exist_ok=True)

    X_path = os.path.join(DATA_PROCESSED, "X.npy")
    y_path = os.path.join(DATA_PROCESSED, "y.npy")

    np.save(X_path, X)
    np.save(y_path, y)

    print(f"Saved X to {X_path} with shape {X.shape}")
    print(f"Saved y to {y_path} with shape {y.shape}")
    print("Dataset creation complete.")


if __name__ == "__main__":
    main()
