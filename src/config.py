# src/config.py
import os

# Image settings
IMG_SIZE = 224               # resize all slices to 224x224
CHANNELS = 1                 # 1 for single modality (e.g., FLAIR). Later: 4 for T1/T1CE/T2/FLAIR

# Root paths (do not change unless you know what you’re doing)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Expected raw data layout (you will put data here manually)
# For example:
# data/raw/ucsf_pdgm/dicom/patient_001/T1/
# data/raw/healthy/dicom/patient_A/T1/
UCSF_PDGM_DICOM = os.path.join(DATA_RAW, "ucsf_pdgm", "dicom")
HEALTHY_DICOM = os.path.join(DATA_RAW, "healthy", "dicom")

# Make sure key folders exist
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
