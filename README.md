# Brain Tumour Detection Model

Brain Tumor Detection – Deep Learning Model

This project uses Python, TensorFlow/Keras, and OpenCV to build a deep learning model that detects glioma brain tumors from MRI scans. The dataset is taken from the UCSF-PDGM collection on The Cancer Imaging Archive (TCIA).

Overview

Classifies MRI slices as Glioma vs Healthy

Uses a Convolutional Neural Network (CNN) for image classification

Includes data preprocessing, data augmentation, and model training scripts

Provides Grad-CAM visualizations to show which parts of the brain influenced the prediction

Evaluation includes precision, recall, F1-score, and a precision–recall curve

Project Structure
```
AI_Detection_Model/
  data/
    raw/           # Original MRI data (DICOM/NIfTI)
    processed/     # Numpy arrays (X.npy, y.npy, test splits)
  models/          # Saved models (.h5)
  src/
    data_prep.py
    make_dataset.py
    model.py
    train.py
    evaluate.py
    grad_cam.py
  README.md
  requirements.txt
```


How to Run
1. Install dependencies
```
pip install -r requirements.txt

```

2. Prepare the dataset

Place your MRI data in:
```
data/raw/ucsf_pdgm/
data/raw/healthy/
```

Then run:
```
python -m src.make_dataset
```

3. Train the model
```
python -m src.train
```

4. Evaluate performance
```
python -m src.evaluate
```
5. Generate Grad-CAM heatmaps
```
python -m src.grad_cam
```
Features

Reads MRI scans (DICOM/NIfTI)

Extracts central slices for training

Applies augmentation to improve robustness

Saves best-performing model automatically

Produces Grad-CAM heatmaps for interpretability

Future Improvements

Multi-modal MRI input (T1, T1-CE, T2, FLAIR)

3D CNN models on full MRI volumes

Clinical label prediction (e.g., IDH mutation status)