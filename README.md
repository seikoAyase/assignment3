# PTB-XL 12-Lead ECG Preprocessing, DSP Pipeline, and PyTorch DataLoader

[![CI/CD Pipeline](https://github.com/seikoAyase/assignment3/actions/workflows/ci.yml/badge.svg)](https://github.com/seikoAyase/assignment3/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Testing: Unittest](https://img.shields.io/badge/tests-16%20passed-success.svg)](tests/)
[![Dataset: PhysioNet](https://img.shields.io/badge/PhysioNet-PTB--XL%20v1.0.3-orange.svg)](https://physionet.org/content/ptb-xl/1.0.3/)

An end-to-end scientific software package for clinical 12-lead electrocardiogram (ECG) data processing, digital filtering, and deep learning data loading, benchmarked on the PhysioNet PTB-XL dataset (21,837 clinical records across 18,885 patients).

Developed by **Zhanbolat Baxen** as part of the course **Applied Software Development Project (CSE-2505M)**.

Repository: [https://github.com/seikoAyase/assignment3](https://github.com/seikoAyase/assignment3)

---

## 1. Scientific and Clinical Motivation

Cardiovascular diseases (CVDs) remain the leading cause of global mortality. The 12-lead surface electrocardiogram is the primary non-invasive diagnostic standard. However, raw clinical ECG recordings suffer from three major noise artifacts:
1. Low-frequency baseline wander (0.05-0.5 Hz) caused by patient respiration and body movement.
2. Powerline interference (50 Hz / 60 Hz) from hospital electrical grids.
3. High-frequency electromyographic (EMG) noise (>40 Hz) and radiofrequency interference.

This repository provides an automated, reproducible, and tested processing pipeline that maps raw SCP-ECG statements into standardized clinical superclasses, removes physiological and electrical artifacts via zero-phase digital filters, and streams normalized tensors into deep neural network architectures.

---

## 2. Mathematical Formulation of the DSP Pipeline

### A. Zero-Phase Butterworth Bandpass Filter (0.5 - 45.0 Hz)
Baseline wander (respiratory drift) and high-frequency muscle noise are eliminated using a 4th-order digital Butterworth bandpass filter.

Filter transfer function in the z-domain:

              b0 + b1*z^(-1) + b2*z^(-2) + ... + bM*z^(-M)
  H_BP(z)  =  --------------------------------------------
              1 + a1*z^(-1) + a2*z^(-2) + ... + aN*z^(-N)

To eliminate non-linear phase distortion that artificially alters critical clinical intervals (such as ST-elevation and QT-interval duration), zero-phase forward-backward filtering is performed:
  Step 1 (Forward pass):  y_forward[n] = Filter( x[n] )
  Step 2 (Reverse pass):  y_zero[n]    = Reverse( Filter( Reverse( y_forward[n] ) ) )

This guarantees an exact zero phase response across all frequency bands.

### B. Adaptive IIR Notch Filter (50.0 Hz)
Powerline electromagnetic interference (50 Hz grid frequency) is excised using a digital infinite impulse response (IIR) notch filter with quality factor Q = 30:

              1 - 2*cos(w0)*z^(-1) + z^(-2)
  H_notch(z) = b0 * ---------------------------------------
              1 - 2*r*cos(w0)*z^(-1) + r^2*z^(-2)

Parameters:
  w0 = 2 * pi * (f0 / fs)   where f0 = 50.0 Hz, fs = 100.0 Hz
  r  = 1 - w0 / (2 * Q)     where Q = 30.0

### C. Robust Per-Lead Z-Score Standardization with Flatline Guard
Signal amplitudes are normalized per channel to zero-mean and unit variance:

                 x[c, t] - mu[c]
  x_norm[c, t] = -------------------
                  sigma[c] + 1e-6

Parameters:
  mu[c]    = (1 / T) * sum_{t=1..T} x[c, t]
  sigma[c] = sqrt( (1 / T) * sum_{t=1..T} (x[c, t] - mu[c])^2 )
  1e-6     = protective constant preventing division by zero on flatline or disconnected leads

### D. Positive Class Loss Weighting (pos_weight)
To counteract severe class imbalance across the 5 diagnostic superclasses:

                  N_negative[c]
  pos_weight[c] = ---------------------
                  N_positive[c] + 1e-5

This weighting vector is passed directly to PyTorch BCEWithLogitsLoss during training.

---

## 3. SCP-ECG Diagnostic Hierarchy and Multi-Hot Mapping

The pipeline translates granular SCP-ECG diagnostic statements into the 5 primary clinical superclasses:
- NORM: Normal ECG (baseline healthy control)
- MI: Myocardial Infarction (ischemic tissue necrosis)
- STTC: ST/T Changes (repolarization abnormalities, ischemia)
- CD: Conduction Disturbance (bundle branch blocks, AV blocks)
- HYP: Hypertrophy (ventricular or atrial cardiac remodeling)

Data splits follow the official PTB-XL stratified 10-fold scheme:
- **Train cohort (Folds 1-8)**: 17,084 records
- **Validation cohort (Fold 9)**: 2,135 records
- **Holdout Test cohort (Fold 10)**: 2,137 records

---

## 4. Architecture and Project Structure

```
assignment3/
|-- .github/
|   |-- ISSUE_TEMPLATE/
|   |   |-- bug_report.md
|   |   `-- feature_request.md
|   `-- workflows/
|       `-- ci.yml
|-- figures/
|   |-- ecg_12lead_strip.png
|   |-- pipeline_architecture.png
|   `-- psd_comparison.png
|-- src/
|   `-- ptbxl_ecg/
|       |-- __init__.py
|       |-- cli.py
|       |-- config.py
|       |-- dataset.py
|       |-- download.py
|       |-- dsp.py
|       |-- metadata.py
|       |-- models.py
|       `-- viz.py
|-- tests/
|   |-- __init__.py
|   |-- test_cli.py
|   |-- test_dataset.py
|   |-- test_dsp.py
|   `-- test_metadata.py
|-- .gitignore
|-- Dockerfile
|-- environment.yml
|-- LICENSE
|-- PTB_XL_ECG_Preprocessing_and_DataLoader.ipynb
|-- pyproject.toml
|-- README.md
`-- requirements.txt
```

---

## 5. Installation and Setup

### Option 1: Virtual Environment (pip)
```bash
git clone https://github.com/seikoAyase/assignment3.git
cd assignment3

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Option 2: Conda / Mamba Environment
```bash
conda env create -f environment.yml
conda activate ptbxl-ecg
pip install -e .
```

### Option 3: Docker Container
```bash
docker build -t ptbxl-ecg:latest .
docker run --rm -v $(pwd)/data:/app/data ptbxl-ecg:latest python -m ptbxl_ecg.cli demo
```

---

## 6. Quickstart Usage

### A. Run Synthetic Pipeline Demo via CLI
```bash
python -m ptbxl_ecg.cli demo --output-dir figures --duration 10.0 --fs 100.0
```

### B. Python API Quickstart
```python
from ptbxl_ecg.dsp import generate_synthetic_ecg, preprocess_ecg_signal
from ptbxl_ecg.viz import plot_clinical_12lead, plot_psd_comparison

# 1. Generate 10-second synthetic 12-lead ECG (100 Hz sampling)
raw_ecg = generate_synthetic_ecg(duration_sec=10.0, fs=100.0)

# 2. Execute biomedical DSP pipeline
clean_ecg = preprocess_ecg_signal(raw_ecg, fs=100.0)

# 3. Render on clinical millimeter paper grid
fig = plot_clinical_12lead(clean_ecg, title="Clinical 12-Lead ECG (Filtered)")
fig.savefig("figures/clean_strip.png", dpi=150)
```

### C. PyTorch DataLoader Integration
```python
import pandas as pd
from ptbxl_ecg.dataset import PTBXLDataset, build_dataloader

df_meta = pd.read_csv("data/ptbxl/ptbxl_database.csv")
dataset = PTBXLDataset(df_meta, base_dir="data/ptbxl", preprocess=True)
dataloader = build_dataloader(dataset, batch_size=64, shuffle=True)

for signals, targets in dataloader:
    # signals shape: (64, 12, 1000) [Channels-first format]
    # targets shape: (64, 5)        [Multi-hot superclass vector]
    break
```

---

## 7. Automated Testing and CI/CD Quality Assurance

The test suite runs deterministically without external network access via synthetic ECG generators and mock metadata fixtures.

Run the test suite locally:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

GitHub Actions executes the following pipeline on every push and pull_request:
- Matrix testing: Python 3.10 and Python 3.11 on ubuntu-latest.
- Linting & code style: flake8 for PEP 8 compliance, black --check.
- Unit test coverage: unittest running all 16 test cases across DSP, metadata, dataset, models, and CLI.

---

## 8. Citation and Acknowledgements

If you utilize this software pipeline, please cite the underlying PTB-XL benchmark:
```bibtex
@article{wagner2020ptbxl,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and Kreiseler, Dieter and Lunze, Fatima I and Samek, Wojciech and Schaeffter, Tobias},
  journal={Scientific Data},
  volume={7},
  number={1},
  pages={154},
  year={2020},
  publisher={Nature Publishing Group}
}
```

---

## 9. License

This project is licensed under the terms of the [MIT License](LICENSE).
