"""Global configuration parameters and constants for the PTB-XL ECG pipeline."""

from pathlib import Path
from typing import List

# Standard 12-lead clinical ordering according to international convention
CLINICAL_LEAD_ORDER: List[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

# The five official diagnostic superclasses defined by PTB-XL benchmark
SUPERCLASSES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]

# Superclass descriptions
SUPERCLASS_DESCRIPTIONS = {
    "NORM": "Normal Electrocardiogram",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy"
}

# Signal sampling characteristics
SAMPLING_RATE: float = 100.0  # Hz (low-resolution records100)
SIGNAL_DURATION_SEC: float = 10.0  # Standard 10-second acquisition
SIGNAL_LENGTH_SAMPLES: int = int(SAMPLING_RATE * SIGNAL_DURATION_SEC)  # 1000 samples
NUM_LEADS: int = 12

# Digital Signal Processing (DSP) filter parameters
DEFAULT_LOWCUT: float = 0.5  # Hz, baseline wander cutoff
DEFAULT_HIGHCUT: float = 45.0  # Hz, high-frequency noise cutoff
FILTER_ORDER: int = 4  # 4th-order Butterworth
DEFAULT_NOTCH_FREQ: float = 50.0  # Hz, European powerline frequency
NOTCH_QUALITY_FACTOR: float = 30.0  # Quality factor Q for notch sharpness
EPSILON_FLATLINE: float = 1e-6  # Guard against zero-division in flatline leads

# Remote PhysioNet archive endpoint
PHYSIONET_PTBXL_URL: str = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
DEFAULT_DATA_DIR: Path = Path("data/ptbxl")
DEFAULT_ZIP_NAME: str = "ptb-xl.zip"

# Deterministic random seed
RANDOM_SEED: int = 42

# Train / Validation / Test fold splitting convention (recommended by PhysioNet)
TRAIN_FOLDS: List[int] = list(range(1, 9))
VAL_FOLDS: List[int] = [9]
TEST_FOLDS: List[int] = [10]
