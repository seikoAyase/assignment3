"""PTB-XL ECG Preprocessing, DSP Pipeline, and PyTorch DataLoader Package.

Author: Zhanbolat Baxen
Course: Applied Software Development Project (CSE-2505M)
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Zhanbolat Baxen"

from ptbxl_ecg.config import (
    CLINICAL_LEAD_ORDER,
    SUPERCLASSES,
    SAMPLING_RATE,
    SIGNAL_LENGTH_SAMPLES,
    DEFAULT_LOWCUT,
    DEFAULT_HIGHCUT,
    DEFAULT_NOTCH_FREQ,
    NOTCH_QUALITY_FACTOR,
)
from ptbxl_ecg.dsp import (
    build_bandpass_filter,
    apply_notch_filter,
    zscore_normalize_signal,
    preprocess_ecg_signal,
    generate_synthetic_ecg,
)
from ptbxl_ecg.metadata import (
    parse_scp_statements,
    map_to_superclasses,
    create_stratified_splits,
    get_dataset_summary_statistics,
)

__all__ = [
    "CLINICAL_LEAD_ORDER",
    "SUPERCLASSES",
    "SAMPLING_RATE",
    "SIGNAL_LENGTH_SAMPLES",
    "DEFAULT_LOWCUT",
    "DEFAULT_HIGHCUT",
    "DEFAULT_NOTCH_FREQ",
    "NOTCH_QUALITY_FACTOR",
    "build_bandpass_filter",
    "apply_notch_filter",
    "zscore_normalize_signal",
    "preprocess_ecg_signal",
    "generate_synthetic_ecg",
    "parse_scp_statements",
    "map_to_superclasses",
    "create_stratified_splits",
    "get_dataset_summary_statistics",
]
