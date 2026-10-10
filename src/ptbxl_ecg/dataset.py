"""Dataset and DataLoader abstractions for PTB-XL ECG waveforms."""

from pathlib import Path
from typing import List, Tuple, Optional, Callable
import numpy as np
import pandas as pd

from ptbxl_ecg.config import (
    SUPERCLASSES,
    SAMPLING_RATE,
    DEFAULT_DATA_DIR,
    SIGNAL_LENGTH_SAMPLES,
    NUM_LEADS,
)
from ptbxl_ecg.dsp import preprocess_ecg_signal

# Attempt to import PyTorch; provide seamless graceful fallback if absent
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object
    DataLoader = None

# Attempt to import WFDB for raw waveform binary parsing
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False


def load_raw_ecg_record(
    record_rel_path: str,
    base_dir: Path = DEFAULT_DATA_DIR,
) -> Tuple[np.ndarray, dict]:
    """Read a 12-lead raw ECG waveform and metadata using WFDB.

    Parameters:
        record_rel_path: Relative path to record without extension.
        base_dir: Base directory containing extracted PTB-XL records.

    Returns:
        Tuple of (signal_array [samples, leads], fields_dict).
    """
    if not WFDB_AVAILABLE:
        raise ImportError(
            "WFDB package is not installed. Install via 'pip install wfdb' to read binary records."
        )

    full_path = str(Path(base_dir) / record_rel_path)
    sig, fields = wfdb.rdsamp(full_path)
    return sig.astype(np.float32), fields


class PTBXLDataset(Dataset):
    """Production-grade PyTorch Dataset for PTB-XL 12-Lead Electrocardiograms.

    Produces channels-first tensors of shape (12, 1000) paired with multi-hot
    diagnostic superclass target vectors of shape (5,).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: Path = DEFAULT_DATA_DIR,
        fs: float = SAMPLING_RATE,
        superclasses: List[str] = SUPERCLASSES,
        preprocess: bool = True,
        transform: Optional[Callable] = None,
        synthetic_mode: bool = False,
    ):
        """Initialize PTBXLDataset.

        Parameters:
            df: Metadata DataFrame containing filenames and superclass labels.
            base_dir: Base folder path containing the PTB-XL records.
            fs: Target sampling frequency in Hertz (default 100 Hz).
            superclasses: Diagnostic target column names.
            preprocess: If True, applies full bandpass, notch, and z-score filters.
            transform: Optional callable data augmentation function.
            synthetic_mode: If True, generates synthetic waveforms for testing.
        """
        self.df = df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.fs = fs
        self.superclasses = superclasses
        self.preprocess = preprocess
        self.transform = transform
        self.synthetic_mode = synthetic_mode

        if "filename_lr" in self.df.columns:
            self.filenames = self.df["filename_lr"].values
        elif "filename_hr" in self.df.columns:
            self.filenames = self.df["filename_hr"].values
        else:
            self.filenames = np.array([f"synthetic_{i}" for i in range(len(self.df))])

        # Extract targets
        valid_cols = [c for c in self.superclasses if c in self.df.columns]
        if valid_cols:
            self.labels = self.df[valid_cols].values.astype(np.float32)
        else:
            self.labels = np.zeros((len(self.df), len(self.superclasses)), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if self.synthetic_mode or not WFDB_AVAILABLE:
            from ptbxl_ecg.dsp import generate_synthetic_ecg
            raw_sig = generate_synthetic_ecg(
                duration_sec=10.0,
                fs=self.fs,
                num_leads=NUM_LEADS,
                random_state=idx,
            )
        else:
            rel_path = self.filenames[idx]
            raw_sig, _ = load_raw_ecg_record(rel_path, base_dir=self.base_dir)

        # Preprocess signal
        if self.preprocess:
            proc_sig = preprocess_ecg_signal(raw_sig, fs=self.fs)
        else:
            proc_sig = raw_sig.astype(np.float32)

        # Transpose from (samples, channels) to (channels, samples) for 1D CNNs
        sig_tensor = np.transpose(proc_sig, (1, 0))  # (12, 1000)

        # Ensure correct temporal length
        if sig_tensor.shape[1] != SIGNAL_LENGTH_SAMPLES:
            if sig_tensor.shape[1] > SIGNAL_LENGTH_SAMPLES:
                sig_tensor = sig_tensor[:, :SIGNAL_LENGTH_SAMPLES]
            else:
                pad_width = SIGNAL_LENGTH_SAMPLES - sig_tensor.shape[1]
                sig_tensor = np.pad(sig_tensor, ((0, 0), (0, pad_width)), mode="constant")

        target = self.labels[idx]

        if self.transform is not None:
            sig_tensor = self.transform(sig_tensor)

        if TORCH_AVAILABLE:
            return torch.from_numpy(sig_tensor), torch.from_numpy(target)
        return sig_tensor, target


def build_dataloader(
    dataset: PTBXLDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
):
    """Construct a PyTorch DataLoader for batching and multi-worker loading."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to build a DataLoader.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
