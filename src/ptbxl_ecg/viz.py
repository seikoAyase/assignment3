"""Clinical visualization suite for 12-lead electrocardiograms and epidemiology."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

from ptbxl_ecg.config import (
    CLINICAL_LEAD_ORDER,
    SAMPLING_RATE,
    SUPERCLASSES,
)


def plot_clinical_12lead(
    sig: np.ndarray,
    fs: float = SAMPLING_RATE,
    title: str = "Standard 12-Lead Electrocardiogram",
    lead_names: List[str] = CLINICAL_LEAD_ORDER,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Render 12-lead ECG on standard clinical millimeter paper grid.

    Grid specification:
        - Minor grid: 0.04s (horizontal), 0.1 mV (vertical).
        - Major grid: 0.20s (horizontal), 0.5 mV (vertical).
        - Standard sweep speed: 25 mm/s.
        - Amplitude calibration: 10 mm/mV.

    Parameters:
        sig: Signal array of shape (N_samples, 12).
        fs: Sampling rate in Hertz.
        title: Plot title string.
        lead_names: List of 12 lead names.
        output_path: Optional path to save high-DPI figure image.

    Returns:
        Matplotlib figure object.
    """
    time = np.arange(sig.shape[0]) / fs
    fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for i, lead in enumerate(lead_names):
        ax = axes[i]
        ax.set_facecolor("#FFF5F5")  # Soft clinical pink background

        # Minor gridlines (0.04s horizontal, 0.5 normalized vertical)
        ax.set_xticks(np.arange(0, time[-1] + 0.04, 0.04), minor=True)
        ax.set_yticks(np.arange(-4, 4.5, 0.5), minor=True)
        ax.grid(which="minor", color="#F9D5D5", linestyle="-", linewidth=0.5, alpha=0.7)

        # Major gridlines (0.20s horizontal, 1.0 normalized vertical)
        ax.set_xticks(np.arange(0, time[-1] + 0.2, 0.2), minor=False)
        ax.set_yticks(np.arange(-4, 5.0, 1.0), minor=False)
        ax.grid(which="major", color="#ECA1A1", linestyle="-", linewidth=0.9, alpha=0.9)

        # Plot ECG lead trace
        lead_data = sig[:, i] if sig.ndim > 1 else sig
        ax.plot(time, lead_data, color="#1A202C", linewidth=1.1, label=lead)

        ax.set_ylabel(f"Lead {lead}", fontsize=10, fontweight="bold", labelpad=6)
        ax.set_ylim(-3.5, 3.5)
        ax.tick_params(colors="#4A5568", labelsize=8)

    axes[-2].set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    axes[-1].set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_psd_comparison(
    raw_sig: np.ndarray,
    filtered_sig: np.ndarray,
    fs: float = SAMPLING_RATE,
    lead_idx: int = 1,
    lead_name: str = "II",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot Welch Power Spectral Density (PSD) comparing raw vs filtered ECG.

    Parameters:
        raw_sig: Raw signal array of shape (N_samples, N_leads).
        filtered_sig: Filtered signal array of shape (N_samples, N_leads).
        fs: Sampling rate in Hertz.
        lead_idx: Channel index to evaluate (default 1 for Lead II).
        lead_name: Lead label string.
        output_path: Optional path to save figure.

    Returns:
        Matplotlib figure object.
    """
    f_raw, p_raw = signal.welch(raw_sig[:, lead_idx], fs=fs, nperseg=256)
    f_filt, p_filt = signal.welch(filtered_sig[:, lead_idx], fs=fs, nperseg=256)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(f_raw, p_raw, color="#E53E3E", linewidth=1.3, label="Raw Signal (Lead " + lead_name + ")")
    ax.semilogy(f_filt, p_filt, color="#2B6CB0", linewidth=1.5, label="Filtered Signal (0.5-45 Hz + Notch)")

    ax.axvline(0.5, color="#718096", linestyle="--", linewidth=0.9, label="Lowcut (0.5 Hz)")
    ax.axvline(45.0, color="#718096", linestyle="--", linewidth=0.9, label="Highcut (45.0 Hz)")
    ax.axvline(50.0, color="#D69E2E", linestyle=":", linewidth=1.2, label="Notch (50.0 Hz)")

    ax.set_title("Welch Power Spectral Density Comparison: Lead " + lead_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency (Hertz)", fontsize=10)
    ax.set_ylabel("Power Spectral Density (V^2 / Hz)", fontsize=10)
    ax.set_xlim(0, 50)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
