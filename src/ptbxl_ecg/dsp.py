"""Digital Signal Processing (DSP) Module for Multi-Lead ECG Signals.

Implements zero-phase Butterworth bandpass filtering, adaptive IIR notch
filtering for powerline interference suppression, and per-lead z-score
normalization with flatline protection.
"""

from typing import Tuple, Optional
import numpy as np
import scipy.signal as signal

from ptbxl_ecg.config import (
    DEFAULT_LOWCUT,
    DEFAULT_HIGHCUT,
    FILTER_ORDER,
    DEFAULT_NOTCH_FREQ,
    NOTCH_QUALITY_FACTOR,
    EPSILON_FLATLINE,
    SAMPLING_RATE,
    NUM_LEADS,
)


def build_bandpass_filter(
    lowcut: float = DEFAULT_LOWCUT,
    highcut: float = DEFAULT_HIGHCUT,
    fs: float = SAMPLING_RATE,
    order: int = FILTER_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a digital Butterworth bandpass filter.

    Parameters:
        lowcut: Lower cutoff frequency in Hertz (removes baseline wander).
        highcut: Upper cutoff frequency in Hertz (suppresses EMG noise).
        fs: Sampling frequency in Hertz.
        order: Order of the Butterworth filter.

    Returns:
        Tuple of numerator (b) and denominator (a) polynomial coefficients.
    """
    if lowcut <= 0:
        raise ValueError(f"lowcut must be positive, got {lowcut}")
    if highcut <= lowcut:
        raise ValueError(f"highcut ({highcut}) must be strictly greater than lowcut ({lowcut})")
    nyquist = 0.5 * fs
    if highcut >= nyquist:
        # Clamp highcut to just below Nyquist if needed
        highcut = nyquist - 1e-3

    low_norm = lowcut / nyquist
    high_norm = highcut / nyquist
    b, a = signal.butter(order, [low_norm, high_norm], btype="bandpass")
    return b, a


def apply_notch_filter(
    sig: np.ndarray,
    f0: float = DEFAULT_NOTCH_FREQ,
    fs: float = SAMPLING_RATE,
    Q: float = NOTCH_QUALITY_FACTOR,
) -> np.ndarray:
    """Apply an IIR notch filter to remove powerline interference.

    Parameters:
        sig: Array of shape (N_samples, N_channels) or (N_samples,).
        f0: Notch center frequency in Hertz (typically 50 Hz or 60 Hz).
        fs: Sampling frequency in Hertz.
        Q: Quality factor defining notch bandwidth.

    Returns:
        Filtered signal array of identical shape.
    """
    nyquist = 0.5 * fs
    # Notch is physically realizable only if f0 is strictly below Nyquist
    if f0 < nyquist:
        w0 = f0 / nyquist
        b_notch, a_notch = signal.iirnotch(w0=w0, Q=Q)
        return signal.filtfilt(b_notch, a_notch, sig, axis=0)
    return sig


def zscore_normalize_signal(
    sig: np.ndarray,
    eps: float = EPSILON_FLATLINE,
) -> np.ndarray:
    """Normalize each lead independently using temporal z-score normalization.

    Computes:
        x_norm = (x - mean(x)) / (std(x) + eps)

    Includes safeguard against disconnected or flatline leads where std < eps.

    Parameters:
        sig: Array of shape (N_samples, N_channels) or (N_samples,).
        eps: Small numerical constant to prevent zero-division.

    Returns:
        Normalized array of identical shape and float32 dtype.
    """
    means = np.mean(sig, axis=0, keepdims=True)
    stds = np.std(sig, axis=0, keepdims=True)

    # Replace zero or near-zero standard deviation with 1.0 to prevent inf/nan
    stds = np.where(stds < eps, 1.0, stds)

    normalized = (sig - means) / (stds + eps)
    return normalized.astype(np.float32)


def preprocess_ecg_signal(
    raw_sig: np.ndarray,
    fs: float = SAMPLING_RATE,
    lowcut: float = DEFAULT_LOWCUT,
    highcut: float = DEFAULT_HIGHCUT,
    notch_f0: float = DEFAULT_NOTCH_FREQ,
    Q: float = NOTCH_QUALITY_FACTOR,
) -> np.ndarray:
    """Execute complete biomedical digital signal processing pipeline.

    Steps:
        1. Numerical hygiene: Replace NaNs and Infinities with zero.
        2. Zero-phase Butterworth bandpass filtering (0.5 to 45.0 Hz).
        3. Adaptive powerline notch filtering (50.0 Hz).
        4. Temporal per-lead z-score standardization with flatline guard.

    Parameters:
        raw_sig: Input signal array of shape (N_samples, N_channels).
        fs: Sampling frequency in Hertz.
        lowcut: Low-frequency cutoff in Hertz.
        highcut: High-frequency cutoff in Hertz.
        notch_f0: Powerline hum frequency in Hertz.
        Q: Quality factor for notch filter.

    Returns:
        Normalized, artifact-suppressed float32 numpy array.
    """
    # 1. Clean non-finite entries
    cleaned = np.nan_to_num(raw_sig, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Zero-phase bandpass filter
    b_bp, a_bp = build_bandpass_filter(lowcut=lowcut, highcut=highcut, fs=fs, order=FILTER_ORDER)
    filtered = signal.filtfilt(b_bp, a_bp, cleaned, axis=0)

    # 3. Powerline notch filter
    filtered = apply_notch_filter(filtered, f0=notch_f0, fs=fs, Q=Q)

    # 4. Per-lead z-score standardization
    normalized = zscore_normalize_signal(filtered)

    return normalized


def generate_synthetic_ecg(
    duration_sec: float = 10.0,
    fs: float = SAMPLING_RATE,
    num_leads: int = NUM_LEADS,
    heart_rate_bpm: float = 60.0,
    noise_level: float = 0.05,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """Generate synthetic 12-lead ECG waveforms for testing and validation.

    Synthesizes a periodic QRS-like cardiac waveform with baseline wander
    and powerline noise for deterministic unit test execution.

    Parameters:
        duration_sec: Total duration in seconds.
        fs: Sampling frequency in Hertz.
        num_leads: Number of ECG channels (default 12).
        heart_rate_bpm: Heart rate in beats per minute.
        noise_level: Gaussian noise amplitude standard deviation.
        random_state: Seed for pseudo-random number generator.

    Returns:
        Synthetic raw signal array of shape (num_samples, num_leads).
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    num_samples = int(duration_sec * fs)
    t = np.arange(num_samples) / fs

    f_cardiac = heart_rate_bpm / 60.0  # Hz

    # Composite QRS approximation: sharp R-wave and P/T waves
    cardiac_cycle = (
        1.0 * np.exp(-((t % (1.0 / f_cardiac) - 0.2) ** 2) / (2 * 0.015 ** 2))
        - 0.2 * np.exp(-((t % (1.0 / f_cardiac) - 0.17) ** 2) / (2 * 0.01 ** 2))
        - 0.3 * np.exp(-((t % (1.0 / f_cardiac) - 0.24) ** 2) / (2 * 0.012 ** 2))
        + 0.15 * np.sin(2 * np.pi * f_cardiac * t)
    )

    signals = np.zeros((num_samples, num_leads), dtype=np.float32)

    for ch in range(num_leads):
        lead_scale = 0.8 + 0.4 * np.sin(ch)
        # Add baseline wander (0.15 Hz sinusoidal drift)
        drift = 0.2 * np.sin(2 * np.pi * 0.15 * t + ch)
        # Add powerline interference (50 Hz)
        powerline = 0.08 * np.sin(2 * np.pi * 50.0 * t)
        # Add random high-frequency measurement noise
        noise = rng.normal(0.0, noise_level, size=num_samples)

        signals[:, ch] = lead_scale * cardiac_cycle + drift + powerline + noise

    return signals
