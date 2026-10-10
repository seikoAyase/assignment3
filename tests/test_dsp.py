"""Unit tests for the Digital Signal Processing (DSP) module."""

import unittest
import numpy as np

from ptbxl_ecg.dsp import (
    build_bandpass_filter,
    apply_notch_filter,
    zscore_normalize_signal,
    preprocess_ecg_signal,
    generate_synthetic_ecg,
)
from ptbxl_ecg.config import SAMPLING_RATE, NUM_LEADS


class TestECGDSP(unittest.TestCase):
    """Test suite for digital filtering and normalization functions."""

    def test_build_bandpass_filter(self):
        """Test Butterworth bandpass filter coefficient generation."""
        b, a = build_bandpass_filter(lowcut=0.5, highcut=45.0, fs=100.0, order=4)
        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(len(b), 9)  # 2 * order + 1 for bandpass
        self.assertEqual(len(a), 9)

        # Negative lowcut or invalid frequency range should raise ValueError
        with self.assertRaises(ValueError):
            build_bandpass_filter(lowcut=-1.0, highcut=45.0, fs=100.0)

        with self.assertRaises(ValueError):
            build_bandpass_filter(lowcut=45.0, highcut=40.0, fs=100.0)

    def test_apply_notch_filter_attenuation(self):
        """Test notch filter suppression of 50 Hz powerline tone."""
        fs = 120.0  # Nyquist = 60 Hz (> 50 Hz)
        t = np.arange(600) / fs
        # 50 Hz pure sinusoid
        sig_50hz = np.sin(2 * np.pi * 50.0 * t)[:, np.newaxis]

        filtered = apply_notch_filter(sig_50hz, f0=50.0, fs=fs, Q=30.0)
        # Power should be heavily attenuated
        raw_power = np.mean(sig_50hz ** 2)
        filt_power = np.mean(filtered ** 2)
        self.assertLess(filt_power, 0.2 * raw_power)

    def test_zscore_normalize_signal(self):
        """Test temporal per-lead standardization to zero-mean and unit variance."""
        rng = np.random.default_rng(123)
        # Random signal with arbitrary mean and standard deviation
        sig = rng.normal(loc=15.0, scale=4.0, size=(1000, 12)).astype(np.float32)

        norm_sig = zscore_normalize_signal(sig)
        self.assertEqual(norm_sig.shape, (1000, 12))
        self.assertEqual(norm_sig.dtype, np.float32)

        # Means should be very close to 0.0
        np.testing.assert_allclose(np.mean(norm_sig, axis=0), 0.0, atol=1e-5)
        # Standard deviations should be very close to 1.0
        np.testing.assert_allclose(np.std(norm_sig, axis=0), 1.0, atol=1e-3)

    def test_zscore_flatline_guard(self):
        """Verify that constant/flatline leads do not produce NaNs or infinities."""
        sig = np.zeros((1000, 12), dtype=np.float32)
        # Make one lead constant non-zero
        sig[:, 3] = 2.5

        norm_sig = zscore_normalize_signal(sig)
        self.assertFalse(np.isnan(norm_sig).any())
        self.assertFalse(np.isinf(norm_sig).any())

    def test_preprocess_ecg_signal_end_to_end(self):
        """Verify end-to-end preprocessing on synthetic 12-lead ECG."""
        raw_sig = generate_synthetic_ecg(duration_sec=10.0, fs=100.0, num_leads=12, random_state=42)
        clean_sig = preprocess_ecg_signal(raw_sig, fs=100.0)

        self.assertEqual(clean_sig.shape, (1000, 12))
        self.assertEqual(clean_sig.dtype, np.float32)
        self.assertFalse(np.isnan(clean_sig).any())
        self.assertFalse(np.isinf(clean_sig).any())

    def test_generate_synthetic_ecg_determinism(self):
        """Verify that synthetic ECG generation is deterministic with fixed seed."""
        sig1 = generate_synthetic_ecg(duration_sec=5.0, fs=SAMPLING_RATE, random_state=100)
        sig2 = generate_synthetic_ecg(duration_sec=5.0, fs=SAMPLING_RATE, random_state=100)
        np.testing.assert_array_equal(sig1, sig2)
        self.assertEqual(sig1.shape, (500, NUM_LEADS))


if __name__ == "__main__":
    unittest.main()
