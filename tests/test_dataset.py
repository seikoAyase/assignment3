"""Unit tests for the PTBXLDataset and 1D-SE-ResNet model architecture."""

import unittest
import numpy as np
import pandas as pd

from ptbxl_ecg.dataset import PTBXLDataset
from ptbxl_ecg.models import SEResNet1D
from ptbxl_ecg.config import SUPERCLASSES, NUM_LEADS, SIGNAL_LENGTH_SAMPLES


class TestDataset(unittest.TestCase):
    """Test suite for PTBXLDataset indexing and tensor formatting."""

    def setUp(self):
        # Create small synthetic metadata DataFrame
        self.df = pd.DataFrame(
            {
                "filename_lr": ["synth_0", "synth_1", "synth_2"],
                "NORM": [1, 0, 0],
                "MI": [0, 1, 0],
                "STTC": [0, 0, 1],
                "CD": [0, 0, 0],
                "HYP": [0, 1, 0],
            }
        )

    def test_synthetic_dataset_len_and_item(self):
        """Test dataset length and channels-first tensor item shapes."""
        ds = PTBXLDataset(self.df, synthetic_mode=True, preprocess=True)
        self.assertEqual(len(ds), 3)

        sig, label = ds[0]
        # Channels-first shape: (12 leads, 1000 temporal samples)
        self.assertEqual(sig.shape, (NUM_LEADS, SIGNAL_LENGTH_SAMPLES))
        # Superclass multi-hot label vector: (5,)
        self.assertEqual(label.shape, (len(SUPERCLASSES),))

        # Check label content for sample 0
        np.testing.assert_array_equal(label, np.array([1, 0, 0, 0, 0], dtype=np.float32))

        # Check label content for sample 1 (MI=1, HYP=1)
        _, label1 = ds[1]
        np.testing.assert_array_equal(label1, np.array([0, 1, 0, 0, 1], dtype=np.float32))

    def test_custom_transform(self):
        """Test custom transform integration into dataset pipeline."""
        def scale_transform(x):
            return x * 2.0

        ds = PTBXLDataset(self.df, synthetic_mode=True, transform=scale_transform)
        sig, _ = ds[0]
        self.assertEqual(sig.shape, (NUM_LEADS, SIGNAL_LENGTH_SAMPLES))


class TestModels(unittest.TestCase):
    """Test suite for neural network model creation and forward pass."""

    def test_model_initialization(self):
        """Test model creation with specified channels and class heads."""
        model = SEResNet1D(in_channels=NUM_LEADS, num_classes=len(SUPERCLASSES))
        self.assertIsNotNone(model)

    def test_mock_or_torch_forward(self):
        """Test forward pass shape preservation."""
        model = SEResNet1D(in_channels=NUM_LEADS, num_classes=len(SUPERCLASSES))
        x = np.zeros((4, NUM_LEADS, 1000), dtype=np.float32)

        try:
            import torch
            t_x = torch.from_numpy(x)
            out = model(t_x)
            self.assertEqual(out.shape, (4, len(SUPERCLASSES)))
        except (ImportError, TypeError):
            out = model.forward(x)
            self.assertEqual(out.shape, (4, len(SUPERCLASSES)))


if __name__ == "__main__":
    unittest.main()
