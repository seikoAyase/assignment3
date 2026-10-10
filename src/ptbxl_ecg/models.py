"""Deep Learning Architecture for 12-Lead ECG: 1D Squeeze-and-Excitation ResNet."""

from typing import Optional, List
import numpy as np

from ptbxl_ecg.config import NUM_LEADS, SUPERCLASSES

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object


if TORCH_AVAILABLE:

    class SEBlock1D(nn.Module):
        """1D Squeeze-and-Excitation attention block across ECG channels.

        Learns inter-lead dependencies to adaptively weight the diagnostic
        importance of different ECG leads (e.g. precordial V1-V6 vs limb leads).
        """

        def __init__(self, channels: int, reduction: int = 8):
            super().__init__()
            reduced = max(channels // reduction, 8)
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(channels, reduced, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(reduced, channels, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            scale = self.fc(x).unsqueeze(-1)
            return x * scale

    class SEResBlock1D(nn.Module):
        """1D Residual Block with wide temporal kernel (k=7) and SE attention."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
        ):
            super().__init__()
            self.conv1 = nn.Conv1d(
                in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv1d(
                out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False
            )
            self.bn2 = nn.BatchNorm1d(out_channels)

            self.se = SEBlock1D(out_channels)
            self.downsample = downsample

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out = self.se(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)
            return out

    class SEResNet1D(nn.Module):
        """Complete 1D-SE-ResNet for 12-Lead ECG Classification.

        Features:
            - Convolutional stem with kernel 15 for capturing P-Q-R-S-T morphology.
            - 4 residual stages with channel dimensions [64, 128, 256, 512].
            - Global adaptive average pooling and dropout regularization.
            - Linear head outputting unnormalized logits for BCEWithLogitsLoss.
        """

        def __init__(
            self,
            in_channels: int = NUM_LEADS,
            num_classes: int = len(SUPERCLASSES),
            base_filters: int = 64,
            dropout_rate: float = 0.3,
        ):
            super().__init__()

            # Stem
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )

            # Residual stages
            self.stage1 = self._make_stage(base_filters, base_filters, stride=1)
            self.stage2 = self._make_stage(base_filters, base_filters * 2, stride=2)
            self.stage3 = self._make_stage(base_filters * 2, base_filters * 4, stride=2)
            self.stage4 = self._make_stage(base_filters * 4, base_filters * 8, stride=2)

            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Linear(base_filters * 8, num_classes)

        def _make_stage(self, in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
            downsample = None
            if stride != 1 or in_ch != out_ch:
                downsample = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_ch),
                )
            block = SEResBlock1D(in_ch, out_ch, stride=stride, downsample=downsample)
            return nn.Sequential(block)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.stem(x)
            out = self.stage1(out)
            out = self.stage2(out)
            out = self.stage3(out)
            out = self.stage4(out)
            out = self.pool(out)
            out = torch.flatten(out, 1)
            out = self.dropout(out)
            logits = self.classifier(out)
            return logits

else:

    class SEResNet1D:
        """Mock fallback for environments lacking PyTorch."""

        def __init__(self, in_channels: int = NUM_LEADS, num_classes: int = len(SUPERCLASSES)):
            self.in_channels = in_channels
            self.num_classes = num_classes

        def forward(self, x: np.ndarray) -> np.ndarray:
            batch_size = x.shape[0] if hasattr(x, "shape") else 1
            return np.zeros((batch_size, self.num_classes), dtype=np.float32)
