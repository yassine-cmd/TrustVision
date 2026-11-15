"""
TrustVision - Model Module (src/model.py)
Shared Model Contract:
- Backbone CNN (TrustVisionCNN) taking input tensors in [0, 1] range of size 96x96.
- Internal mean/std normalization layer so gradient attacks (FGSM) operate directly on [0, 1] pixels.
- Exposes `get_features(x)` for SimCLR / linear probing.
- Exposes `get_last_conv_layer()` for Grad-CAM explainability.
"""

from typing import Optional
import torch
import torch.nn as nn


class InputNormalization(nn.Module):
    """
    Internal differentiable normalization layer.
    Applies (x - mean) / std inside the forward pass, ensuring external inputs
    remain cleanly bounded in [0, 1] for FGSM adversarial attacks.
    """

    def __init__(
        self,
        mean: Optional[tuple] = (0.485, 0.456, 0.406),
        std: Optional[tuple] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class TrustVisionCNN(nn.Module):
    """
    Standard CNN backbone for the TrustVision pipeline.
    Architecture:
      Input (3, 96, 96) in [0, 1]
      -> InputNormalization
      -> 4 ConvBlocks (32 -> 64 -> 128 -> 256 channels)
      -> AdaptiveAvgPool2d(1, 1)
      -> Flatten (256-d features)
      -> Linear Classification Head (256 -> num_classes)
    """

    def __init__(
        self,
        num_classes: int = 6,
        feature_dim: int = 256,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Input normalization wrapper
        self.norm = InputNormalization() if normalize_input else nn.Identity()

        # Feature extractor
        self.block1 = ConvBlock(3, 32, pool=True)     # 96 -> 48
        self.block2 = ConvBlock(32, 64, pool=True)    # 48 -> 24
        self.block3 = ConvBlock(64, 128, pool=True)   # 24 -> 12
        self.block4 = ConvBlock(128, 256, pool=True)  # 12 -> 6

        expected_feature_dim = self.block4.conv.out_channels
        if feature_dim != expected_feature_dim:
            raise ValueError(
                f"feature_dim={feature_dim} does not match block4 output channels "
                f"({expected_feature_dim})."
            )
        self.feature_dim = expected_feature_dim

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)

        # Classification Head
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def get_last_conv_layer(self) -> nn.Conv2d:
        """
        Required contract for Member B (Grad-CAM).
        Returns the final convolutional layer of the network.
        """
        return self.block4.conv

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Required contract for Member C (SimCLR) and Member A (Linear Probe).
        Extracts 256-dimensional feature representations.
        Input x must be in range [0, 1].
        """
        x = self.norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        features = torch.flatten(x, 1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw logits."""
        features = self.get_features(x)
        features = self.dropout(features)
        logits = self.fc(features)
        return logits


class SimCLRProjectionHead(nn.Module):
    """
    MLP Projection head used during SimCLR contrastive pre-training (Theme 8).
    Maps feature_dim (256) -> hidden_dim (256) -> proj_dim (128).
    Discarded after pre-training when training classifiers on frozen representations.
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
