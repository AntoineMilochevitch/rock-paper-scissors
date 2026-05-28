from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Simple convolutional block used by the baseline CNN."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, use_batchnorm: bool, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class BaselineCNN(nn.Module):
    """Baseline CNN for rock-paper-scissors classification.

    The architecture stays intentionally compact so it can serve as the first
    from-scratch reference model in the experiment grid.
    """

    def __init__(
        self,
        num_classes: int = 3,
        channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        feature_layers: list[nn.Module] = []
        in_channels = 3
        for index, out_channels in enumerate(channels):
            # Each block extracts progressively richer spatial features.
            block_dropout = dropout * 0.5 if index < len(channels) - 1 else dropout
            feature_layers.append(ConvBlock(in_channels, out_channels, kernel_size, use_batchnorm, block_dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*feature_layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.global_pool(features)
        return self.classifier(pooled)
