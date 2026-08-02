from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FeatureMLP(nn.Module):
    def __init__(
        self,
        feature_count: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()

        if feature_count <= 0:
            raise ValueError("feature_count must be positive")

        if not hidden_dims:
            raise ValueError("At least one hidden layer is required")

        if any(width <= 0 for width in hidden_dims):
            raise ValueError("Hidden dimensions must be positive")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = [
            nn.LayerNorm(feature_count),
        ]

        input_width = feature_count

        for output_width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_width, output_width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )

            input_width = output_width

        layers.append(nn.Linear(input_width, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                "FeatureMLP expects a two-dimensional tensor"
            )

        return self.network(features).squeeze(-1)
