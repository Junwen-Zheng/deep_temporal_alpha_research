from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional


class CausalConv1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError(
                "input_channels must be positive"
            )

        if output_channels <= 0:
            raise ValueError(
                "output_channels must be positive"
            )

        if kernel_size <= 0:
            raise ValueError(
                "kernel_size must be positive"
            )

        if dilation <= 0:
            raise ValueError("dilation must be positive")

        self.left_padding = (
            kernel_size - 1
        ) * dilation

        self.convolution = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                "CausalConv1d expects a three-dimensional tensor"
            )

        padded = functional.pad(
            features,
            (self.left_padding, 0),
        )

        return self.convolution(padded)


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.convolution = CausalConv1d(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.normalization = nn.LayerNorm(
            output_channels
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        if input_channels == output_channels:
            self.residual_projection: nn.Module = (
                nn.Identity()
            )
        else:
            self.residual_projection = nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=1,
            )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.residual_projection(features)

        output = self.convolution(features)

        output = output.transpose(1, 2)
        output = self.normalization(output)
        output = output.transpose(1, 2)

        output = self.activation(output)
        output = self.dropout(output)

        return self.activation(output + residual)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        feature_count: int,
        channel_count: int,
        kernel_size: int,
        dilations: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()

        if feature_count <= 0:
            raise ValueError(
                "feature_count must be positive"
            )

        if channel_count <= 0:
            raise ValueError(
                "channel_count must be positive"
            )

        if kernel_size <= 0:
            raise ValueError(
                "kernel_size must be positive"
            )

        dilation_values = tuple(
            int(value) for value in dilations
        )

        if not dilation_values:
            raise ValueError(
                "At least one dilation is required"
            )

        if any(value <= 0 for value in dilation_values):
            raise ValueError(
                "Dilations must be positive"
            )

        self.feature_count = feature_count
        self.channel_count = channel_count
        self.kernel_size = kernel_size
        self.dilations = dilation_values

        self.receptive_field = (
            1
            + (kernel_size - 1)
            * sum(dilation_values)
        )

        self.input_projection = nn.Conv1d(
            in_channels=feature_count,
            out_channels=channel_count,
            kernel_size=1,
        )

        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    input_channels=channel_count,
                    output_channels=channel_count,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilation_values
            ]
        )

        self.output_normalization = nn.LayerNorm(
            channel_count
        )

        self.output_head = nn.Linear(
            channel_count,
            1,
        )

        nn.init.normal_(
            self.output_head.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.zeros_(self.output_head.bias)

    def encode_sequence(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        if sequences.ndim != 3:
            raise ValueError(
                "TemporalConvNet expects "
                "(batch, sequence, feature)"
            )

        if sequences.shape[2] != self.feature_count:
            raise ValueError(
                "Input feature count differs from the model"
            )

        if sequences.shape[1] < self.receptive_field:
            raise ValueError(
                "Input sequence is shorter than the "
                "TCN receptive field"
            )

        output = sequences.transpose(1, 2)
        output = self.input_projection(output)

        for block in self.blocks:
            output = block(output)

        output = output.transpose(1, 2)

        return self.output_normalization(output)

    def forward(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode_sequence(sequences)

        endpoint_representation = encoded[:, -1, :]

        return self.output_head(
            endpoint_representation
        ).squeeze(-1)
