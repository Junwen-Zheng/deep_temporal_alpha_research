from __future__ import annotations

import torch
from torch import nn


class TemporalPatchTransformer(nn.Module):
    def __init__(
        self,
        feature_count: int,
        sequence_length: int,
        patch_size: int,
        model_dimension: int,
        attention_heads: int,
        layer_count: int,
        feedforward_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()

        positive_values = {
            "feature_count": feature_count,
            "sequence_length": sequence_length,
            "patch_size": patch_size,
            "model_dimension": model_dimension,
            "attention_heads": attention_heads,
            "layer_count": layer_count,
            "feedforward_dimension": (
                feedforward_dimension
            ),
        }

        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if sequence_length % patch_size != 0:
            raise ValueError(
                "sequence_length must be divisible "
                "by patch_size"
            )

        if model_dimension % attention_heads != 0:
            raise ValueError(
                "model_dimension must be divisible "
                "by attention_heads"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        self.feature_count = feature_count
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.patch_count = (
            sequence_length // patch_size
        )
        self.model_dimension = model_dimension

        patch_dimension = (
            patch_size * feature_count
        )

        self.patch_projection = nn.Linear(
            patch_dimension,
            model_dimension,
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.patch_count,
                model_dimension,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=attention_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=layer_count,
            enable_nested_tensor=False,
        )

        self.output_normalization = nn.LayerNorm(
            model_dimension
        )

        self.output_head = nn.Linear(
            model_dimension,
            1,
        )

        causal_mask = torch.triu(
            torch.ones(
                self.patch_count,
                self.patch_count,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.output_head.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.zeros_(self.output_head.bias)

    def patchify(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        if sequences.ndim != 3:
            raise ValueError(
                "Transformer expects "
                "(batch, sequence, feature)"
            )

        if sequences.shape[1] != self.sequence_length:
            raise ValueError(
                "Input sequence length differs "
                "from the configured length"
            )

        if sequences.shape[2] != self.feature_count:
            raise ValueError(
                "Input feature count differs "
                "from the configured feature count"
            )

        batch_size = sequences.shape[0]

        return sequences.reshape(
            batch_size,
            self.patch_count,
            self.patch_size * self.feature_count,
        )

    def encode_tokens(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        patches = self.patchify(sequences)

        tokens = self.patch_projection(patches)

        tokens = (
            tokens + self.position_embedding
        )

        encoded = self.encoder(
            tokens,
            mask=self.causal_mask,
        )

        return self.output_normalization(encoded)

    def forward(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode_tokens(sequences)

        endpoint_representation = encoded[:, -1, :]

        return self.output_head(
            endpoint_representation
        ).squeeze(-1)
