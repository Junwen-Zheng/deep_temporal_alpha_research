from __future__ import annotations

import pytest
import torch

from deep_alpha.models.transformer import (
    TemporalPatchTransformer,
)


def build_model(
    dropout: float = 0.0,
) -> TemporalPatchTransformer:
    return TemporalPatchTransformer(
        feature_count=16,
        sequence_length=128,
        patch_size=4,
        model_dimension=32,
        attention_heads=4,
        layer_count=2,
        feedforward_dimension=64,
        dropout=dropout,
    )


def test_transformer_output_shape() -> None:
    model = build_model(dropout=0.1)

    sequences = torch.zeros(
        (10, 128, 16),
        dtype=torch.float32,
    )

    predictions = model(sequences)

    assert predictions.shape == (10,)
    assert torch.isfinite(predictions).all()


def test_transformer_uses_expected_patch_count() -> None:
    model = build_model()

    assert model.patch_count == 32

    sequences = torch.zeros(
        (2, 128, 16),
        dtype=torch.float32,
    )

    tokens = model.patchify(sequences)

    assert tokens.shape == (2, 32, 64)


def test_causal_attention_blocks_future_patches() -> None:
    torch.manual_seed(42)

    model = build_model()
    model.eval()

    original = torch.randn(
        (1, 128, 16),
        dtype=torch.float32,
    )

    modified = original.clone()
    modified[:, 64:, :] += 100.0

    with torch.inference_mode():
        original_encoded = model.encode_tokens(
            original
        )

        modified_encoded = model.encode_tokens(
            modified
        )

    torch.testing.assert_close(
        original_encoded[:, :16, :],
        modified_encoded[:, :16, :],
        rtol=0.0,
        atol=1e-6,
    )


def test_transformer_rejects_wrong_sequence_length() -> None:
    model = build_model()

    with pytest.raises(
        ValueError,
        match="sequence length differs",
    ):
        model(
            torch.zeros(
                (2, 127, 16),
                dtype=torch.float32,
            )
        )
