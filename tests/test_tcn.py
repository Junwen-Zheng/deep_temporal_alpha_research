from __future__ import annotations

import torch

from deep_alpha.models.tcn import (
    CausalConv1d,
    TemporalConvNet,
)


def test_tcn_output_shape() -> None:
    model = TemporalConvNet(
        feature_count=16,
        channel_count=16,
        kernel_size=2,
        dilations=[1, 2, 4, 8, 16, 32, 64],
        dropout=0.1,
    )

    sequences = torch.zeros(
        (10, 128, 16),
        dtype=torch.float32,
    )

    predictions = model(sequences)

    assert predictions.shape == (10,)
    assert torch.isfinite(predictions).all()


def test_tcn_receptive_field_matches_sequence() -> None:
    model = TemporalConvNet(
        feature_count=16,
        channel_count=16,
        kernel_size=2,
        dilations=[1, 2, 4, 8, 16, 32, 64],
        dropout=0.0,
    )

    assert model.receptive_field == 128


def test_causal_convolution_does_not_use_future_values() -> None:
    torch.manual_seed(42)

    convolution = CausalConv1d(
        input_channels=2,
        output_channels=4,
        kernel_size=3,
        dilation=2,
    )

    original = torch.randn(
        (1, 2, 12),
        dtype=torch.float32,
    )

    modified = original.clone()
    modified[:, :, 7:] += 100.0

    original_output = convolution(original)
    modified_output = convolution(modified)

    torch.testing.assert_close(
        original_output[:, :, :7],
        modified_output[:, :, :7],
        rtol=0.0,
        atol=0.0,
    )
