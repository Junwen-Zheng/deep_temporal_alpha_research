from __future__ import annotations

import numpy as np
import torch

from deep_alpha.models.mlp import FeatureMLP
from deep_alpha.training.engine import (
    iterate_minibatches,
)


def test_mlp_output_shape() -> None:
    model = FeatureMLP(
        feature_count=16,
        hidden_dims=[32, 16],
        dropout=0.1,
    )

    features = torch.zeros(
        (10, 16),
        dtype=torch.float32,
    )

    predictions = model(features)

    assert predictions.shape == (10,)
    assert torch.isfinite(predictions).all()


def test_minibatches_cover_all_rows() -> None:
    features = np.arange(
        40,
        dtype=np.float32,
    ).reshape(10, 4)

    targets = np.arange(
        10,
        dtype=np.float32,
    )

    batches = list(
        iterate_minibatches(
            features=features,
            targets=targets,
            batch_size=3,
            shuffle=False,
            seed=42,
        )
    )

    combined_features = np.concatenate(
        [batch[0] for batch in batches],
        axis=0,
    )

    combined_targets = np.concatenate(
        [batch[1] for batch in batches],
        axis=0,
    )

    np.testing.assert_array_equal(
        combined_features,
        features,
    )

    np.testing.assert_array_equal(
        combined_targets,
        targets,
    )


def test_minibatch_shuffle_is_reproducible() -> None:
    features = np.arange(
        40,
        dtype=np.float32,
    ).reshape(10, 4)

    targets = np.arange(
        10,
        dtype=np.float32,
    )

    first = list(
        iterate_minibatches(
            features=features,
            targets=targets,
            batch_size=4,
            shuffle=True,
            seed=17,
        )
    )

    second = list(
        iterate_minibatches(
            features=features,
            targets=targets,
            batch_size=4,
            shuffle=True,
            seed=17,
        )
    )

    first_targets = np.concatenate(
        [batch[1] for batch in first]
    )

    second_targets = np.concatenate(
        [batch[1] for batch in second]
    )

    np.testing.assert_array_equal(
        first_targets,
        second_targets,
    )
