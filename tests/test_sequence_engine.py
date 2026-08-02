from __future__ import annotations

import numpy as np
import pandas as pd

from deep_alpha.data.sequences import TemporalPanel
from deep_alpha.training.sequence_engine import (
    extract_cross_sectional_batch,
    iterate_endpoint_batches,
    subsample_endpoints,
)


def sample_temporal_panel() -> TemporalPanel:
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=8,
        freq="5min",
    )

    features = np.arange(
        8 * 2 * 3,
        dtype=np.float32,
    ).reshape(8, 2, 3)

    targets = np.arange(
        8 * 2,
        dtype=np.float32,
    ).reshape(8, 2)

    return TemporalPanel(
        timestamps=timestamps,
        symbols=("A", "B"),
        features=features,
        targets=targets,
        raw_targets=targets.copy(),
    )


def test_vectorized_sequence_batch_ordering() -> None:
    panel = sample_temporal_panel()

    (
        features,
        targets,
        endpoints,
        symbols,
    ) = extract_cross_sectional_batch(
        temporal_panel=panel,
        endpoint_positions=np.asarray(
            [3, 5],
            dtype=np.int64,
        ),
        sequence_length=4,
    )

    assert features.shape == (4, 4, 3)

    np.testing.assert_array_equal(
        endpoints,
        np.asarray([3, 3, 5, 5]),
    )

    np.testing.assert_array_equal(
        symbols,
        np.asarray([0, 1, 0, 1]),
    )

    np.testing.assert_array_equal(
        features[0],
        panel.features[0:4, 0, :],
    )

    np.testing.assert_array_equal(
        features[1],
        panel.features[0:4, 1, :],
    )

    np.testing.assert_array_equal(
        targets,
        panel.targets[[3, 5], :].reshape(-1),
    )


def test_endpoint_subsampling_is_deterministic() -> None:
    endpoints = np.arange(
        10,
        dtype=np.int64,
    )

    selected = subsample_endpoints(
        endpoints,
        stride=3,
    )

    np.testing.assert_array_equal(
        selected,
        np.asarray([0, 3, 6, 9]),
    )


def test_endpoint_batch_shuffle_is_reproducible() -> None:
    endpoints = np.arange(
        20,
        dtype=np.int64,
    )

    first = list(
        iterate_endpoint_batches(
            endpoint_positions=endpoints,
            batch_size=6,
            shuffle=True,
            seed=17,
        )
    )

    second = list(
        iterate_endpoint_batches(
            endpoint_positions=endpoints,
            batch_size=6,
            shuffle=True,
            seed=17,
        )
    )

    np.testing.assert_array_equal(
        np.concatenate(first),
        np.concatenate(second),
    )
