from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_alpha.evaluation.nulls import (
    build_ranked_prediction_panel,
    cross_sectional_rank_ic,
    empirical_two_sided_p_value,
    permute_rows,
    rank_rows,
    shifted_cross_sectional_rank_ic,
)


def test_rank_ic_recovers_perfect_ordering() -> None:
    predictions = rank_rows(
        np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
    )

    targets = rank_rows(
        np.asarray(
            [
                [10.0, 20.0, 30.0, 40.0],
                [40.0, 30.0, 20.0, 10.0],
            ]
        )
    )

    rank_ic = cross_sectional_rank_ic(
        predictions,
        targets,
    )

    np.testing.assert_allclose(
        rank_ic,
        np.ones(2),
    )


def test_rank_ic_recovers_inverse_ordering() -> None:
    predictions = rank_rows(
        np.asarray(
            [[1.0, 2.0, 3.0, 4.0]]
        )
    )

    targets = rank_rows(
        np.asarray(
            [[4.0, 3.0, 2.0, 1.0]]
        )
    )

    rank_ic = cross_sectional_rank_ic(
        predictions,
        targets,
    )

    np.testing.assert_allclose(
        rank_ic,
        np.asarray([-1.0]),
    )


def test_row_permutation_is_reproducible() -> None:
    values = np.arange(
        20,
        dtype=np.float64,
    ).reshape(4, 5)

    first = permute_rows(
        values,
        generator=np.random.default_rng(17),
    )

    second = permute_rows(
        values,
        generator=np.random.default_rng(17),
    )

    np.testing.assert_array_equal(
        first,
        second,
    )

    for row_index in range(len(values)):
        np.testing.assert_array_equal(
            np.sort(first[row_index]),
            values[row_index],
        )


def test_time_shift_uses_expected_alignment() -> None:
    predictions = rank_rows(
        np.asarray(
            [
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
            ]
        )
    )

    targets = predictions.copy()

    zero_shift = (
        shifted_cross_sectional_rank_ic(
            predictions,
            targets,
            shift_bars=0,
        )
    )

    positive_shift = (
        shifted_cross_sectional_rank_ic(
            predictions,
            targets,
            shift_bars=1,
        )
    )

    assert len(zero_shift) == 3
    assert len(positive_shift) == 2

    np.testing.assert_allclose(
        zero_shift,
        np.ones(3),
    )


def test_empirical_p_value_includes_correction() -> None:
    null_values = np.asarray(
        [-0.2, -0.1, 0.0, 0.1, 0.2]
    )

    result = empirical_two_sided_p_value(
        observed_value=0.15,
        null_values=null_values,
    )

    assert np.isclose(
        result,
        3.0 / 6.0,
    )


def test_incomplete_cross_section_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:05:00Z",
                ]
            ),
            "symbol": ["A", "B", "A"],
            "target": [0.1, -0.1, 0.2],
            "prediction": [1.0, -1.0, 0.5],
        }
    )

    with pytest.raises(
        ValueError,
        match="Incomplete prediction",
    ):
        build_ranked_prediction_panel(
            frame=frame,
            expected_symbol_count=2,
        )


def test_zero_variance_cross_section_returns_nan() -> None:
    predictions = rank_rows(
        np.asarray(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
    )

    targets = rank_rows(
        np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
    )

    rank_ic = cross_sectional_rank_ic(
        predictions,
        targets,
    )

    assert np.isnan(rank_ic[0])
    assert np.isclose(rank_ic[1], 1.0)
