from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_alpha.evaluation.bootstrap import (
    build_portfolio_bootstrap_distribution,
    build_timestamp_rank_ic_series,
    circular_block_bootstrap_means,
    circular_block_bootstrap_nanmeans,
    validate_block_lengths,
)


def test_constant_series_bootstrap_is_exact() -> None:
    samples = (
        circular_block_bootstrap_means(
            values=np.full(
                24,
                3.5,
            ),
            resamples=100,
            block_length=6,
            seed=7,
        )
    )

    np.testing.assert_allclose(
        samples,
        np.full(
            100,
            3.5,
        ),
    )


def test_bootstrap_is_deterministic() -> None:
    values = np.arange(
        30,
        dtype=np.float64,
    )

    first = (
        circular_block_bootstrap_means(
            values=values,
            resamples=50,
            block_length=5,
            seed=11,
        )
    )

    second = (
        circular_block_bootstrap_means(
            values=values,
            resamples=50,
            block_length=5,
            seed=11,
        )
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_paired_bootstrap_preserves_linear_relation() -> None:
    first = np.arange(
        24,
        dtype=np.float64,
    )

    values = np.column_stack(
        [
            first,
            first * 2.0,
        ]
    )

    samples = (
        circular_block_bootstrap_means(
            values=values,
            resamples=100,
            block_length=6,
            seed=17,
        )
    )

    np.testing.assert_allclose(
        samples[:, 1],
        samples[:, 0] * 2.0,
    )


def test_timestamp_rank_ic_is_exact() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
        ]
    )

    rows = []

    for timestamp, reverse in [
        (
            timestamps[0],
            False,
        ),
        (
            timestamps[1],
            True,
        ),
    ]:
        for index, symbol in enumerate(
            [
                "A",
                "B",
                "C",
                "D",
            ]
        ):
            prediction = float(index)

            target = (
                -prediction
                if reverse
                else prediction
            )

            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "prediction": prediction,
                    "target": target,
                }
            )

    result = (
        build_timestamp_rank_ic_series(
            frame=pd.DataFrame(rows),
            expected_symbol_count=4,
        )
    )

    np.testing.assert_allclose(
        result["rank_ic"].to_numpy(),
        np.asarray(
            [
                1.0,
                -1.0,
            ]
        ),
    )


def test_portfolio_bootstrap_constant_returns() -> None:
    rows = []

    for cohort in [
        0,
        1,
    ]:
        for index in range(12):
            rows.append(
                {
                    "cohort": cohort,
                    "timestamp": (
                        pd.Timestamp(
                            "2026-01-01T00:00:00Z"
                        )
                        + pd.Timedelta(
                            hours=index,
                            minutes=(
                                cohort * 5
                            ),
                        )
                    ),
                    "gross_return": 0.0002,
                    "turnover": 1.0,
                }
            )

    (
        distribution,
        observed,
    ) = (
        build_portfolio_bootstrap_distribution(
            periods=pd.DataFrame(rows),
            resamples=100,
            block_length_periods=4,
            seed=23,
            cost_bps=1.0,
            annualization_periods=8760,
        )
    )

    np.testing.assert_allclose(
        distribution[
            "break_even_cost_bps"
        ].to_numpy(),
        np.full(
            100,
            2.0,
        ),
    )

    np.testing.assert_allclose(
        distribution[
            "net_mean_return"
        ].to_numpy(),
        np.full(
            100,
            0.0001,
        ),
    )

    assert np.isclose(
        observed[
            "break_even_cost_bps"
        ],
        2.0,
    )


def test_invalid_block_length_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        validate_block_lengths(
            [
                0,
                12,
            ]
        )

def test_nan_aware_bootstrap_preserves_timeline() -> None:
    values = np.asarray(
        [
            2.0,
            2.0,
            np.nan,
            2.0,
            2.0,
            np.nan,
        ]
        * 4,
        dtype=np.float64,
    )

    samples = (
        circular_block_bootstrap_nanmeans(
            values=values,
            resamples=100,
            block_length=6,
            seed=29,
        )
    )

    np.testing.assert_allclose(
        samples,
        np.full(
            100,
            2.0,
        ),
    )
