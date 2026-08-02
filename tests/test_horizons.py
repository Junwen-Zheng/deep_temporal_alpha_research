from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_alpha.evaluation.horizons import (
    annualization_periods_for_horizon,
    build_cross_sectional_horizon_target,
    build_horizon_break_even,
    build_symbol_horizon_reference,
    validate_horizon_bars,
)


def sample_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=6,
                freq="5min",
            ),
            "open": [
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
            ],
            "close": [
                10.5,
                11.5,
                12.5,
                13.5,
                14.5,
                15.5,
            ],
        }
    )


def test_horizon_reference_builds_forward_label() -> None:
    bars = sample_bars()

    result = (
        build_symbol_horizon_reference(
            bars=bars,
            horizon_bars=2,
            interval_minutes=5,
        )
    )

    expected = np.log(
        bars.loc[2, "close"]
        / bars.loc[0, "close"]
    )

    assert np.isclose(
        result.loc[
            0,
            "raw_target",
        ],
        expected,
    )

    assert (
        result.loc[
            0,
            "label_end_timestamp",
        ]
        == bars.loc[
            2,
            "timestamp",
        ]
    )


def test_horizon_execution_uses_next_open() -> None:
    bars = sample_bars()

    result = (
        build_symbol_horizon_reference(
            bars=bars,
            horizon_bars=2,
            interval_minutes=5,
            entry_offset_bars=1,
        )
    )

    expected = np.log(
        bars.loc[2, "close"]
        / bars.loc[1, "open"]
    )

    assert np.isclose(
        result.loc[
            0,
            "execution_return",
        ],
        expected,
    )


def test_horizon_target_has_zero_cross_sectional_mean() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
            "2026-01-01T00:05:00Z",
        ]
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [
                "A",
                "B",
                "A",
                "B",
            ],
            "horizon_bars": [
                3,
                3,
                3,
                3,
            ],
            "raw_target": [
                0.01,
                0.03,
                -0.02,
                0.04,
            ],
        }
    )

    result = (
        build_cross_sectional_horizon_target(
            frame=frame,
            expected_symbol_count=2,
        )
    )

    means = result.groupby(
        "timestamp"
    )["target"].mean()

    np.testing.assert_allclose(
        means.to_numpy(),
        np.zeros(2),
        atol=1e-15,
    )


def test_horizon_annualization_is_exact() -> None:
    assert (
        annualization_periods_for_horizon(
            interval_minutes=5,
            horizon_bars=3,
        )
        == 35_040
    )

    assert (
        annualization_periods_for_horizon(
            interval_minutes=5,
            horizon_bars=12,
        )
        == 8_760
    )

    assert (
        annualization_periods_for_horizon(
            interval_minutes=5,
            horizon_bars=24,
        )
        == 4_380
    )


def test_horizon_break_even_is_exact() -> None:
    cohort_metrics = pd.DataFrame(
        {
            "model": [
                "example",
                "example",
            ],
            "horizon_bars": [
                3,
                3,
            ],
            "horizon_minutes": [
                15,
                15,
            ],
            "cost_bps": [
                0.0,
                0.0,
            ],
            "cohort": [
                0,
                1,
            ],
            "period_count": [
                100,
                200,
            ],
            "mean_net_return": [
                0.0002,
                0.0001,
            ],
            "mean_turnover": [
                1.0,
                0.5,
            ],
            "annualized_net_return": [
                7.008,
                3.504,
            ],
            "annualized_sharpe": [
                1.0,
                0.5,
            ],
            "cumulative_net_log_return": [
                0.02,
                0.02,
            ],
        }
    )

    result = build_horizon_break_even(
        cohort_metrics
    )

    assert len(result) == 1

    assert np.isclose(
        result.loc[
            0,
            "pooled_break_even_cost_bps",
        ],
        2.0,
    )


def test_invalid_horizon_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        validate_horizon_bars(
            [
                0,
                3,
            ]
        )
