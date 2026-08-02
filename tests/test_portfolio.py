from __future__ import annotations

import numpy as np
import pandas as pd

from deep_alpha.evaluation.portfolio import (
    apply_transaction_cost,
    assign_rebalance_cohort,
    build_portfolio_periods,
    build_rank_weights,
)


def sample_predictions() -> pd.DataFrame:
    rows = []

    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=3,
        freq="60min",
    )

    symbols = ["A", "B", "C", "D"]

    predictions = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0],
    ]

    execution_returns = [
        [-0.01, 0.0, 0.01, 0.02],
        [-0.02, 0.0, 0.01, 0.03],
        [0.02, 0.01, 0.0, -0.01],
    ]

    for timestamp, scores, returns in zip(
        timestamps,
        predictions,
        execution_returns,
        strict=True,
    ):
        for symbol, score, return_value in zip(
            symbols,
            scores,
            returns,
            strict=True,
        ):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "prediction": score,
                    "target": return_value,
                    "execution_return": (
                        return_value
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_rank_weights_are_neutral_and_unit_gross() -> None:
    weighted = build_rank_weights(
        frame=sample_predictions(),
        top_count=1,
    )

    exposure = weighted.groupby(
        "timestamp"
    )["weight"].agg(
        net="sum",
        gross=lambda values: np.abs(
            values
        ).sum(),
    )

    np.testing.assert_allclose(
        exposure["net"].to_numpy(),
        np.zeros(3),
    )

    np.testing.assert_allclose(
        exposure["gross"].to_numpy(),
        np.ones(3),
    )


def test_hourly_period_turnover_is_correct() -> None:
    periods = build_portfolio_periods(
        frame=sample_predictions(),
        top_count=1,
        interval_minutes=5,
        holding_bars=12,
    )

    np.testing.assert_allclose(
        periods["turnover"].to_numpy(),
        np.asarray([1.0, 0.0, 2.0]),
    )


def test_transaction_cost_uses_traded_notional() -> None:
    periods = build_portfolio_periods(
        frame=sample_predictions(),
        top_count=1,
        interval_minutes=5,
        holding_bars=12,
    )

    adjusted = apply_transaction_cost(
        periods=periods,
        cost_bps=10.0,
    )

    expected_cost = (
        periods["turnover"].to_numpy()
        * 0.001
    )

    np.testing.assert_allclose(
        adjusted["cost"].to_numpy(),
        expected_cost,
    )

    np.testing.assert_allclose(
        adjusted["net_return"].to_numpy(),
        (
            periods["gross_return"].to_numpy()
            - expected_cost
        ),
    )


def test_rebalance_cohorts_cover_twelve_phases() -> None:
    timestamps = pd.Series(
        pd.date_range(
            "2026-01-01T00:00:00Z",
            periods=13,
            freq="5min",
        )
    )

    cohorts = assign_rebalance_cohort(
        timestamps=timestamps,
        interval_minutes=5,
        holding_bars=12,
    )

    np.testing.assert_array_equal(
        cohorts[:12],
        np.arange(12),
    )

    assert cohorts[12] == cohorts[0]
