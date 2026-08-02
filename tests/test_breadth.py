from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_alpha.evaluation.breadth import (
    build_breadth_break_even,
    build_breadth_periods,
    build_breadth_summary,
    build_prediction_matrices,
    build_rank_weight_matrix,
)
from deep_alpha.evaluation.portfolio import (
    summarize_portfolio_cohorts,
)


def sample_prediction_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T03:00:00Z",
        ]
    )

    symbols = [
        "A",
        "B",
        "C",
        "D",
    ]

    scores = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0],
    ]

    returns = [
        [-0.01, 0.00, 0.01, 0.02],
        [-0.02, 0.00, 0.01, 0.03],
        [0.02, 0.01, 0.00, -0.01],
    ]

    rows = []

    for timestamp, predictions, outcomes in zip(
        timestamps,
        scores,
        returns,
        strict=True,
    ):
        for symbol, prediction, outcome in zip(
            symbols,
            predictions,
            outcomes,
            strict=True,
        ):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "prediction": prediction,
                    "execution_return": (
                        outcome
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_prediction_matrices_have_expected_shape() -> None:
    matrices = build_prediction_matrices(
        frame=sample_prediction_frame(),
        expected_symbol_count=4,
    )

    assert matrices.timestamp_count == 3
    assert matrices.symbol_count == 4
    assert matrices.symbols == (
        "A",
        "B",
        "C",
        "D",
    )
    assert matrices.predictions.shape == (
        3,
        4,
    )


def test_rank_weights_are_neutral_and_unit_gross() -> None:
    predictions = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )

    weights = build_rank_weight_matrix(
        predictions=predictions,
        top_count=1,
    )

    np.testing.assert_allclose(
        weights.sum(axis=1),
        np.zeros(2),
    )

    np.testing.assert_allclose(
        np.abs(weights).sum(axis=1),
        np.ones(2),
    )

    np.testing.assert_allclose(
        weights[0],
        np.asarray(
            [
                -0.5,
                0.0,
                0.0,
                0.5,
            ]
        ),
    )


def test_ties_follow_stable_symbol_order() -> None:
    predictions = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    weights = build_rank_weight_matrix(
        predictions=predictions,
        top_count=1,
    )

    np.testing.assert_allclose(
        weights[0],
        np.asarray(
            [
                -0.5,
                0.0,
                0.0,
                0.5,
            ]
        ),
    )


def test_discontinuity_resets_turnover_from_cash() -> None:
    matrices = build_prediction_matrices(
        frame=sample_prediction_frame(),
        expected_symbol_count=4,
    )

    periods = build_breadth_periods(
        matrices=matrices,
        top_count=1,
        interval_minutes=5,
        holding_bars=12,
    ).sort_values("timestamp")

    np.testing.assert_allclose(
        periods["turnover"].to_numpy(),
        np.asarray(
            [
                1.0,
                0.0,
                1.0,
            ]
        ),
    )


def test_invalid_breadth_is_rejected() -> None:
    predictions = np.zeros(
        (3, 4),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="too large",
    ):
        build_rank_weight_matrix(
            predictions=predictions,
            top_count=3,
        )


def test_breadth_summary_and_break_even() -> None:
    matrices = build_prediction_matrices(
        frame=sample_prediction_frame(),
        expected_symbol_count=4,
    )

    periods = build_breadth_periods(
        matrices=matrices,
        top_count=1,
        interval_minutes=5,
        holding_bars=12,
    )

    cohort_metrics = (
        summarize_portfolio_cohorts(
            periods=periods,
            cost_levels_bps=[
                0.0,
                1.0,
            ],
            annualization_periods=8760,
        )
    )

    cohort_metrics.insert(
        0,
        "top_count",
        1,
    )

    cohort_metrics.insert(
        0,
        "model",
        "example",
    )

    summary = build_breadth_summary(
        cohort_metrics
    )

    break_even = build_breadth_break_even(
        cohort_metrics
    )

    assert len(summary) == 2
    assert len(break_even) == 1

    assert (
        break_even.loc[
            0,
            "top_count",
        ]
        == 1
    )

    assert np.isfinite(
        break_even.loc[
            0,
            "pooled_break_even_cost_bps",
        ]
    )
