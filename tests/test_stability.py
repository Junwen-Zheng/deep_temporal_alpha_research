from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_alpha.evaluation.stability import (
    add_calendar_month,
    build_temporal_stability_summary,
    summarize_grouped_predictions,
    summarize_temporal_portfolio_groups,
)


def sample_predictions() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-02-01T00:00:00Z",
            "2026-02-02T00:00:00Z",
        ]
    )

    rows = []

    for timestamp in timestamps:
        for symbol_index, symbol in enumerate(
            [
                "A",
                "B",
                "C",
                "D",
            ]
        ):
            value = float(
                symbol_index
            )

            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "raw_target": value,
                    "target": value,
                    "prediction": value,
                    "fold": 1,
                }
            )

    return pd.DataFrame(rows)


def sample_periods() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [
                1,
                1,
                1,
                1,
            ],
            "month": [
                "2026-01",
                "2026-01",
                "2026-01",
                "2026-01",
            ],
            "cohort": [
                0,
                0,
                1,
                1,
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T00:05:00Z",
                    "2026-01-01T01:05:00Z",
                ]
            ),
            "gross_return": [
                0.0002,
                0.0002,
                0.0002,
                0.0002,
            ],
            "turnover": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        }
    )


def test_calendar_month_assignment() -> None:
    result = add_calendar_month(
        sample_predictions()
    )

    assert set(result["month"]) == {
        "2026-01",
        "2026-02",
    }


def test_grouped_prediction_rank_ic() -> None:
    predictions = add_calendar_month(
        sample_predictions()
    )

    result = (
        summarize_grouped_predictions(
            predictions,
            group_columns=[
                "fold",
                "month",
            ],
        )
    )

    assert len(result) == 2

    np.testing.assert_allclose(
        result[
            "mean_rank_ic"
        ].to_numpy(),
        np.ones(2),
    )


def test_temporal_portfolio_break_even_is_exact() -> None:
    result = (
        summarize_temporal_portfolio_groups(
            periods=sample_periods(),
            group_columns=[
                "fold",
                "month",
            ],
            cost_levels_bps=[
                0.0,
                1.0,
            ],
            annualization_periods=8760,
        )
    )

    assert len(result) == 2

    np.testing.assert_allclose(
        result[
            "break_even_cost_bps"
        ].to_numpy(),
        np.asarray(
            [
                2.0,
                2.0,
            ]
        ),
    )


def test_one_basis_point_reduces_mean_return() -> None:
    result = (
        summarize_temporal_portfolio_groups(
            periods=sample_periods(),
            group_columns=[
                "fold",
            ],
            cost_levels_bps=[
                0.0,
                1.0,
            ],
            annualization_periods=8760,
        )
        .sort_values("cost_bps")
    )

    np.testing.assert_allclose(
        result[
            "pooled_net_mean_return"
        ].to_numpy(),
        np.asarray(
            [
                0.0002,
                0.0001,
            ]
        ),
    )


def test_stability_summary_uses_focus_cost() -> None:
    fold_prediction = pd.DataFrame(
        {
            "model": [
                "example",
                "example",
                "example",
            ],
            "fold": [
                1,
                2,
                3,
            ],
            "mean_rank_ic": [
                0.01,
                0.02,
                -0.01,
            ],
        }
    )

    month_prediction = pd.DataFrame(
        {
            "model": [
                "example",
                "example",
            ],
            "fold": [
                1,
                2,
            ],
            "month": [
                "2026-01",
                "2026-02",
            ],
            "mean_rank_ic": [
                0.01,
                -0.01,
            ],
        }
    )

    portfolio_rows = []

    for fold in [
        1,
        2,
        3,
    ]:
        for cost in [
            0.0,
            1.0,
        ]:
            portfolio_rows.append(
                {
                    "model": "example",
                    "fold": fold,
                    "cost_bps": cost,
                    "break_even_cost_bps": (
                        0.5
                    ),
                    "pooled_annualized_net_return": (
                        1.0
                        if cost == 0.0
                        else -1.0
                    ),
                }
            )

    fold_portfolio = pd.DataFrame(
        portfolio_rows
    )

    month_portfolio = pd.DataFrame(
        {
            "model": [
                "example",
                "example",
                "example",
                "example",
            ],
            "fold": [
                1,
                1,
                2,
                2,
            ],
            "month": [
                "2026-01",
                "2026-01",
                "2026-02",
                "2026-02",
            ],
            "cost_bps": [
                0.0,
                1.0,
                0.0,
                1.0,
            ],
            "break_even_cost_bps": [
                0.5,
                0.5,
                -0.2,
                -0.2,
            ],
            "pooled_annualized_net_return": [
                1.0,
                -1.0,
                -0.5,
                -2.0,
            ],
        }
    )

    result = (
        build_temporal_stability_summary(
            fold_prediction_metrics=(
                fold_prediction
            ),
            month_prediction_metrics=(
                month_prediction
            ),
            fold_portfolio_metrics=(
                fold_portfolio
            ),
            month_portfolio_metrics=(
                month_portfolio
            ),
            focus_cost_bps=1.0,
        )
    )

    assert len(result) == 1

    assert np.isclose(
        result.loc[
            0,
            "positive_fold_fraction",
        ],
        2.0 / 3.0,
    )

    assert np.isclose(
        result.loc[
            0,
            (
                "positive_month_fraction_"
                "at_focus_cost"
            ),
        ],
        0.0,
    )


def test_missing_group_column_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Missing prediction grouping",
    ):
        summarize_grouped_predictions(
            sample_predictions(),
            group_columns=[
                "month",
            ],
        )
