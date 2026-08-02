from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_alpha.evaluation.costs import (
    build_break_even_cohort_metrics,
    build_break_even_cost_curve,
    build_break_even_model_summary,
    build_cost_grid,
)


def sample_zero_cost_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["example", "example"],
            "cost_bps": [0.0, 0.0],
            "cohort": [0, 1],
            "period_count": [100, 200],
            "mean_net_return": [
                0.0002,
                0.0001,
            ],
            "mean_turnover": [
                1.0,
                0.5,
            ],
            "cumulative_net_log_return": [
                0.02,
                0.02,
            ],
        }
    )


def test_cohort_break_even_cost_is_exact() -> None:
    result = (
        build_break_even_cohort_metrics(
            cohort_metrics=(
                sample_zero_cost_metrics()
            ),
            annualization_periods=8760,
        )
    )

    np.testing.assert_allclose(
        result[
            "break_even_cost_bps"
        ].to_numpy(),
        np.asarray([2.0, 2.0]),
    )


def test_pooled_break_even_cost_is_exact() -> None:
    cohorts = (
        build_break_even_cohort_metrics(
            cohort_metrics=(
                sample_zero_cost_metrics()
            ),
            annualization_periods=8760,
        )
    )

    summary = (
        build_break_even_model_summary(
            cohort_metrics=cohorts,
            annualization_periods=8760,
            robustness_costs_bps=[
                0.0,
                2.0,
            ],
        )
    )

    assert np.isclose(
        summary.loc[
            0,
            "pooled_break_even_cost_bps",
        ],
        2.0,
    )

    assert np.isclose(
        summary.loc[
            0,
            (
                "equal_weighted_"
                "break_even_cost_bps"
            ),
        ],
        2.0,
    )


def test_cost_curve_reaches_zero_at_break_even() -> None:
    cohorts = (
        build_break_even_cohort_metrics(
            cohort_metrics=(
                sample_zero_cost_metrics()
            ),
            annualization_periods=8760,
        )
    )

    curve = build_break_even_cost_curve(
        cohort_metrics=cohorts,
        cost_grid_bps=np.asarray(
            [0.0, 2.0, 3.0]
        ),
        annualization_periods=8760,
    )

    at_two = curve.loc[
        curve["cost_bps"] == 2.0
    ].iloc[0]

    at_three = curve.loc[
        curve["cost_bps"] == 3.0
    ].iloc[0]

    assert np.isclose(
        at_two[
            "pooled_mean_net_return"
        ],
        0.0,
        atol=1e-15,
    )

    assert (
        at_three[
            "pooled_mean_net_return"
        ]
        < 0
    )


def test_cost_grid_contains_exact_endpoints() -> None:
    grid = build_cost_grid(
        start_bps=0.0,
        end_bps=1.0,
        step_bps=0.1,
    )

    assert len(grid) == 11
    assert grid[0] == 0.0
    assert grid[-1] == 1.0

    np.testing.assert_allclose(
        np.diff(grid),
        np.full(10, 0.1),
    )


def test_zero_turnover_is_rejected() -> None:
    metrics = sample_zero_cost_metrics()
    metrics.loc[0, "mean_turnover"] = 0.0

    with pytest.raises(
        ValueError,
        match="mean_turnover",
    ):
        build_break_even_cohort_metrics(
            cohort_metrics=metrics,
            annualization_periods=8760,
        )
