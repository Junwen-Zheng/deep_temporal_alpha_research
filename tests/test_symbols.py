from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_alpha.evaluation.breadth import (
    PredictionMatrices,
)
from deep_alpha.evaluation.symbols import (
    build_symbol_contribution_metrics,
    build_symbol_dependence_model_summary,
    exclude_symbol_from_matrices,
    summarize_symbol_exclusion_portfolios,
)


def sample_matrices() -> PredictionMatrices:
    return PredictionMatrices(
        timestamps=pd.DatetimeIndex(
            pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                ]
            )
        ),
        symbols=(
            "A",
            "B",
            "C",
            "D",
        ),
        predictions=np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
            ]
        ),
        execution_returns=np.asarray(
            [
                [-0.01, 0.00, 0.01, 0.02],
                [0.02, 0.01, 0.00, -0.01],
            ]
        ),
    )


def test_symbol_exclusion_reduces_matrix() -> None:
    result = exclude_symbol_from_matrices(
        matrices=sample_matrices(),
        symbol="B",
    )

    assert result.symbol_count == 3
    assert result.symbols == (
        "A",
        "C",
        "D",
    )
    assert result.predictions.shape == (
        2,
        3,
    )


def test_unknown_symbol_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown symbol",
    ):
        exclude_symbol_from_matrices(
            matrices=sample_matrices(),
            symbol="X",
        )


def test_symbol_contributions_sum_to_portfolio() -> None:
    result = build_symbol_contribution_metrics(
        model="example",
        matrices=sample_matrices(),
        top_count=1,
    )

    assert len(result) == 4

    assert np.isclose(
        result[
            "absolute_contribution_share"
        ].sum(),
        1.0,
    )

    assert np.isclose(
        result[
            "cumulative_gross_log_return_contribution"
        ].sum(),
        result[
            "portfolio_gross_cumulative_log_return"
        ].iloc[0],
    )


def sample_exclusion_cohorts() -> pd.DataFrame:
    rows = []

    for excluded_symbol in [
        "A",
        "B",
    ]:
        gross_mean = (
            0.0002
            if excluded_symbol == "A"
            else 0.0001
        )

        for cost_bps in [
            0.0,
            1.0,
        ]:
            for cohort in [
                0,
                1,
            ]:
                mean_return = (
                    gross_mean
                    - cost_bps
                    / 10_000.0
                )

                rows.append(
                    {
                        "model": "example",
                        "excluded_symbol": (
                            excluded_symbol
                        ),
                        "cost_bps": cost_bps,
                        "cohort": cohort,
                        "period_count": 100,
                        "mean_net_return": (
                            mean_return
                        ),
                        "mean_turnover": 1.0,
                        "cumulative_net_log_return": (
                            mean_return * 100
                        ),
                    }
                )

    return pd.DataFrame(rows)


def test_exclusion_break_even_is_exact() -> None:
    result = (
        summarize_symbol_exclusion_portfolios(
            cohort_metrics=(
                sample_exclusion_cohorts()
            ),
            focus_cost_bps=1.0,
            annualization_periods=8760,
        )
    )

    assert len(result) == 2

    values = (
        result.set_index(
            "excluded_symbol"
        )[
            "pooled_break_even_cost_bps"
        ]
    )

    assert np.isclose(
        values["A"],
        2.0,
    )

    assert np.isclose(
        values["B"],
        1.0,
    )


def test_dependence_summary_identifies_influence() -> None:
    exclusions = pd.DataFrame(
        {
            "model": [
                "example",
                "example",
            ],
            "excluded_symbol": [
                "A",
                "B",
            ],
            "full_mean_rank_ic": [
                0.02,
                0.02,
            ],
            "mean_rank_ic": [
                0.01,
                0.021,
            ],
            "rank_ic_delta": [
                -0.01,
                0.001,
            ],
            "full_break_even_cost_bps": [
                0.5,
                0.5,
            ],
            "pooled_break_even_cost_bps": [
                0.1,
                0.6,
            ],
            "break_even_delta_bps": [
                -0.4,
                0.1,
            ],
            "pooled_annualized_net_return_at_focus_cost": [
                -1.0,
                0.2,
            ],
        }
    )

    contributions = pd.DataFrame(
        {
            "model": [
                "example",
                "example",
            ],
            "symbol": [
                "A",
                "B",
            ],
            "absolute_contribution_share": [
                0.8,
                0.2,
            ],
        }
    )

    result = (
        build_symbol_dependence_model_summary(
            exclusion_summary=exclusions,
            contribution_metrics=(
                contributions
            ),
        )
    )

    assert len(result) == 1

    assert (
        result.loc[
            0,
            "most_influential_prediction_symbol",
        ]
        == "A"
    )

    assert (
        result.loc[
            0,
            "most_influential_economic_symbol",
        ]
        == "A"
    )

    assert (
        result.loc[
            0,
            "largest_absolute_contribution_symbol",
        ]
        == "A"
    )


def test_exclusion_summary_requires_focus_cost() -> None:
    cohorts = sample_exclusion_cohorts()

    cohorts = cohorts.loc[
        cohorts["cost_bps"] == 0.0
    ]

    with pytest.raises(
        ValueError,
        match="focus-cost",
    ):
        summarize_symbol_exclusion_portfolios(
            cohort_metrics=cohorts,
            focus_cost_bps=1.0,
            annualization_periods=8760,
        )
