from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_alpha.evaluation.ablations import (
    add_full_variant_deltas,
    build_ablation_portfolio_summary,
    build_leave_one_family_out_sets,
    validate_feature_families,
)


def test_feature_families_form_partition() -> None:
    validate_feature_families(
        feature_columns=[
            "a",
            "b",
            "c",
            "d",
        ],
        feature_families={
            "first": [
                "a",
                "b",
            ],
            "second": [
                "c",
                "d",
            ],
        },
    )


def test_duplicate_family_assignment_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="multiple families",
    ):
        validate_feature_families(
            feature_columns=[
                "a",
                "b",
            ],
            feature_families={
                "first": [
                    "a",
                ],
                "second": [
                    "a",
                    "b",
                ],
            },
        )


def test_leave_one_family_out_sets() -> None:
    result = (
        build_leave_one_family_out_sets(
            feature_columns=[
                "a",
                "b",
                "c",
                "d",
            ],
            feature_families={
                "first": [
                    "a",
                    "b",
                ],
                "second": [
                    "c",
                    "d",
                ],
            },
        )
    )

    assert result == {
        "full": [
            "a",
            "b",
            "c",
            "d",
        ],
        "without_first": [
            "c",
            "d",
        ],
        "without_second": [
            "a",
            "b",
        ],
    }


def sample_cohort_metrics() -> pd.DataFrame:
    rows = []

    for cost_bps in [
        0.0,
        1.0,
    ]:
        for cohort in [
            0,
            1,
        ]:
            gross_mean = 0.0002

            net_mean = (
                gross_mean
                - cost_bps
                / 10_000.0
            )

            rows.append(
                {
                    "model": "ridge",
                    "variant": "full",
                    "cost_bps": cost_bps,
                    "cohort": cohort,
                    "period_count": 100,
                    "mean_net_return": (
                        net_mean
                    ),
                    "mean_turnover": 1.0,
                    "annualized_sharpe": 1.0,
                    "cumulative_net_log_return": (
                        net_mean * 100
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_ablation_break_even_is_exact() -> None:
    result = (
        build_ablation_portfolio_summary(
            cohort_metrics=(
                sample_cohort_metrics()
            ),
            focus_cost_bps=1.0,
            annualization_periods=8760,
        )
    )

    assert len(result) == 1

    assert np.isclose(
        result.loc[
            0,
            "pooled_break_even_cost_bps",
        ],
        2.0,
    )

    assert np.isclose(
        result.loc[
            0,
            (
                "pooled_annualized_net_return_"
                "at_focus_cost"
            ),
        ],
        0.876,
    )


def test_full_variant_deltas_are_exact() -> None:
    frame = pd.DataFrame(
        {
            "model": [
                "ridge",
                "ridge",
            ],
            "variant": [
                "full",
                "without_returns",
            ],
            "mean_rank_ic": [
                0.02,
                0.01,
            ],
        }
    )

    result = add_full_variant_deltas(
        frame=frame,
        metric_columns=[
            "mean_rank_ic",
        ],
    )

    ablated = result.loc[
        result["variant"]
        == "without_returns"
    ].iloc[0]

    assert np.isclose(
        ablated[
            "mean_rank_ic_delta_vs_full"
        ],
        -0.01,
    )


def test_portfolio_summary_requires_focus_cost() -> None:
    frame = sample_cohort_metrics()

    frame = frame.loc[
        frame["cost_bps"] == 0.0
    ]

    with pytest.raises(
        ValueError,
        match="focus-cost",
    ):
        build_ablation_portfolio_summary(
            cohort_metrics=frame,
            focus_cost_bps=1.0,
            annualization_periods=8760,
        )
