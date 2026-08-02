from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_alpha.evaluation.rebalance import (
    build_rebalance_cadence_summary,
    build_rebalance_phase_map,
    build_rebalance_phase_metrics,
    validate_rebalance_cadences,
)


def sample_cohort_metrics() -> pd.DataFrame:
    gross_means = np.asarray(
        [
            0.0002,
            0.0001,
            -0.0001,
            0.0003,
        ]
    )

    period_count = 100
    turnover = 1.0

    return pd.DataFrame(
        {
            "model": ["example"] * 4,
            "cohort": [0, 1, 2, 3],
            "period_count": [
                period_count
            ] * 4,
            "gross_mean_return": (
                gross_means
            ),
            "mean_turnover": [
                turnover
            ] * 4,
            "gross_cumulative_log_return": (
                gross_means
                * period_count
            ),
            "total_turnover": [
                turnover
                * period_count
            ] * 4,
            "break_even_cost_bps": (
                gross_means
                / turnover
                * 10_000.0
            ),
        }
    )


def test_phase_map_has_expected_counts() -> None:
    phase_map = build_rebalance_phase_map(
        cadence_bars=[
            1,
            2,
            4,
        ],
        holding_bars=4,
        interval_minutes=5,
    )

    assert len(phase_map) == 7

    assert (
        phase_map.groupby(
            "cadence_bars"
        )["phase"]
        .nunique()
        .to_dict()
        == {
            1: 1,
            2: 2,
            4: 4,
        }
    )


def test_full_refresh_reproduces_pooled_break_even() -> None:
    result = build_rebalance_phase_metrics(
        cohort_metrics=(
            sample_cohort_metrics()
        ),
        cadence_bars=[1],
        cost_levels_bps=[0.0],
        holding_bars=4,
        interval_minutes=5,
        annualization_periods=8760,
    )

    assert len(result) == 1

    expected = (
        np.asarray(
            [
                0.0002,
                0.0001,
                -0.0001,
                0.0003,
            ]
        ).mean()
        * 10_000.0
    )

    assert np.isclose(
        result.loc[
            0,
            "break_even_cost_bps",
        ],
        expected,
    )

    assert result.loc[
        0,
        "sleeve_count",
    ] == 4


def test_hourly_refresh_matches_each_cohort() -> None:
    result = build_rebalance_phase_metrics(
        cohort_metrics=(
            sample_cohort_metrics()
        ),
        cadence_bars=[4],
        cost_levels_bps=[0.0],
        holding_bars=4,
        interval_minutes=5,
        annualization_periods=8760,
    )

    assert len(result) == 4

    np.testing.assert_allclose(
        result.sort_values("phase")[
            "break_even_cost_bps"
        ].to_numpy(),
        sample_cohort_metrics()[
            "break_even_cost_bps"
        ].to_numpy(),
    )


def test_transaction_cost_reduces_return_linearly() -> None:
    result = build_rebalance_phase_metrics(
        cohort_metrics=(
            sample_cohort_metrics()
        ),
        cadence_bars=[1],
        cost_levels_bps=[
            0.0,
            1.0,
            2.0,
        ],
        holding_bars=4,
        interval_minutes=5,
        annualization_periods=8760,
    ).sort_values("cost_bps")

    differences = np.diff(
        result[
            "pooled_net_mean_return"
        ].to_numpy()
    )

    np.testing.assert_allclose(
        differences,
        np.asarray(
            [
                -0.0001,
                -0.0001,
            ]
        ),
    )


def test_cadence_summary_preserves_phase_count() -> None:
    phases = build_rebalance_phase_metrics(
        cohort_metrics=(
            sample_cohort_metrics()
        ),
        cadence_bars=[
            1,
            2,
            4,
        ],
        cost_levels_bps=[0.0],
        holding_bars=4,
        interval_minutes=5,
        annualization_periods=8760,
    )

    summary = (
        build_rebalance_cadence_summary(
            phases
        )
    )

    assert (
        summary.set_index(
            "cadence_bars"
        )["phase_count"].to_dict()
        == {
            1: 1,
            2: 2,
            4: 4,
        }
    )


def test_nondividing_cadence_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="does not divide",
    ):
        validate_rebalance_cadences(
            cadence_bars=[3],
            holding_bars=4,
        )
