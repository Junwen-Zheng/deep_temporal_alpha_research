from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def validate_rebalance_cadences(
    cadence_bars: Sequence[int],
    holding_bars: int,
) -> tuple[int, ...]:
    if holding_bars <= 0:
        raise ValueError(
            "holding_bars must be positive"
        )

    cadences = tuple(
        sorted(
            {
                int(value)
                for value in cadence_bars
            }
        )
    )

    if not cadences:
        raise ValueError(
            "At least one cadence is required"
        )

    for cadence in cadences:
        if cadence <= 0:
            raise ValueError(
                "Cadences must be positive"
            )

        if cadence > holding_bars:
            raise ValueError(
                "Cadence cannot exceed "
                "the holding period"
            )

        if holding_bars % cadence != 0:
            raise ValueError(
                f"Cadence {cadence} does not divide "
                f"holding period {holding_bars}"
            )

    return cadences


def build_rebalance_phase_map(
    cadence_bars: Sequence[int],
    holding_bars: int,
    interval_minutes: int,
) -> pd.DataFrame:
    if interval_minutes <= 0:
        raise ValueError(
            "interval_minutes must be positive"
        )

    cadences = validate_rebalance_cadences(
        cadence_bars=cadence_bars,
        holding_bars=holding_bars,
    )

    rows = []

    for cadence in cadences:
        sleeve_count = (
            holding_bars // cadence
        )

        for phase in range(cadence):
            selected_cohorts = tuple(
                range(
                    phase,
                    holding_bars,
                    cadence,
                )
            )

            if len(selected_cohorts) != sleeve_count:
                raise RuntimeError(
                    "Unexpected sleeve count"
                )

            rows.append(
                {
                    "cadence_bars": cadence,
                    "cadence_minutes": (
                        cadence
                        * interval_minutes
                    ),
                    "phase": phase,
                    "phase_offset_minutes": (
                        phase
                        * interval_minutes
                    ),
                    "sleeve_count": sleeve_count,
                    "capital_fraction_per_sleeve": (
                        1.0 / sleeve_count
                    ),
                    "selected_cohorts": ",".join(
                        str(value)
                        for value in selected_cohorts
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        [
            "cadence_bars",
            "phase",
        ]
    ).reset_index(drop=True)


def _validate_cohort_metrics(
    cohort_metrics: pd.DataFrame,
    holding_bars: int,
) -> None:
    required_columns = {
        "model",
        "cohort",
        "period_count",
        "gross_mean_return",
        "mean_turnover",
        "gross_cumulative_log_return",
        "total_turnover",
        "break_even_cost_bps",
    }

    missing_columns = required_columns - set(
        cohort_metrics.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing cohort columns: "
            f"{sorted(missing_columns)}"
        )

    if cohort_metrics.empty:
        raise ValueError(
            "Cohort metrics are empty"
        )

    if cohort_metrics.duplicated(
        [
            "model",
            "cohort",
        ]
    ).any():
        raise ValueError(
            "Duplicate model/cohort rows"
        )

    expected_cohorts = set(
        range(holding_bars)
    )

    for model, group in cohort_metrics.groupby(
        "model",
        sort=True,
    ):
        actual_cohorts = set(
            group["cohort"].astype(int)
        )

        if actual_cohorts != expected_cohorts:
            raise ValueError(
                f"Unexpected cohorts for {model}: "
                f"{sorted(actual_cohorts)}"
            )

    numeric_columns = [
        "period_count",
        "gross_mean_return",
        "mean_turnover",
        "gross_cumulative_log_return",
        "total_turnover",
        "break_even_cost_bps",
    ]

    if not np.isfinite(
        cohort_metrics[
            numeric_columns
        ].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Cohort metrics contain "
            "non-finite values"
        )

    if (
        cohort_metrics["period_count"] <= 0
    ).any():
        raise ValueError(
            "period_count must be positive"
        )

    if (
        cohort_metrics["mean_turnover"] <= 0
    ).any():
        raise ValueError(
            "mean_turnover must be positive"
        )

    if (
        cohort_metrics["total_turnover"] <= 0
    ).any():
        raise ValueError(
            "total_turnover must be positive"
        )


def build_rebalance_phase_metrics(
    cohort_metrics: pd.DataFrame,
    cadence_bars: Sequence[int],
    cost_levels_bps: Sequence[float],
    holding_bars: int,
    interval_minutes: int,
    annualization_periods: int,
) -> pd.DataFrame:
    _validate_cohort_metrics(
        cohort_metrics=cohort_metrics,
        holding_bars=holding_bars,
    )

    if interval_minutes <= 0:
        raise ValueError(
            "interval_minutes must be positive"
        )

    if annualization_periods <= 0:
        raise ValueError(
            "annualization_periods must be positive"
        )

    cadences = validate_rebalance_cadences(
        cadence_bars=cadence_bars,
        holding_bars=holding_bars,
    )

    costs = tuple(
        float(value)
        for value in cost_levels_bps
    )

    if not costs:
        raise ValueError(
            "At least one cost level is required"
        )

    if not np.isfinite(
        np.asarray(costs)
    ).all():
        raise ValueError(
            "Cost levels contain "
            "non-finite values"
        )

    if any(value < 0 for value in costs):
        raise ValueError(
            "Cost levels cannot be negative"
        )

    rows = []

    for model, model_metrics in (
        cohort_metrics.groupby(
            "model",
            sort=True,
        )
    ):
        model_metrics = (
            model_metrics.sort_values(
                "cohort"
            )
            .reset_index(drop=True)
        )

        for cadence in cadences:
            sleeve_count = (
                holding_bars // cadence
            )

            for phase in range(cadence):
                phase_metrics = (
                    model_metrics.loc[
                        (
                            model_metrics[
                                "cohort"
                            ].astype(int)
                            % cadence
                        )
                        == phase
                    ]
                    .sort_values("cohort")
                    .reset_index(drop=True)
                )

                if len(
                    phase_metrics
                ) != sleeve_count:
                    raise ValueError(
                        "Unexpected phase sleeve count"
                    )

                selected_cohorts = tuple(
                    phase_metrics[
                        "cohort"
                    ].astype(int)
                )

                total_periods = int(
                    phase_metrics[
                        "period_count"
                    ].sum()
                )

                total_gross_return = float(
                    phase_metrics[
                        "gross_cumulative_log_return"
                    ].sum()
                )

                total_turnover = float(
                    phase_metrics[
                        "total_turnover"
                    ].sum()
                )

                if total_turnover <= 0:
                    raise ValueError(
                        "Phase total turnover "
                        "must be positive"
                    )

                equal_weighted_gross_mean = float(
                    phase_metrics[
                        "gross_mean_return"
                    ].mean()
                )

                pooled_gross_mean = (
                    total_gross_return
                    / total_periods
                )

                equal_weighted_turnover = float(
                    phase_metrics[
                        "mean_turnover"
                    ].mean()
                )

                pooled_turnover = (
                    total_turnover
                    / total_periods
                )

                break_even_cost = (
                    total_gross_return
                    / total_turnover
                    * 10_000.0
                )

                for cost_bps in costs:
                    sleeve_net_means = (
                        phase_metrics[
                            "gross_mean_return"
                        ]
                        - (
                            phase_metrics[
                                "mean_turnover"
                            ]
                            * cost_bps
                            / 10_000.0
                        )
                    )

                    total_net_return = (
                        total_gross_return
                        - total_turnover
                        * cost_bps
                        / 10_000.0
                    )

                    pooled_net_mean = (
                        total_net_return
                        / total_periods
                    )

                    equal_weighted_net_mean = (
                        float(
                            sleeve_net_means.mean()
                        )
                    )

                    rows.append(
                        {
                            "model": str(model),
                            "cadence_bars": cadence,
                            "cadence_minutes": (
                                cadence
                                * interval_minutes
                            ),
                            "phase": phase,
                            "phase_offset_minutes": (
                                phase
                                * interval_minutes
                            ),
                            "sleeve_count": (
                                sleeve_count
                            ),
                            "capital_fraction_per_sleeve": (
                                1.0
                                / sleeve_count
                            ),
                            "selected_cohorts": (
                                ",".join(
                                    str(value)
                                    for value
                                    in selected_cohorts
                                )
                            ),
                            "cost_bps": cost_bps,
                            "total_periods": (
                                total_periods
                            ),
                            "total_gross_log_return": (
                                total_gross_return
                            ),
                            "total_turnover": (
                                total_turnover
                            ),
                            "break_even_cost_bps": (
                                break_even_cost
                            ),
                            "equal_weighted_gross_mean_return": (
                                equal_weighted_gross_mean
                            ),
                            "pooled_gross_mean_return": (
                                pooled_gross_mean
                            ),
                            "equal_weighted_mean_turnover": (
                                equal_weighted_turnover
                            ),
                            "pooled_mean_turnover": (
                                pooled_turnover
                            ),
                            "equal_weighted_net_mean_return": (
                                equal_weighted_net_mean
                            ),
                            "pooled_net_mean_return": (
                                pooled_net_mean
                            ),
                            "equal_weighted_annualized_net_return": (
                                equal_weighted_net_mean
                                * annualization_periods
                            ),
                            "pooled_annualized_net_return": (
                                pooled_net_mean
                                * annualization_periods
                            ),
                            "positive_sleeve_fraction": (
                                float(
                                    (
                                        sleeve_net_means
                                        > 0
                                    ).mean()
                                )
                            ),
                            "all_sleeves_positive": (
                                bool(
                                    (
                                        sleeve_net_means
                                        > 0
                                    ).all()
                                )
                            ),
                        }
                    )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "model",
                "cadence_bars",
                "phase",
                "cost_bps",
            ]
        )
        .reset_index(drop=True)
    )


def build_rebalance_cadence_summary(
    phase_metrics: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "model",
        "cadence_bars",
        "cadence_minutes",
        "phase",
        "sleeve_count",
        "cost_bps",
        "break_even_cost_bps",
        "equal_weighted_annualized_net_return",
        "pooled_annualized_net_return",
        "pooled_mean_turnover",
        "positive_sleeve_fraction",
    }

    missing_columns = required_columns - set(
        phase_metrics.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing phase metric columns: "
            f"{sorted(missing_columns)}"
        )

    rows = []

    grouping_columns = [
        "model",
        "cadence_bars",
        "cadence_minutes",
        "cost_bps",
    ]

    for keys, group in phase_metrics.groupby(
        grouping_columns,
        sort=True,
    ):
        (
            model,
            cadence_bars,
            cadence_minutes,
            cost_bps,
        ) = keys

        expected_phase_count = int(
            cadence_bars
        )

        if len(group) != expected_phase_count:
            raise ValueError(
                "Unexpected cadence phase count"
            )

        pooled_returns = group[
            "pooled_annualized_net_return"
        ].to_numpy(dtype=np.float64)

        break_even_values = group[
            "break_even_cost_bps"
        ].to_numpy(dtype=np.float64)

        rows.append(
            {
                "model": str(model),
                "cadence_bars": int(
                    cadence_bars
                ),
                "cadence_minutes": int(
                    cadence_minutes
                ),
                "cost_bps": float(
                    cost_bps
                ),
                "phase_count": len(group),
                "sleeve_count": int(
                    group[
                        "sleeve_count"
                    ].iloc[0]
                ),
                "mean_phase_break_even_cost_bps": (
                    float(
                        break_even_values.mean()
                    )
                ),
                "median_phase_break_even_cost_bps": (
                    float(
                        np.median(
                            break_even_values
                        )
                    )
                ),
                "worst_phase_break_even_cost_bps": (
                    float(
                        break_even_values.min()
                    )
                ),
                "best_phase_break_even_cost_bps": (
                    float(
                        break_even_values.max()
                    )
                ),
                "mean_pooled_annualized_net_return": (
                    float(
                        pooled_returns.mean()
                    )
                ),
                "median_pooled_annualized_net_return": (
                    float(
                        np.median(
                            pooled_returns
                        )
                    )
                ),
                "worst_pooled_annualized_net_return": (
                    float(
                        pooled_returns.min()
                    )
                ),
                "best_pooled_annualized_net_return": (
                    float(
                        pooled_returns.max()
                    )
                ),
                "positive_phase_fraction": (
                    float(
                        (
                            pooled_returns > 0
                        ).mean()
                    )
                ),
                "mean_positive_sleeve_fraction": (
                    float(
                        group[
                            "positive_sleeve_fraction"
                        ].mean()
                    )
                ),
                "mean_pooled_turnover": (
                    float(
                        group[
                            "pooled_mean_turnover"
                        ].mean()
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "cost_bps",
                "cadence_bars",
                "model",
            ]
        )
        .reset_index(drop=True)
    )
