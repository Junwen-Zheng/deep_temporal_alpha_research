from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_cost_grid(
    start_bps: float,
    end_bps: float,
    step_bps: float,
) -> np.ndarray:
    if start_bps < 0:
        raise ValueError(
            "start_bps cannot be negative"
        )

    if end_bps < start_bps:
        raise ValueError(
            "end_bps cannot precede start_bps"
        )

    if step_bps <= 0:
        raise ValueError(
            "step_bps must be positive"
        )

    interval_count_float = (
        end_bps - start_bps
    ) / step_bps

    interval_count = round(
        interval_count_float
    )

    if not np.isclose(
        interval_count_float,
        interval_count,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "The cost range is not divisible "
            "by the requested step"
        )

    return np.linspace(
        start_bps,
        end_bps,
        interval_count + 1,
        dtype=np.float64,
    )


def build_break_even_cohort_metrics(
    cohort_metrics: pd.DataFrame,
    annualization_periods: int,
) -> pd.DataFrame:
    required_columns = {
        "model",
        "cost_bps",
        "cohort",
        "period_count",
        "mean_net_return",
        "mean_turnover",
        "cumulative_net_log_return",
    }

    missing_columns = required_columns - set(
        cohort_metrics.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing cohort metric columns: "
            f"{sorted(missing_columns)}"
        )

    if annualization_periods <= 0:
        raise ValueError(
            "annualization_periods must be positive"
        )

    zero_cost_mask = np.isclose(
        cohort_metrics["cost_bps"].to_numpy(
            dtype=np.float64
        ),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    zero_cost = cohort_metrics.loc[
        zero_cost_mask
    ].copy()

    if zero_cost.empty:
        raise ValueError(
            "No zero-cost cohort metrics were found"
        )

    if zero_cost.duplicated(
        ["model", "cohort"]
    ).any():
        raise ValueError(
            "Duplicate zero-cost model/cohort rows"
        )

    numeric_columns = [
        "period_count",
        "mean_net_return",
        "mean_turnover",
        "cumulative_net_log_return",
    ]

    numeric_values = zero_cost[
        numeric_columns
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Zero-cost metrics contain "
            "non-finite values"
        )

    period_counts = zero_cost[
        "period_count"
    ].to_numpy(dtype=np.float64)

    if np.any(period_counts <= 0):
        raise ValueError(
            "period_count must be positive"
        )

    if not np.allclose(
        period_counts,
        np.round(period_counts),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "period_count must contain integers"
        )

    if (
        zero_cost["mean_turnover"] <= 0
    ).any():
        raise ValueError(
            "mean_turnover must be positive"
        )

    expected_cumulative = (
        zero_cost["mean_net_return"]
        * zero_cost["period_count"]
    )

    cumulative_error = np.abs(
        expected_cumulative
        - zero_cost[
            "cumulative_net_log_return"
        ]
    ).max()

    if cumulative_error > 1e-8:
        raise ValueError(
            "Zero-cost cumulative returns "
            "do not match mean return times "
            "period count"
        )

    result = zero_cost[
        [
            "model",
            "cohort",
            "period_count",
            "mean_net_return",
            "mean_turnover",
            "cumulative_net_log_return",
        ]
    ].copy()

    result = result.rename(
        columns={
            "mean_net_return": (
                "gross_mean_return"
            ),
            "cumulative_net_log_return": (
                "gross_cumulative_log_return"
            ),
        }
    )

    result["period_count"] = result[
        "period_count"
    ].astype(int)

    result["gross_annualized_return"] = (
        result["gross_mean_return"]
        * annualization_periods
    )

    result["total_turnover"] = (
        result["mean_turnover"]
        * result["period_count"]
    )

    result["break_even_cost_bps"] = (
        result[
            "gross_cumulative_log_return"
        ]
        / result["total_turnover"]
        * 10_000.0
    )

    result["positive_at_zero_cost"] = (
        result["gross_mean_return"] > 0
    )

    return (
        result.sort_values(
            ["model", "cohort"]
        )
        .reset_index(drop=True)
    )


def _cost_column_label(
    cost_bps: float,
) -> str:
    if float(cost_bps).is_integer():
        return str(int(cost_bps))

    return (
        f"{cost_bps:g}"
        .replace(".", "p")
    )


def build_break_even_model_summary(
    cohort_metrics: pd.DataFrame,
    annualization_periods: int,
    robustness_costs_bps: Sequence[float],
) -> pd.DataFrame:
    if annualization_periods <= 0:
        raise ValueError(
            "annualization_periods must be positive"
        )

    rows = []

    for model, group in cohort_metrics.groupby(
        "model",
        sort=True,
    ):
        total_periods = int(
            group["period_count"].sum()
        )

        total_gross_return = float(
            group[
                "gross_cumulative_log_return"
            ].sum()
        )

        total_turnover = float(
            group["total_turnover"].sum()
        )

        if total_turnover <= 0:
            raise ValueError(
                f"Non-positive total turnover "
                f"for {model}"
            )

        equal_weighted_gross_mean = float(
            group["gross_mean_return"].mean()
        )

        equal_weighted_turnover = float(
            group["mean_turnover"].mean()
        )

        pooled_gross_mean = (
            total_gross_return
            / total_periods
        )

        pooled_mean_turnover = (
            total_turnover
            / total_periods
        )

        row = {
            "model": str(model),
            "cohort_count": len(group),
            "total_periods": total_periods,
            "equal_weighted_gross_mean_return": (
                equal_weighted_gross_mean
            ),
            "equal_weighted_gross_annualized_return": (
                equal_weighted_gross_mean
                * annualization_periods
            ),
            "pooled_gross_mean_return": (
                pooled_gross_mean
            ),
            "pooled_gross_annualized_return": (
                pooled_gross_mean
                * annualization_periods
            ),
            "equal_weighted_mean_turnover": (
                equal_weighted_turnover
            ),
            "pooled_mean_turnover": (
                pooled_mean_turnover
            ),
            "equal_weighted_break_even_cost_bps": (
                equal_weighted_gross_mean
                / equal_weighted_turnover
                * 10_000.0
            ),
            "pooled_break_even_cost_bps": (
                total_gross_return
                / total_turnover
                * 10_000.0
            ),
            "mean_individual_break_even_cost_bps": (
                float(
                    group[
                        "break_even_cost_bps"
                    ].mean()
                )
            ),
            "median_individual_break_even_cost_bps": (
                float(
                    group[
                        "break_even_cost_bps"
                    ].median()
                )
            ),
            "worst_individual_break_even_cost_bps": (
                float(
                    group[
                        "break_even_cost_bps"
                    ].min()
                )
            ),
            "best_individual_break_even_cost_bps": (
                float(
                    group[
                        "break_even_cost_bps"
                    ].max()
                )
            ),
        }

        for cost_bps in robustness_costs_bps:
            cost = float(cost_bps)

            net_means = (
                group["gross_mean_return"]
                - (
                    group["mean_turnover"]
                    * cost
                    / 10_000.0
                )
            )

            label = _cost_column_label(cost)

            row[
                (
                    "positive_cohort_fraction_at_"
                    f"{label}bps"
                )
            ] = float(
                (net_means > 0).mean()
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            "pooled_break_even_cost_bps",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_break_even_cost_curve(
    cohort_metrics: pd.DataFrame,
    cost_grid_bps: np.ndarray,
    annualization_periods: int,
) -> pd.DataFrame:
    costs = np.asarray(
        cost_grid_bps,
        dtype=np.float64,
    )

    if costs.ndim != 1:
        raise ValueError(
            "cost_grid_bps must be one-dimensional"
        )

    if len(costs) == 0:
        raise ValueError(
            "At least one cost value is required"
        )

    if not np.isfinite(costs).all():
        raise ValueError(
            "Cost grid contains non-finite values"
        )

    if np.any(costs < 0):
        raise ValueError(
            "Cost grid cannot contain "
            "negative costs"
        )

    if annualization_periods <= 0:
        raise ValueError(
            "annualization_periods must be positive"
        )

    rows = []

    for model, group in cohort_metrics.groupby(
        "model",
        sort=True,
    ):
        gross_means = group[
            "gross_mean_return"
        ].to_numpy(dtype=np.float64)

        turnovers = group[
            "mean_turnover"
        ].to_numpy(dtype=np.float64)

        period_counts = group[
            "period_count"
        ].to_numpy(dtype=np.float64)

        total_periods = int(
            period_counts.sum()
        )

        for cost_bps in costs:
            net_means = (
                gross_means
                - turnovers
                * cost_bps
                / 10_000.0
            )

            pooled_mean = float(
                np.average(
                    net_means,
                    weights=period_counts,
                )
            )

            rows.append(
                {
                    "model": str(model),
                    "cost_bps": float(
                        cost_bps
                    ),
                    "cohort_count": len(group),
                    "total_periods": (
                        total_periods
                    ),
                    "mean_cohort_net_return": (
                        float(
                            net_means.mean()
                        )
                    ),
                    "median_cohort_net_return": (
                        float(
                            np.median(
                                net_means
                            )
                        )
                    ),
                    "worst_cohort_net_return": (
                        float(
                            net_means.min()
                        )
                    ),
                    "best_cohort_net_return": (
                        float(
                            net_means.max()
                        )
                    ),
                    "mean_cohort_annualized_net_return": (
                        float(
                            net_means.mean()
                            * annualization_periods
                        )
                    ),
                    "median_cohort_annualized_net_return": (
                        float(
                            np.median(
                                net_means
                            )
                            * annualization_periods
                        )
                    ),
                    "worst_cohort_annualized_net_return": (
                        float(
                            net_means.min()
                            * annualization_periods
                        )
                    ),
                    "best_cohort_annualized_net_return": (
                        float(
                            net_means.max()
                            * annualization_periods
                        )
                    ),
                    "positive_cohort_fraction": (
                        float(
                            (
                                net_means > 0
                            ).mean()
                        )
                    ),
                    "pooled_mean_net_return": (
                        pooled_mean
                    ),
                    "pooled_annualized_net_return": (
                        pooled_mean
                        * annualization_periods
                    ),
                    "pooled_cumulative_net_log_return": (
                        float(
                            np.sum(
                                net_means
                                * period_counts
                            )
                        )
                    ),
                    "mean_turnover": float(
                        turnovers.mean()
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["model", "cost_bps"]
        )
        .reset_index(drop=True)
    )
