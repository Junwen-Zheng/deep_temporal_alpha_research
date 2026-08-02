from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from deep_alpha.evaluation.metrics import (
    validate_prediction_frame,
)


def assign_rebalance_cohort(
    timestamps: pd.Series,
    interval_minutes: int,
    holding_bars: int,
) -> np.ndarray:
    if interval_minutes <= 0:
        raise ValueError(
            "interval_minutes must be positive"
        )

    if holding_bars <= 0:
        raise ValueError(
            "holding_bars must be positive"
        )

    timestamp_index = pd.DatetimeIndex(timestamps)

    if timestamp_index.tz is None:
        raise ValueError(
            "Portfolio timestamps must be timezone-aware"
        )

    interval_nanoseconds = pd.Timedelta(
        minutes=interval_minutes
    ).value

    timestamp_nanoseconds = timestamp_index.asi8

    if np.any(
        timestamp_nanoseconds
        % interval_nanoseconds
        != 0
    ):
        raise ValueError(
            "Timestamps are not aligned to the bar interval"
        )

    bar_numbers = (
        timestamp_nanoseconds
        // interval_nanoseconds
    )

    return (
        bar_numbers % holding_bars
    ).astype(np.int16)


def build_rank_weights(
    frame: pd.DataFrame,
    top_count: int,
) -> pd.DataFrame:
    validate_prediction_frame(frame)

    required_columns = {
        "execution_return",
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            "Missing portfolio columns: "
            f"{sorted(missing_columns)}"
        )

    if top_count <= 0:
        raise ValueError(
            "top_count must be positive"
        )

    result = (
        frame.sort_values(
            ["timestamp", "symbol"],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    group_sizes = result.groupby(
        "timestamp",
        sort=False,
    )["symbol"].transform("size")

    unique_group_sizes = np.unique(
        group_sizes.to_numpy()
    )

    if len(unique_group_sizes) != 1:
        raise ValueError(
            "Cross-sectional symbol counts are inconsistent"
        )

    symbol_count = int(unique_group_sizes[0])

    if top_count * 2 > symbol_count:
        raise ValueError(
            "top_count is too large for the cross-section"
        )

    ranks = result.groupby(
        "timestamp",
        sort=False,
    )["prediction"].rank(
        method="first",
        ascending=True,
    )

    short_mask = ranks <= top_count

    long_mask = ranks > (
        symbol_count - top_count
    )

    weights = np.zeros(
        len(result),
        dtype=np.float64,
    )

    side_weight = 0.5 / top_count

    weights[short_mask.to_numpy()] = -side_weight
    weights[long_mask.to_numpy()] = side_weight

    result["weight"] = weights

    exposure = result.groupby(
        "timestamp",
        sort=False,
    )["weight"].agg(
        net_exposure="sum",
        gross_exposure=lambda values: float(
            np.abs(values).sum()
        ),
    )

    if (
        exposure["net_exposure"].abs().max()
        > 1e-12
    ):
        raise ValueError(
            "Portfolio is not dollar-neutral"
        )

    if not np.allclose(
        exposure["gross_exposure"].to_numpy(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Portfolio gross exposure is not one"
        )

    return result


def build_portfolio_periods(
    frame: pd.DataFrame,
    top_count: int,
    interval_minutes: int,
    holding_bars: int,
) -> pd.DataFrame:
    weighted = build_rank_weights(
        frame=frame,
        top_count=top_count,
    )

    weighted["cohort"] = assign_rebalance_cohort(
        timestamps=weighted["timestamp"],
        interval_minutes=interval_minutes,
        holding_bars=holding_bars,
    )

    index_columns = [
        "cohort",
        "timestamp",
    ]

    weight_matrix = weighted.pivot(
        index=index_columns,
        columns="symbol",
        values="weight",
    ).sort_index()

    return_matrix = weighted.pivot(
        index=index_columns,
        columns="symbol",
        values="execution_return",
    ).sort_index()

    if not weight_matrix.columns.equals(
        return_matrix.columns
    ):
        raise ValueError(
            "Weight and return symbol columns differ"
        )

    if weight_matrix.isna().any().any():
        raise ValueError(
            "Missing portfolio weights"
        )

    if return_matrix.isna().any().any():
        raise ValueError(
            "Missing execution returns"
        )

    gross_return = (
        weight_matrix * return_matrix
    ).sum(axis=1)

    holding_period = pd.Timedelta(
        minutes=(
            interval_minutes * holding_bars
        )
    )

    turnover_parts = []

    for cohort, cohort_weights in (
        weight_matrix.groupby(
            level="cohort",
            sort=True,
        )
    ):
        values = cohort_weights.to_numpy(
            dtype=np.float64,
        )

        timestamps = pd.DatetimeIndex(
            cohort_weights.index.get_level_values(
                "timestamp"
            )
        )

        differences = np.empty_like(values)

        differences[0] = values[0]

        if len(values) > 1:
            differences[1:] = (
                values[1:] - values[:-1]
            )

            gaps = (
                np.diff(timestamps.asi8)
                != holding_period.value
            )

            reset_positions = (
                np.flatnonzero(gaps) + 1
            )

            differences[reset_positions] = (
                values[reset_positions]
            )

        cohort_turnover = np.abs(
            differences
        ).sum(axis=1)

        turnover_parts.append(
            pd.DataFrame(
                {
                    "cohort": int(cohort),
                    "timestamp": timestamps,
                    "turnover": cohort_turnover,
                }
            )
        )

    turnover = pd.concat(
        turnover_parts,
        ignore_index=True,
    ).set_index(index_columns)["turnover"]

    periods = pd.DataFrame(
        {
            "gross_return": gross_return,
            "turnover": turnover,
        }
    ).reset_index()

    periods = periods.sort_values(
        ["cohort", "timestamp"]
    ).reset_index(drop=True)

    numeric_values = periods[
        [
            "gross_return",
            "turnover",
        ]
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Portfolio periods contain non-finite values"
        )

    if (periods["turnover"] < 0).any():
        raise ValueError(
            "Portfolio turnover is negative"
        )

    return periods


def apply_transaction_cost(
    periods: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    if cost_bps < 0:
        raise ValueError(
            "cost_bps cannot be negative"
        )

    result = periods.copy()

    result["cost"] = (
        result["turnover"]
        * cost_bps
        / 10_000.0
    )

    result["net_return"] = (
        result["gross_return"]
        - result["cost"]
    )

    return result


def summarize_portfolio_cohorts(
    periods: pd.DataFrame,
    cost_levels_bps: Sequence[float],
    annualization_periods: int,
) -> pd.DataFrame:
    if annualization_periods <= 0:
        raise ValueError(
            "annualization_periods must be positive"
        )

    rows = []

    for cost_bps in cost_levels_bps:
        cost_adjusted = apply_transaction_cost(
            periods=periods,
            cost_bps=float(cost_bps),
        )

        for cohort, group in (
            cost_adjusted.groupby(
                "cohort",
                sort=True,
            )
        ):
            returns = group[
                "net_return"
            ].to_numpy(dtype=np.float64)

            if len(returns) < 2:
                raise ValueError(
                    "A cohort has fewer than two periods"
                )

            mean_return = float(
                returns.mean()
            )

            standard_deviation = float(
                returns.std(ddof=1)
            )

            annualized_return = (
                mean_return
                * annualization_periods
            )

            annualized_volatility = (
                standard_deviation
                * np.sqrt(annualization_periods)
            )

            annualized_sharpe = np.nan

            if standard_deviation > 0:
                annualized_sharpe = float(
                    mean_return
                    / standard_deviation
                    * np.sqrt(
                        annualization_periods
                    )
                )

            rows.append(
                {
                    "cost_bps": float(cost_bps),
                    "cohort": int(cohort),
                    "period_count": len(group),
                    "mean_net_return": (
                        mean_return
                    ),
                    "annualized_net_return": (
                        annualized_return
                    ),
                    "annualized_volatility": (
                        annualized_volatility
                    ),
                    "annualized_sharpe": (
                        annualized_sharpe
                    ),
                    "positive_fraction": float(
                        (returns > 0).mean()
                    ),
                    "mean_turnover": float(
                        group["turnover"].mean()
                    ),
                    "cumulative_net_log_return": (
                        float(returns.sum())
                    ),
                }
            )

    result = pd.DataFrame(rows)

    numeric_columns = [
        "mean_net_return",
        "annualized_net_return",
        "annualized_volatility",
        "annualized_sharpe",
        "positive_fraction",
        "mean_turnover",
        "cumulative_net_log_return",
    ]

    if not np.isfinite(
        result[numeric_columns].to_numpy()
    ).all():
        raise ValueError(
            "Portfolio cohort metrics are non-finite"
        )

    return result.sort_values(
        ["cost_bps", "cohort"]
    ).reset_index(drop=True)


def summarize_portfolio_models(
    cohort_metrics: pd.DataFrame,
) -> pd.DataFrame:
    return (
        cohort_metrics.groupby(
            ["model", "cost_bps"],
            as_index=False,
            sort=True,
        )
        .agg(
            cohort_count=(
                "cohort",
                "nunique",
            ),
            total_periods=(
                "period_count",
                "sum",
            ),
            mean_cohort_sharpe=(
                "annualized_sharpe",
                "mean",
            ),
            median_cohort_sharpe=(
                "annualized_sharpe",
                "median",
            ),
            worst_cohort_sharpe=(
                "annualized_sharpe",
                "min",
            ),
            best_cohort_sharpe=(
                "annualized_sharpe",
                "max",
            ),
            mean_annualized_net_return=(
                "annualized_net_return",
                "mean",
            ),
            worst_annualized_net_return=(
                "annualized_net_return",
                "min",
            ),
            mean_annualized_volatility=(
                "annualized_volatility",
                "mean",
            ),
            mean_positive_fraction=(
                "positive_fraction",
                "mean",
            ),
            mean_turnover=(
                "mean_turnover",
                "mean",
            ),
            mean_cumulative_net_log_return=(
                "cumulative_net_log_return",
                "mean",
            ),
        )
        .sort_values(
            ["cost_bps", "model"]
        )
        .reset_index(drop=True)
    )
