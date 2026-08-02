from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def validate_horizon_bars(
    horizon_bars: Sequence[int],
) -> tuple[int, ...]:
    horizons = tuple(
        sorted(
            {
                int(value)
                for value in horizon_bars
            }
        )
    )

    if not horizons:
        raise ValueError(
            "At least one horizon is required"
        )

    if any(value <= 0 for value in horizons):
        raise ValueError(
            "Horizons must be positive"
        )

    return horizons


def annualization_periods_for_horizon(
    interval_minutes: int,
    horizon_bars: int,
) -> int:
    if interval_minutes <= 0:
        raise ValueError(
            "interval_minutes must be positive"
        )

    if horizon_bars <= 0:
        raise ValueError(
            "horizon_bars must be positive"
        )

    minutes_per_period = (
        interval_minutes
        * horizon_bars
    )

    periods_float = (
        365
        * 24
        * 60
        / minutes_per_period
    )

    periods = round(periods_float)

    if not np.isclose(
        periods_float,
        periods,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Horizon does not produce an "
            "integer annualization factor"
        )

    return periods


def build_symbol_horizon_reference(
    bars: pd.DataFrame,
    horizon_bars: int,
    interval_minutes: int,
    entry_offset_bars: int = 1,
) -> pd.DataFrame:
    required_columns = {
        "timestamp",
        "open",
        "close",
    }

    missing_columns = required_columns - set(
        bars.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing bar columns: "
            f"{sorted(missing_columns)}"
        )

    if horizon_bars <= 0:
        raise ValueError(
            "horizon_bars must be positive"
        )

    if interval_minutes <= 0:
        raise ValueError(
            "interval_minutes must be positive"
        )

    if entry_offset_bars <= 0:
        raise ValueError(
            "entry_offset_bars must be positive"
        )

    if entry_offset_bars > horizon_bars:
        raise ValueError(
            "Entry offset cannot exceed "
            "the horizon"
        )

    ordered = (
        bars.sort_values(
            "timestamp",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if ordered.empty:
        raise ValueError(
            "Bar data is empty"
        )

    if ordered["timestamp"].duplicated().any():
        raise ValueError(
            "Bar timestamps contain duplicates"
        )

    timestamps = pd.DatetimeIndex(
        ordered["timestamp"]
    )

    if timestamps.tz is None:
        raise ValueError(
            "Bar timestamps must be "
            "timezone-aware"
        )

    expected_interval = pd.Timedelta(
        minutes=interval_minutes
    ).value

    if not np.all(
        np.diff(timestamps.asi8)
        == expected_interval
    ):
        raise ValueError(
            "Bar timestamps are not contiguous"
        )

    open_prices = ordered[
        "open"
    ].to_numpy(dtype=np.float64)

    close_prices = ordered[
        "close"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(
        open_prices
    ).all():
        raise ValueError(
            "Open prices contain "
            "non-finite values"
        )

    if not np.isfinite(
        close_prices
    ).all():
        raise ValueError(
            "Close prices contain "
            "non-finite values"
        )

    if np.any(open_prices <= 0):
        raise ValueError(
            "Open prices must be positive"
        )

    if np.any(close_prices <= 0):
        raise ValueError(
            "Close prices must be positive"
        )

    result = pd.DataFrame(
        {
            "timestamp": (
                ordered["timestamp"]
            ),
            "label_end_timestamp": (
                ordered["timestamp"].shift(
                    -horizon_bars
                )
            ),
            "raw_target": np.log(
                ordered["close"].shift(
                    -horizon_bars
                )
                / ordered["close"]
            ),
            "execution_return": np.log(
                ordered["close"].shift(
                    -horizon_bars
                )
                / ordered["open"].shift(
                    -entry_offset_bars
                )
            ),
        }
    )

    result["horizon_bars"] = (
        horizon_bars
    )

    result["horizon_minutes"] = (
        horizon_bars
        * interval_minutes
    )

    return result.dropna(
        subset=[
            "label_end_timestamp",
            "raw_target",
            "execution_return",
        ]
    ).reset_index(drop=True)


def build_cross_sectional_horizon_target(
    frame: pd.DataFrame,
    expected_symbol_count: int,
) -> pd.DataFrame:
    required_columns = {
        "timestamp",
        "symbol",
        "horizon_bars",
        "raw_target",
    }

    missing_columns = required_columns - set(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing horizon target columns: "
            f"{sorted(missing_columns)}"
        )

    if expected_symbol_count <= 1:
        raise ValueError(
            "expected_symbol_count must "
            "exceed one"
        )

    result = (
        frame.sort_values(
            [
                "horizon_bars",
                "timestamp",
                "symbol",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    group_columns = [
        "horizon_bars",
        "timestamp",
    ]

    group_sizes = result.groupby(
        group_columns,
        sort=False,
    ).size()

    if not group_sizes.eq(
        expected_symbol_count
    ).all():
        raise ValueError(
            "Horizon target cross-sections "
            "are incomplete"
        )

    values = result[
        "raw_target"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Raw horizon targets contain "
            "non-finite values"
        )

    cross_sectional_mean = result.groupby(
        group_columns,
        sort=False,
    )["raw_target"].transform("mean")

    result["target"] = (
        result["raw_target"]
        - cross_sectional_mean
    )

    target_values = result[
        "target"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(
        target_values
    ).all():
        raise ValueError(
            "Demeaned horizon targets "
            "contain non-finite values"
        )

    maximum_mean = float(
        result.groupby(
            group_columns,
            sort=False,
        )["target"]
        .mean()
        .abs()
        .max()
    )

    if maximum_mean > 1e-12:
        raise ValueError(
            "Horizon targets do not have "
            "zero cross-sectional mean"
        )

    return result


def build_horizon_portfolio_summary(
    cohort_metrics: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "model",
        "horizon_bars",
        "horizon_minutes",
        "cost_bps",
        "cohort",
        "period_count",
        "annualized_net_return",
        "annualized_volatility",
        "annualized_sharpe",
        "positive_fraction",
        "mean_turnover",
        "cumulative_net_log_return",
    }

    missing_columns = required_columns - set(
        cohort_metrics.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing horizon portfolio columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        cohort_metrics.groupby(
            [
                "model",
                "horizon_bars",
                "horizon_minutes",
                "cost_bps",
            ],
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
            median_annualized_net_return=(
                "annualized_net_return",
                "median",
            ),
            worst_annualized_net_return=(
                "annualized_net_return",
                "min",
            ),
            best_annualized_net_return=(
                "annualized_net_return",
                "max",
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
            [
                "cost_bps",
                "horizon_bars",
                "model",
            ]
        )
        .reset_index(drop=True)
    )


def build_horizon_break_even(
    cohort_metrics: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "model",
        "horizon_bars",
        "horizon_minutes",
        "cost_bps",
        "cohort",
        "period_count",
        "mean_net_return",
        "mean_turnover",
        "annualized_net_return",
        "annualized_sharpe",
        "cumulative_net_log_return",
    }

    missing_columns = required_columns - set(
        cohort_metrics.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing horizon break-even columns: "
            f"{sorted(missing_columns)}"
        )

    zero_cost = cohort_metrics.loc[
        np.isclose(
            cohort_metrics[
                "cost_bps"
            ].to_numpy(
                dtype=np.float64
            ),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
    ].copy()

    if zero_cost.empty:
        raise ValueError(
            "No zero-cost horizon metrics"
        )

    rows = []

    for keys, group in zero_cost.groupby(
        [
            "model",
            "horizon_bars",
            "horizon_minutes",
        ],
        sort=True,
    ):
        (
            model,
            horizon_bars,
            horizon_minutes,
        ) = keys

        total_turnover = float(
            (
                group["mean_turnover"]
                * group["period_count"]
            ).sum()
        )

        total_gross_return = float(
            group[
                "cumulative_net_log_return"
            ].sum()
        )

        if total_turnover <= 0:
            raise ValueError(
                "Horizon total turnover "
                "must be positive"
            )

        individual_break_even = (
            group["mean_net_return"]
            / group["mean_turnover"]
            * 10_000.0
        )

        rows.append(
            {
                "model": str(model),
                "horizon_bars": int(
                    horizon_bars
                ),
                "horizon_minutes": int(
                    horizon_minutes
                ),
                "cohort_count": len(
                    group
                ),
                "total_periods": int(
                    group[
                        "period_count"
                    ].sum()
                ),
                "equal_weighted_break_even_cost_bps": (
                    float(
                        group[
                            "mean_net_return"
                        ].mean()
                        / group[
                            "mean_turnover"
                        ].mean()
                        * 10_000.0
                    )
                ),
                "pooled_break_even_cost_bps": (
                    total_gross_return
                    / total_turnover
                    * 10_000.0
                ),
                "mean_individual_break_even_cost_bps": (
                    float(
                        individual_break_even.mean()
                    )
                ),
                "median_individual_break_even_cost_bps": (
                    float(
                        individual_break_even.median()
                    )
                ),
                "worst_individual_break_even_cost_bps": (
                    float(
                        individual_break_even.min()
                    )
                ),
                "best_individual_break_even_cost_bps": (
                    float(
                        individual_break_even.max()
                    )
                ),
                "mean_zero_cost_sharpe": (
                    float(
                        group[
                            "annualized_sharpe"
                        ].mean()
                    )
                ),
                "mean_gross_annualized_return": (
                    float(
                        group[
                            "annualized_net_return"
                        ].mean()
                    )
                ),
                "mean_turnover": float(
                    group[
                        "mean_turnover"
                    ].mean()
                ),
                "positive_cohort_fraction": (
                    float(
                        (
                            group[
                                "mean_net_return"
                            ]
                            > 0
                        ).mean()
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "horizon_bars",
                "pooled_break_even_cost_bps",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )
