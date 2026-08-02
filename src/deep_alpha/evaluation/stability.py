from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from deep_alpha.evaluation.metrics import (
    summarize_predictions,
)


def add_calendar_month(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    if timestamp_column not in frame.columns:
        raise ValueError(
            f"Missing timestamp column: "
            f"{timestamp_column}"
        )

    result = frame.copy()

    timestamps = pd.to_datetime(
        result[timestamp_column],
        utc=True,
    )

    if timestamps.isna().any():
        raise ValueError(
            "Timestamps contain NaT"
        )

    result["month"] = (
        timestamps.dt.strftime("%Y-%m")
    )

    return result


def summarize_grouped_predictions(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    columns = tuple(group_columns)

    if not columns:
        raise ValueError(
            "At least one grouping column "
            "is required"
        )

    missing_columns = set(columns) - set(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing prediction grouping columns: "
            f"{sorted(missing_columns)}"
        )

    rows = []

    grouped = frame.groupby(
        list(columns),
        sort=True,
        observed=True,
    )

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        metrics = summarize_predictions(
            group
        )

        row = dict(
            zip(
                columns,
                keys,
                strict=True,
            )
        )

        row.update(metrics)
        rows.append(row)

    if not rows:
        raise ValueError(
            "No prediction groups were found"
        )

    return (
        pd.DataFrame(rows)
        .sort_values(list(columns))
        .reset_index(drop=True)
    )


def summarize_temporal_portfolio_groups(
    periods: pd.DataFrame,
    group_columns: Sequence[str],
    cost_levels_bps: Sequence[float],
    annualization_periods: int,
) -> pd.DataFrame:
    columns = tuple(group_columns)

    if not columns:
        raise ValueError(
            "At least one grouping column "
            "is required"
        )

    required_columns = {
        *columns,
        "cohort",
        "timestamp",
        "gross_return",
        "turnover",
    }

    missing_columns = required_columns - set(
        periods.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing temporal portfolio columns: "
            f"{sorted(missing_columns)}"
        )

    if annualization_periods <= 0:
        raise ValueError(
            "annualization_periods must be positive"
        )

    costs = tuple(
        float(value)
        for value in cost_levels_bps
    )

    if not costs:
        raise ValueError(
            "At least one cost level is required"
        )

    cost_array = np.asarray(
        costs,
        dtype=np.float64,
    )

    if not np.isfinite(cost_array).all():
        raise ValueError(
            "Cost levels contain "
            "non-finite values"
        )

    if np.any(cost_array < 0):
        raise ValueError(
            "Cost levels cannot be negative"
        )

    numeric_values = periods[
        [
            "gross_return",
            "turnover",
        ]
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            "Portfolio periods contain "
            "non-finite values"
        )

    if (
        periods["turnover"] < 0
    ).any():
        raise ValueError(
            "Portfolio turnover is negative"
        )

    rows = []

    grouped = periods.groupby(
        list(columns),
        sort=True,
        observed=True,
    )

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        group_values = dict(
            zip(
                columns,
                keys,
                strict=True,
            )
        )

        gross_returns = group[
            "gross_return"
        ].to_numpy(dtype=np.float64)

        turnover = group[
            "turnover"
        ].to_numpy(dtype=np.float64)

        total_gross_return = float(
            gross_returns.sum()
        )

        total_turnover = float(
            turnover.sum()
        )

        if total_turnover <= 0:
            raise ValueError(
                "Temporal group has "
                "non-positive turnover"
            )

        break_even_cost = (
            total_gross_return
            / total_turnover
            * 10_000.0
        )

        for cost_bps in costs:
            net_returns = (
                gross_returns
                - turnover
                * cost_bps
                / 10_000.0
            )

            working = group[
                ["cohort"]
            ].copy()

            working["net_return"] = (
                net_returns
            )

            cohort_means = (
                working.groupby(
                    "cohort",
                    sort=True,
                )["net_return"]
                .mean()
            )

            if cohort_means.empty:
                raise ValueError(
                    "Temporal group has "
                    "no cohorts"
                )

            pooled_net_mean = float(
                net_returns.mean()
            )

            equal_weighted_net_mean = float(
                cohort_means.mean()
            )

            row = dict(group_values)

            row.update(
                {
                    "cost_bps": cost_bps,
                    "period_count": len(
                        group
                    ),
                    "cohort_count": int(
                        cohort_means.index.nunique()
                    ),
                    "total_gross_log_return": (
                        total_gross_return
                    ),
                    "total_turnover": (
                        total_turnover
                    ),
                    "mean_turnover": float(
                        turnover.mean()
                    ),
                    "break_even_cost_bps": (
                        break_even_cost
                    ),
                    "pooled_gross_mean_return": (
                        float(
                            gross_returns.mean()
                        )
                    ),
                    "pooled_net_mean_return": (
                        pooled_net_mean
                    ),
                    "equal_weighted_net_mean_return": (
                        equal_weighted_net_mean
                    ),
                    "pooled_annualized_net_return": (
                        pooled_net_mean
                        * annualization_periods
                    ),
                    "equal_weighted_annualized_net_return": (
                        equal_weighted_net_mean
                        * annualization_periods
                    ),
                    "worst_cohort_annualized_net_return": (
                        float(
                            cohort_means.min()
                            * annualization_periods
                        )
                    ),
                    "best_cohort_annualized_net_return": (
                        float(
                            cohort_means.max()
                            * annualization_periods
                        )
                    ),
                    "positive_cohort_fraction": (
                        float(
                            (
                                cohort_means > 0
                            ).mean()
                        )
                    ),
                }
            )

            rows.append(row)

    if not rows:
        raise ValueError(
            "No temporal portfolio groups "
            "were found"
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                *columns,
                "cost_bps",
            ]
        )
        .reset_index(drop=True)
    )


def _select_cost(
    frame: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    selected = frame.loc[
        np.isclose(
            frame["cost_bps"].to_numpy(
                dtype=np.float64
            ),
            cost_bps,
            rtol=0.0,
            atol=1e-12,
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No portfolio rows at "
            f"{cost_bps} bps"
        )

    return selected


def build_temporal_stability_summary(
    fold_prediction_metrics: pd.DataFrame,
    month_prediction_metrics: pd.DataFrame,
    fold_portfolio_metrics: pd.DataFrame,
    month_portfolio_metrics: pd.DataFrame,
    focus_cost_bps: float,
) -> pd.DataFrame:
    required_prediction_columns = {
        "model",
        "mean_rank_ic",
    }

    for frame in [
        fold_prediction_metrics,
        month_prediction_metrics,
    ]:
        missing_columns = (
            required_prediction_columns
            - set(frame.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing prediction stability "
                f"columns: {sorted(missing_columns)}"
            )

    required_portfolio_columns = {
        "model",
        "cost_bps",
        "break_even_cost_bps",
        "pooled_annualized_net_return",
    }

    for frame in [
        fold_portfolio_metrics,
        month_portfolio_metrics,
    ]:
        missing_columns = (
            required_portfolio_columns
            - set(frame.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing portfolio stability "
                f"columns: {sorted(missing_columns)}"
            )

    fold_zero = _select_cost(
        fold_portfolio_metrics,
        0.0,
    )

    month_zero = _select_cost(
        month_portfolio_metrics,
        0.0,
    )

    fold_focus = _select_cost(
        fold_portfolio_metrics,
        focus_cost_bps,
    )

    month_focus = _select_cost(
        month_portfolio_metrics,
        focus_cost_bps,
    )

    models = sorted(
        set(
            fold_prediction_metrics[
                "model"
            ]
        )
    )

    rows = []

    for model in models:
        fold_predictions = (
            fold_prediction_metrics.loc[
                fold_prediction_metrics[
                    "model"
                ]
                == model
            ]
        )

        month_predictions = (
            month_prediction_metrics.loc[
                month_prediction_metrics[
                    "model"
                ]
                == model
            ]
        )

        model_fold_zero = fold_zero.loc[
            fold_zero["model"] == model
        ]

        model_month_zero = month_zero.loc[
            month_zero["model"] == model
        ]

        model_fold_focus = fold_focus.loc[
            fold_focus["model"] == model
        ]

        model_month_focus = (
            month_focus.loc[
                month_focus["model"]
                == model
            ]
        )

        frames = [
            fold_predictions,
            month_predictions,
            model_fold_zero,
            model_month_zero,
            model_fold_focus,
            model_month_focus,
        ]

        if any(frame.empty for frame in frames):
            raise ValueError(
                f"Incomplete stability evidence "
                f"for {model}"
            )

        fold_ic = fold_predictions[
            "mean_rank_ic"
        ].to_numpy(dtype=np.float64)

        month_ic = month_predictions[
            "mean_rank_ic"
        ].to_numpy(dtype=np.float64)

        month_break_even = (
            model_month_zero[
                "break_even_cost_bps"
            ].to_numpy(dtype=np.float64)
        )

        fold_break_even = (
            model_fold_zero[
                "break_even_cost_bps"
            ].to_numpy(dtype=np.float64)
        )

        month_focus_returns = (
            model_month_focus[
                "pooled_annualized_net_return"
            ].to_numpy(dtype=np.float64)
        )

        fold_focus_returns = (
            model_fold_focus[
                "pooled_annualized_net_return"
            ].to_numpy(dtype=np.float64)
        )

        rows.append(
            {
                "model": model,
                "fold_count": len(
                    fold_predictions
                ),
                "month_segment_count": len(
                    month_predictions
                ),
                "mean_fold_rank_ic": float(
                    fold_ic.mean()
                ),
                "fold_rank_ic_std": float(
                    fold_ic.std(ddof=1)
                ),
                "worst_fold_rank_ic": float(
                    fold_ic.min()
                ),
                "best_fold_rank_ic": float(
                    fold_ic.max()
                ),
                "positive_fold_fraction": float(
                    (fold_ic > 0).mean()
                ),
                "mean_month_rank_ic": float(
                    month_ic.mean()
                ),
                "month_rank_ic_std": float(
                    month_ic.std(ddof=1)
                ),
                "worst_month_rank_ic": float(
                    month_ic.min()
                ),
                "best_month_rank_ic": float(
                    month_ic.max()
                ),
                "positive_month_rank_ic_fraction": (
                    float(
                        (month_ic > 0).mean()
                    )
                ),
                "mean_fold_break_even_cost_bps": (
                    float(
                        fold_break_even.mean()
                    )
                ),
                "worst_fold_break_even_cost_bps": (
                    float(
                        fold_break_even.min()
                    )
                ),
                "mean_month_break_even_cost_bps": (
                    float(
                        month_break_even.mean()
                    )
                ),
                "worst_month_break_even_cost_bps": (
                    float(
                        month_break_even.min()
                    )
                ),
                "positive_month_break_even_fraction": (
                    float(
                        (
                            month_break_even > 0
                        ).mean()
                    )
                ),
                "focus_cost_bps": (
                    focus_cost_bps
                ),
                "mean_fold_annualized_net_return_at_focus_cost": (
                    float(
                        fold_focus_returns.mean()
                    )
                ),
                "worst_fold_annualized_net_return_at_focus_cost": (
                    float(
                        fold_focus_returns.min()
                    )
                ),
                "positive_fold_fraction_at_focus_cost": (
                    float(
                        (
                            fold_focus_returns > 0
                        ).mean()
                    )
                ),
                "mean_month_annualized_net_return_at_focus_cost": (
                    float(
                        month_focus_returns.mean()
                    )
                ),
                "worst_month_annualized_net_return_at_focus_cost": (
                    float(
                        month_focus_returns.min()
                    )
                ),
                "positive_month_fraction_at_focus_cost": (
                    float(
                        (
                            month_focus_returns > 0
                        ).mean()
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "mean_month_rank_ic",
            ascending=False,
        )
        .reset_index(drop=True)
    )
