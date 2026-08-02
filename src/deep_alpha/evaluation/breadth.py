from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deep_alpha.evaluation.portfolio import (
    assign_rebalance_cohort,
)


@dataclass(frozen=True)
class PredictionMatrices:
    timestamps: pd.DatetimeIndex
    symbols: tuple[str, ...]
    predictions: np.ndarray
    execution_returns: np.ndarray

    @property
    def timestamp_count(self) -> int:
        return len(self.timestamps)

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)


def build_prediction_matrices(
    frame: pd.DataFrame,
    expected_symbol_count: int,
) -> PredictionMatrices:
    required_columns = {
        "timestamp",
        "symbol",
        "prediction",
        "execution_return",
    }

    missing_columns = required_columns - set(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing prediction matrix columns: "
            f"{sorted(missing_columns)}"
        )

    if expected_symbol_count <= 1:
        raise ValueError(
            "expected_symbol_count must exceed one"
        )

    ordered = (
        frame.sort_values(
            [
                "timestamp",
                "symbol",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if ordered.empty:
        raise ValueError(
            "Prediction data is empty"
        )

    group_sizes = ordered.groupby(
        "timestamp",
        sort=False,
    ).size()

    if not group_sizes.eq(
        expected_symbol_count
    ).all():
        raise ValueError(
            "Prediction cross-sections "
            "are incomplete"
        )

    timestamp_count = len(group_sizes)

    symbols = tuple(
        sorted(
            str(symbol)
            for symbol
            in ordered["symbol"].unique()
        )
    )

    if len(symbols) != expected_symbol_count:
        raise ValueError(
            "Unexpected symbol count"
        )

    symbol_matrix = (
        ordered["symbol"]
        .to_numpy()
        .reshape(
            timestamp_count,
            expected_symbol_count,
        )
    )

    expected_symbol_row = np.asarray(
        symbols,
        dtype=object,
    )

    if not np.all(
        symbol_matrix
        == expected_symbol_row[None, :]
    ):
        raise ValueError(
            "Symbol ordering differs "
            "across timestamps"
        )

    timestamp_matrix = (
        ordered["timestamp"]
        .array.asi8
        .reshape(
            timestamp_count,
            expected_symbol_count,
        )
    )

    if not np.all(
        timestamp_matrix
        == timestamp_matrix[:, [0]]
    ):
        raise ValueError(
            "Timestamp values differ "
            "inside a cross-section"
        )

    predictions = (
        ordered["prediction"]
        .to_numpy(dtype=np.float64)
        .reshape(
            timestamp_count,
            expected_symbol_count,
        )
    )

    execution_returns = (
        ordered["execution_return"]
        .to_numpy(dtype=np.float64)
        .reshape(
            timestamp_count,
            expected_symbol_count,
        )
    )

    if not np.isfinite(
        predictions
    ).all():
        raise ValueError(
            "Predictions contain "
            "non-finite values"
        )

    if not np.isfinite(
        execution_returns
    ).all():
        raise ValueError(
            "Execution returns contain "
            "non-finite values"
        )

    timestamps = pd.to_datetime(
        timestamp_matrix[:, 0],
        utc=True,
    )

    return PredictionMatrices(
        timestamps=pd.DatetimeIndex(
            timestamps
        ),
        symbols=symbols,
        predictions=predictions,
        execution_returns=(
            execution_returns
        ),
    )


def build_rank_weight_matrix(
    predictions: np.ndarray,
    top_count: int,
) -> np.ndarray:
    values = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "Predictions must be "
            "two-dimensional"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Predictions contain "
            "non-finite values"
        )

    timestamp_count, symbol_count = (
        values.shape
    )

    if timestamp_count == 0:
        raise ValueError(
            "Predictions are empty"
        )

    if top_count <= 0:
        raise ValueError(
            "top_count must be positive"
        )

    if top_count * 2 > symbol_count:
        raise ValueError(
            "top_count is too large "
            "for the symbol universe"
        )

    order = np.argsort(
        values,
        axis=1,
        kind="stable",
    )

    weights = np.zeros_like(
        values,
        dtype=np.float64,
    )

    row_indices = np.arange(
        timestamp_count,
        dtype=np.int64,
    )[:, None]

    side_weight = 0.5 / top_count

    weights[
        row_indices,
        order[:, :top_count],
    ] = -side_weight

    weights[
        row_indices,
        order[:, -top_count:],
    ] = side_weight

    net_exposure = weights.sum(axis=1)

    gross_exposure = np.abs(
        weights
    ).sum(axis=1)

    if not np.allclose(
        net_exposure,
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Breadth portfolio is not "
            "dollar-neutral"
        )

    if not np.allclose(
        gross_exposure,
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Breadth portfolio does not "
            "have unit gross exposure"
        )

    return weights


def build_breadth_periods(
    matrices: PredictionMatrices,
    top_count: int,
    interval_minutes: int,
    holding_bars: int,
) -> pd.DataFrame:
    if interval_minutes <= 0:
        raise ValueError(
            "interval_minutes must be positive"
        )

    if holding_bars <= 0:
        raise ValueError(
            "holding_bars must be positive"
        )

    weights = build_rank_weight_matrix(
        predictions=matrices.predictions,
        top_count=top_count,
    )

    gross_returns = (
        weights
        * matrices.execution_returns
    ).sum(axis=1)

    cohorts = assign_rebalance_cohort(
        timestamps=pd.Series(
            matrices.timestamps
        ),
        interval_minutes=interval_minutes,
        holding_bars=holding_bars,
    )

    turnover = np.empty(
        matrices.timestamp_count,
        dtype=np.float64,
    )

    holding_period_nanoseconds = (
        pd.Timedelta(
            minutes=(
                interval_minutes
                * holding_bars
            )
        ).value
    )

    timestamp_nanoseconds = (
        matrices.timestamps.asi8
    )

    observed_cohorts = np.unique(cohorts)

    for cohort in observed_cohorts:
        positions = np.flatnonzero(
            cohorts == cohort
        )

        cohort_weights = weights[
            positions
        ]

        differences = np.empty_like(
            cohort_weights
        )

        differences[0] = (
            cohort_weights[0]
        )

        if len(positions) > 1:
            differences[1:] = (
                cohort_weights[1:]
                - cohort_weights[:-1]
            )

            gaps = (
                np.diff(
                    timestamp_nanoseconds[
                        positions
                    ]
                )
                != holding_period_nanoseconds
            )

            reset_positions = (
                np.flatnonzero(gaps) + 1
            )

            differences[
                reset_positions
            ] = cohort_weights[
                reset_positions
            ]

        turnover[positions] = np.abs(
            differences
        ).sum(axis=1)

    periods = pd.DataFrame(
        {
            "cohort": cohorts,
            "timestamp": (
                matrices.timestamps
            ),
            "gross_return": gross_returns,
            "turnover": turnover,
        }
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
            "Breadth periods contain "
            "non-finite values"
        )

    if (
        periods["turnover"] < 0
    ).any():
        raise ValueError(
            "Breadth turnover is negative"
        )

    return (
        periods.sort_values(
            [
                "cohort",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )


def build_breadth_summary(
    cohort_metrics: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "model",
        "top_count",
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
            "Missing breadth cohort columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        cohort_metrics.groupby(
            [
                "model",
                "top_count",
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
        .assign(
            selected_symbol_count=lambda frame: (
                frame["top_count"] * 2
            )
        )
        .sort_values(
            [
                "cost_bps",
                "top_count",
                "model",
            ]
        )
        .reset_index(drop=True)
    )


def build_breadth_break_even(
    cohort_metrics: pd.DataFrame,
) -> pd.DataFrame:
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
            "No zero-cost breadth metrics"
        )

    rows = []

    for (
        model,
        top_count,
    ), group in zero_cost.groupby(
        [
            "model",
            "top_count",
        ],
        sort=True,
    ):
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
                "Total breadth turnover "
                "must be positive"
            )

        individual_break_even = (
            group["mean_net_return"]
            / group["mean_turnover"]
            * 10_000.0
        )

        equal_weighted_break_even = (
            group["mean_net_return"].mean()
            / group["mean_turnover"].mean()
            * 10_000.0
        )

        pooled_break_even = (
            total_gross_return
            / total_turnover
            * 10_000.0
        )

        rows.append(
            {
                "model": str(model),
                "top_count": int(
                    top_count
                ),
                "selected_symbol_count": (
                    int(top_count) * 2
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
                        equal_weighted_break_even
                    )
                ),
                "pooled_break_even_cost_bps": (
                    float(
                        pooled_break_even
                    )
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
                "mean_gross_annualized_return": (
                    float(
                        group[
                            "annualized_net_return"
                        ].mean()
                    )
                ),
                "mean_zero_cost_sharpe": (
                    float(
                        group[
                            "annualized_sharpe"
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
                "top_count",
                "pooled_break_even_cost_bps",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )
