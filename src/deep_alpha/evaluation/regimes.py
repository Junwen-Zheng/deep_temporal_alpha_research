from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

REGIME_COLUMNS = {
    "direction": "direction_regime",
    "volatility": "volatility_regime",
    "dispersion": "dispersion_regime",
}


def validate_regime_quantiles(
    lower_quantile: float,
    upper_quantile: float,
) -> None:
    values = np.asarray(
        [
            lower_quantile,
            upper_quantile,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Regime quantiles must be finite"
        )

    if not (
        0.0
        < lower_quantile
        < upper_quantile
        < 1.0
    ):
        raise ValueError(
            "Regime quantiles must satisfy "
            "0 < lower < upper < 1"
        )


def build_market_state_panel(
    bars: pd.DataFrame,
    window_bars: int,
    expected_symbol_count: int,
) -> pd.DataFrame:
    required_columns = {
        "timestamp",
        "symbol",
        "close",
    }

    missing_columns = required_columns - set(
        bars.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing market-state columns: "
            f"{sorted(missing_columns)}"
        )

    if window_bars <= 0:
        raise ValueError(
            "window_bars must be positive"
        )

    if expected_symbol_count <= 1:
        raise ValueError(
            "expected_symbol_count must "
            "exceed one"
        )

    ordered = (
        bars.sort_values(
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
            "Market-state bar data is empty"
        )

    if ordered.duplicated(
        [
            "timestamp",
            "symbol",
        ]
    ).any():
        raise ValueError(
            "Duplicate market-state keys"
        )

    group_sizes = ordered.groupby(
        "timestamp",
        sort=False,
    ).size()

    if not group_sizes.eq(
        expected_symbol_count
    ).all():
        raise ValueError(
            "Market-state cross-sections "
            "are incomplete"
        )

    symbol_count = ordered[
        "symbol"
    ].nunique()

    if symbol_count != expected_symbol_count:
        raise ValueError(
            "Unexpected market-state "
            "symbol count"
        )

    close_values = ordered[
        "close"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(
        close_values
    ).all():
        raise ValueError(
            "Close prices contain "
            "non-finite values"
        )

    if np.any(close_values <= 0):
        raise ValueError(
            "Close prices must be positive"
        )

    close_matrix = (
        ordered.pivot(
            index="timestamp",
            columns="symbol",
            values="close",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    if close_matrix.shape[1] != (
        expected_symbol_count
    ):
        raise ValueError(
            "Unexpected close-matrix width"
        )

    timestamps = pd.DatetimeIndex(
        close_matrix.index
    )

    if timestamps.tz is None:
        raise ValueError(
            "Market-state timestamps must "
            "be timezone-aware"
        )

    if len(timestamps) <= window_bars:
        raise ValueError(
            "Insufficient bars for the "
            "state window"
        )

    differences = np.diff(
        timestamps.asi8
    )

    expected_interval = int(
        pd.Series(
            differences
        ).mode().iloc[0]
    )

    if expected_interval <= 0:
        raise ValueError(
            "Invalid timestamp interval"
        )

    if not np.all(
        differences == expected_interval
    ):
        raise ValueError(
            "Market-state timestamps are "
            "not contiguous"
        )

    log_close = np.log(
        close_matrix
    )

    bar_returns = log_close.diff()

    trailing_returns = (
        log_close
        - log_close.shift(window_bars)
    )

    realized_volatility = np.sqrt(
        bar_returns.pow(2)
        .rolling(
            window=window_bars,
            min_periods=window_bars,
        )
        .sum()
    )

    result = pd.DataFrame(
        {
            "timestamp": timestamps,
            "market_return": (
                trailing_returns.mean(
                    axis=1
                ).to_numpy()
            ),
            "market_realized_volatility": (
                realized_volatility.mean(
                    axis=1
                ).to_numpy()
            ),
            "cross_sectional_dispersion": (
                trailing_returns.std(
                    axis=1,
                    ddof=0,
                ).to_numpy()
            ),
        }
    )

    result = (
        result.dropna()
        .reset_index(drop=True)
    )

    numeric_columns = [
        "market_return",
        "market_realized_volatility",
        "cross_sectional_dispersion",
    ]

    if not np.isfinite(
        result[
            numeric_columns
        ].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Market-state metrics contain "
            "non-finite values"
        )

    if (
        result[
            "market_realized_volatility"
        ]
        < 0
    ).any():
        raise ValueError(
            "Realized volatility is negative"
        )

    if (
        result[
            "cross_sectional_dispersion"
        ]
        < 0
    ).any():
        raise ValueError(
            "Cross-sectional dispersion "
            "is negative"
        )

    return result


def assign_regime_labels(
    values: np.ndarray | pd.Series,
    lower_threshold: float,
    upper_threshold: float,
    labels: Sequence[str],
) -> np.ndarray:
    numeric_values = np.asarray(
        values,
        dtype=np.float64,
    )

    if numeric_values.ndim != 1:
        raise ValueError(
            "Regime values must be "
            "one-dimensional"
        )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            "Regime values contain "
            "non-finite values"
        )

    if not np.isfinite(
        [
            lower_threshold,
            upper_threshold,
        ]
    ).all():
        raise ValueError(
            "Regime thresholds must be finite"
        )

    if lower_threshold >= upper_threshold:
        raise ValueError(
            "Lower regime threshold must "
            "precede upper threshold"
        )

    labels_tuple = tuple(labels)

    if len(labels_tuple) != 3:
        raise ValueError(
            "Exactly three regime labels "
            "are required"
        )

    result = np.full(
        len(numeric_values),
        labels_tuple[1],
        dtype=object,
    )

    result[
        numeric_values <= lower_threshold
    ] = labels_tuple[0]

    result[
        numeric_values > upper_threshold
    ] = labels_tuple[2]

    return result


def build_fold_regime_assignments(
    market_states: pd.DataFrame,
    fold: object,
    test_timestamps: pd.Series | pd.DatetimeIndex,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    validate_regime_quantiles(
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    required_columns = {
        "timestamp",
        "market_return",
        "market_realized_volatility",
        "cross_sectional_dispersion",
    }

    missing_columns = required_columns - set(
        market_states.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing market-state columns: "
            f"{sorted(missing_columns)}"
        )

    fold_id = int(
        fold.fold_id
    )

    validation_end = pd.Timestamp(
        fold.validation_end
    )

    test_end = pd.Timestamp(
        fold.test_end
    )

    calibration = market_states.loc[
        market_states["timestamp"]
        <= validation_end
    ].copy()

    if calibration.empty:
        raise ValueError(
            f"Fold {fold_id} calibration "
            "sample is empty"
        )

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(
            test_timestamps,
            utc=True,
        )
    ).drop_duplicates().sort_values()

    if len(timestamps) == 0:
        raise ValueError(
            f"Fold {fold_id} test timestamps "
            "are empty"
        )

    if not (
        timestamps > validation_end
    ).all():
        raise ValueError(
            f"Fold {fold_id} test timestamps "
            "cross the validation boundary"
        )

    if not (
        timestamps <= test_end
    ).all():
        raise ValueError(
            f"Fold {fold_id} test timestamps "
            "cross the test boundary"
        )

    threshold_columns = [
        "market_return",
        "market_realized_volatility",
        "cross_sectional_dispersion",
    ]

    thresholds = {}

    for column in threshold_columns:
        lower = float(
            calibration[column].quantile(
                lower_quantile
            )
        )

        upper = float(
            calibration[column].quantile(
                upper_quantile
            )
        )

        if not np.isfinite(
            [
                lower,
                upper,
            ]
        ).all():
            raise ValueError(
                f"Fold {fold_id} {column} "
                "thresholds are non-finite"
            )

        if lower >= upper:
            raise ValueError(
                f"Fold {fold_id} {column} "
                "thresholds are not ordered"
            )

        thresholds[column] = (
            lower,
            upper,
        )

    test_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
        }
    ).merge(
        market_states,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    if test_frame[
        threshold_columns
    ].isna().any().any():
        raise ValueError(
            f"Fold {fold_id} is missing "
            "market-state values"
        )

    return_lower, return_upper = (
        thresholds["market_return"]
    )

    volatility_lower, volatility_upper = (
        thresholds[
            "market_realized_volatility"
        ]
    )

    dispersion_lower, dispersion_upper = (
        thresholds[
            "cross_sectional_dispersion"
        ]
    )

    test_frame.insert(
        0,
        "fold",
        fold_id,
    )

    test_frame["direction_regime"] = (
        assign_regime_labels(
            values=test_frame[
                "market_return"
            ],
            lower_threshold=return_lower,
            upper_threshold=return_upper,
            labels=(
                "down",
                "flat",
                "up",
            ),
        )
    )

    test_frame["volatility_regime"] = (
        assign_regime_labels(
            values=test_frame[
                "market_realized_volatility"
            ],
            lower_threshold=volatility_lower,
            upper_threshold=volatility_upper,
            labels=(
                "low",
                "medium",
                "high",
            ),
        )
    )

    test_frame["dispersion_regime"] = (
        assign_regime_labels(
            values=test_frame[
                "cross_sectional_dispersion"
            ],
            lower_threshold=dispersion_lower,
            upper_threshold=dispersion_upper,
            labels=(
                "low",
                "medium",
                "high",
            ),
        )
    )

    threshold_record = {
        "fold": fold_id,
        "calibration_timestamp_count": (
            calibration[
                "timestamp"
            ].nunique()
        ),
        "test_timestamp_count": len(
            test_frame
        ),
        "calibration_end_timestamp": (
            calibration[
                "timestamp"
            ].max()
        ),
        "validation_end": validation_end,
        "test_end": test_end,
        "market_return_lower": (
            return_lower
        ),
        "market_return_upper": (
            return_upper
        ),
        "market_realized_volatility_lower": (
            volatility_lower
        ),
        "market_realized_volatility_upper": (
            volatility_upper
        ),
        "cross_sectional_dispersion_lower": (
            dispersion_lower
        ),
        "cross_sectional_dispersion_upper": (
            dispersion_upper
        ),
    }

    return test_frame, threshold_record


def melt_regime_assignments(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "fold",
        "timestamp",
        *REGIME_COLUMNS.values(),
    }

    missing_columns = required_columns - set(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing regime assignment columns: "
            f"{sorted(missing_columns)}"
        )

    frames = []

    for dimension, column in (
        REGIME_COLUMNS.items()
    ):
        working = frame.copy()

        working[
            "regime_dimension"
        ] = dimension

        working["regime"] = working[
            column
        ]

        frames.append(working)

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "fold",
                "regime_dimension",
                "regime",
                "timestamp",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_regime_model_summary(
    prediction_metrics: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    focus_cost_bps: float,
) -> pd.DataFrame:
    required_prediction_columns = {
        "model",
        "fold",
        "regime_dimension",
        "regime",
        "mean_rank_ic",
    }

    missing_prediction_columns = (
        required_prediction_columns
        - set(prediction_metrics.columns)
    )

    if missing_prediction_columns:
        raise ValueError(
            "Missing regime prediction columns: "
            f"{sorted(missing_prediction_columns)}"
        )

    required_portfolio_columns = {
        "model",
        "fold",
        "regime_dimension",
        "regime",
        "cost_bps",
        "break_even_cost_bps",
        "pooled_annualized_net_return",
    }

    missing_portfolio_columns = (
        required_portfolio_columns
        - set(portfolio_metrics.columns)
    )

    if missing_portfolio_columns:
        raise ValueError(
            "Missing regime portfolio columns: "
            f"{sorted(missing_portfolio_columns)}"
        )

    zero_cost = portfolio_metrics.loc[
        np.isclose(
            portfolio_metrics[
                "cost_bps"
            ].to_numpy(
                dtype=np.float64
            ),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    focus_cost = portfolio_metrics.loc[
        np.isclose(
            portfolio_metrics[
                "cost_bps"
            ].to_numpy(
                dtype=np.float64
            ),
            focus_cost_bps,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    if zero_cost.empty:
        raise ValueError(
            "No zero-cost regime metrics"
        )

    if focus_cost.empty:
        raise ValueError(
            "No focus-cost regime metrics"
        )

    rows = []

    group_columns = [
        "model",
        "regime_dimension",
    ]

    for keys, predictions in (
        prediction_metrics.groupby(
            group_columns,
            sort=True,
        )
    ):
        model, regime_dimension = keys

        zero = zero_cost.loc[
            (
                zero_cost["model"]
                == model
            )
            & (
                zero_cost[
                    "regime_dimension"
                ]
                == regime_dimension
            )
        ]

        focus = focus_cost.loc[
            (
                focus_cost["model"]
                == model
            )
            & (
                focus_cost[
                    "regime_dimension"
                ]
                == regime_dimension
            )
        ]

        if zero.empty or focus.empty:
            raise ValueError(
                "Incomplete regime portfolio "
                f"metrics for {model}/"
                f"{regime_dimension}"
            )

        prediction_keys = set(
            zip(
                predictions["fold"],
                predictions["regime"],
                strict=True,
            )
        )

        zero_keys = set(
            zip(
                zero["fold"],
                zero["regime"],
                strict=True,
            )
        )

        focus_keys = set(
            zip(
                focus["fold"],
                focus["regime"],
                strict=True,
            )
        )

        if (
            prediction_keys
            != zero_keys
            or prediction_keys
            != focus_keys
        ):
            raise ValueError(
                "Regime prediction and "
                "portfolio cells differ"
            )

        rank_ic_values = predictions[
            "mean_rank_ic"
        ].to_numpy(dtype=np.float64)

        break_even_values = zero[
            "break_even_cost_bps"
        ].to_numpy(dtype=np.float64)

        focus_returns = focus[
            "pooled_annualized_net_return"
        ].to_numpy(dtype=np.float64)

        worst_prediction_index = (
            predictions[
                "mean_rank_ic"
            ].idxmin()
        )

        worst_break_even_index = (
            zero[
                "break_even_cost_bps"
            ].idxmin()
        )

        rows.append(
            {
                "model": str(model),
                "regime_dimension": str(
                    regime_dimension
                ),
                "regime_cell_count": len(
                    predictions
                ),
                "mean_cell_rank_ic": float(
                    rank_ic_values.mean()
                ),
                "worst_cell_rank_ic": float(
                    rank_ic_values.min()
                ),
                "best_cell_rank_ic": float(
                    rank_ic_values.max()
                ),
                "positive_rank_ic_cell_fraction": (
                    float(
                        (
                            rank_ic_values > 0
                        ).mean()
                    )
                ),
                "worst_rank_ic_fold": int(
                    predictions.loc[
                        worst_prediction_index,
                        "fold",
                    ]
                ),
                "worst_rank_ic_regime": str(
                    predictions.loc[
                        worst_prediction_index,
                        "regime",
                    ]
                ),
                "mean_break_even_cost_bps": (
                    float(
                        break_even_values.mean()
                    )
                ),
                "worst_break_even_cost_bps": (
                    float(
                        break_even_values.min()
                    )
                ),
                "best_break_even_cost_bps": (
                    float(
                        break_even_values.max()
                    )
                ),
                "positive_break_even_cell_fraction": (
                    float(
                        (
                            break_even_values > 0
                        ).mean()
                    )
                ),
                "worst_break_even_fold": int(
                    zero.loc[
                        worst_break_even_index,
                        "fold",
                    ]
                ),
                "worst_break_even_regime": str(
                    zero.loc[
                        worst_break_even_index,
                        "regime",
                    ]
                ),
                "focus_cost_bps": (
                    focus_cost_bps
                ),
                "mean_annualized_net_return_at_focus_cost": (
                    float(
                        focus_returns.mean()
                    )
                ),
                "worst_annualized_net_return_at_focus_cost": (
                    float(
                        focus_returns.min()
                    )
                ),
                "positive_cell_fraction_at_focus_cost": (
                    float(
                        (
                            focus_returns > 0
                        ).mean()
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "regime_dimension",
                "worst_cell_rank_ic",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )
