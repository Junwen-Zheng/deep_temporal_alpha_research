from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_12",
    "return_48",
    "log_range",
    "close_open_return",
    "realized_vol_12",
    "realized_vol_48",
    "log_quote_volume",
    "volume_zscore_48",
    "trade_count_zscore_48",
    "taker_buy_ratio",
    "time_of_day_sin",
    "time_of_day_cos",
    "weekday_sin",
    "weekday_cos",
]

METADATA_COLUMNS = [
    "timestamp",
    "available_time",
    "label_end_timestamp",
    "label_end_time",
    "symbol",
    "close",
    "raw_target",
    "target",
]


def rolling_zscore(
    values: pd.Series,
    window: int,
) -> pd.Series:
    rolling = values.rolling(
        window=window,
        min_periods=window,
    )

    mean = rolling.mean()
    standard_deviation = rolling.std(ddof=0)

    return (
        (values - mean)
        / standard_deviation.replace(0.0, np.nan)
    )


def build_symbol_features(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "timestamp",
        "available_time",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "quote_volume",
        "trade_count",
        "taker_buy_quote_volume",
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing feature inputs: {sorted(missing_columns)}"
        )

    result = (
        frame.sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    if result["symbol"].nunique() != 1:
        raise ValueError(
            "build_symbol_features requires one symbol"
        )

    log_close = np.log(result["close"])
    one_bar_return = log_close.diff()

    result["return_1"] = one_bar_return
    result["return_3"] = log_close.diff(3)
    result["return_12"] = log_close.diff(12)
    result["return_48"] = log_close.diff(48)

    result["log_range"] = np.log(
        result["high"] / result["low"]
    )

    result["close_open_return"] = np.log(
        result["close"] / result["open"]
    )

    result["realized_vol_12"] = (
        one_bar_return
        .rolling(window=12, min_periods=12)
        .std(ddof=0)
        .mul(np.sqrt(12))
    )

    result["realized_vol_48"] = (
        one_bar_return
        .rolling(window=48, min_periods=48)
        .std(ddof=0)
        .mul(np.sqrt(48))
    )

    result["log_quote_volume"] = np.log1p(
        result["quote_volume"]
    )

    result["volume_zscore_48"] = rolling_zscore(
        result["log_quote_volume"],
        window=48,
    )

    result["trade_count_zscore_48"] = rolling_zscore(
        np.log1p(result["trade_count"]),
        window=48,
    )

    quote_volume = result["quote_volume"].to_numpy(
        dtype=np.float64
    )

    taker_buy_quote_volume = result[
        "taker_buy_quote_volume"
    ].to_numpy(dtype=np.float64)

    taker_buy_ratio = np.full(
        shape=len(result),
        fill_value=0.5,
        dtype=np.float64,
    )

    np.divide(
        taker_buy_quote_volume,
        quote_volume,
        out=taker_buy_ratio,
        where=quote_volume > 0.0,
    )

    result["taker_buy_ratio"] = taker_buy_ratio

    minute_of_day = (
        result["timestamp"].dt.hour * 60
        + result["timestamp"].dt.minute
    )

    time_angle = (
        2.0
        * np.pi
        * minute_of_day
        / (24 * 60)
    )

    result["time_of_day_sin"] = np.sin(time_angle)
    result["time_of_day_cos"] = np.cos(time_angle)

    weekday_angle = (
        2.0
        * np.pi
        * result["timestamp"].dt.dayofweek
        / 7
    )

    result["weekday_sin"] = np.sin(weekday_angle)
    result["weekday_cos"] = np.cos(weekday_angle)

    return result


def add_forward_targets(
    panel: pd.DataFrame,
    horizon_bars: int,
    interval_minutes: int,
) -> pd.DataFrame:
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")

    result = (
        panel.sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
        .copy()
    )

    grouped = result.groupby("symbol", sort=False)

    future_close = grouped["close"].shift(-horizon_bars)

    result["label_end_timestamp"] = grouped[
        "timestamp"
    ].shift(-horizon_bars)

    result["label_end_time"] = grouped[
        "available_time"
    ].shift(-horizon_bars)

    result["raw_target"] = np.log(
        future_close / result["close"]
    )

    valid_label_rows = result["label_end_timestamp"].notna()

    expected_horizon = pd.Timedelta(
        minutes=horizon_bars * interval_minutes
    )

    actual_horizon = (
        result.loc[valid_label_rows, "label_end_timestamp"]
        - result.loc[valid_label_rows, "timestamp"]
    )

    if not actual_horizon.eq(expected_horizon).all():
        raise ValueError(
            "Forward labels are not aligned to the configured horizon"
        )

    cross_sectional_mean = result.groupby("timestamp")[
        "raw_target"
    ].transform("mean")

    result["target"] = (
        result["raw_target"] - cross_sectional_mean
    )

    return result


def finalize_research_panel(
    panel: pd.DataFrame,
    expected_symbol_count: int,
) -> pd.DataFrame:
    required_columns = [
        *METADATA_COLUMNS,
        *FEATURE_COLUMNS,
    ]

    missing_columns = set(required_columns) - set(panel.columns)

    if missing_columns:
        raise ValueError(
            f"Missing research columns: {sorted(missing_columns)}"
        )

    result = panel.dropna(
        subset=[
            *FEATURE_COLUMNS,
            "raw_target",
            "target",
            "label_end_timestamp",
            "label_end_time",
        ]
    ).copy()

    result = result.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)

    if result.duplicated(["symbol", "timestamp"]).any():
        raise ValueError(
            "Duplicate symbol/timestamp rows in research panel"
        )

    feature_values = result[FEATURE_COLUMNS].to_numpy()

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Non-finite feature values in research panel"
        )

    target_values = result[
        ["raw_target", "target"]
    ].to_numpy()

    if not np.isfinite(target_values).all():
        raise ValueError(
            "Non-finite target values in research panel"
        )

    symbol_counts = result.groupby("timestamp")[
        "symbol"
    ].nunique()

    if not symbol_counts.eq(expected_symbol_count).all():
        invalid_counts = symbol_counts[
            symbol_counts != expected_symbol_count
        ]

        raise ValueError(
            "Incomplete cross-sections found: "
            f"{invalid_counts.head().to_dict()}"
        )

    cross_sectional_target_mean = result.groupby(
        "timestamp"
    )["target"].mean()

    if cross_sectional_target_mean.abs().max() > 1e-12:
        raise ValueError(
            "Cross-sectional target means are not zero"
        )

    if (
        result["available_time"]
        > result["label_end_time"]
    ).any():
        raise ValueError(
            "Feature availability occurs after label end time"
        )

    for column in FEATURE_COLUMNS:
        result[column] = result[column].astype("float32")

    result["close"] = result["close"].astype("float64")
    result["raw_target"] = result["raw_target"].astype(
        "float64"
    )
    result["target"] = result["target"].astype("float64")

    return result[required_columns]
