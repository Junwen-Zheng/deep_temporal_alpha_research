from __future__ import annotations

import numpy as np
import pandas as pd


def build_bar_alignment_reference(
    bars: pd.DataFrame,
    horizon_bars: int,
    entry_offset_bars: int,
    exit_offset_bars: int,
    interval_minutes: int,
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

    positive_values = {
        "horizon_bars": horizon_bars,
        "entry_offset_bars": entry_offset_bars,
        "exit_offset_bars": exit_offset_bars,
        "interval_minutes": interval_minutes,
    }

    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(
                f"{name} must be positive"
            )

    ordered = (
        bars.sort_values(
            "timestamp",
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    if ordered.empty:
        raise ValueError("Bar data is empty")

    if ordered["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate bar timestamps"
        )

    timestamp_index = pd.DatetimeIndex(
        ordered["timestamp"]
    )

    if timestamp_index.tz is None:
        raise ValueError(
            "Bar timestamps must be timezone-aware"
        )

    expected_interval = pd.Timedelta(
        minutes=interval_minutes
    ).value

    differences = np.diff(
        timestamp_index.asi8
    )

    if not np.all(
        differences == expected_interval
    ):
        raise ValueError(
            "Bar timestamps are not contiguous"
        )

    open_prices = ordered["open"].to_numpy(
        dtype=np.float64
    )

    close_prices = ordered["close"].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(open_prices).all():
        raise ValueError(
            "Open prices contain non-finite values"
        )

    if not np.isfinite(close_prices).all():
        raise ValueError(
            "Close prices contain non-finite values"
        )

    if np.any(open_prices <= 0):
        raise ValueError(
            "Open prices must be positive"
        )

    if np.any(close_prices <= 0):
        raise ValueError(
            "Close prices must be positive"
        )

    reference = ordered[
        ["timestamp"]
    ].copy()

    reference[
        "expected_label_end_timestamp"
    ] = ordered["timestamp"].shift(
        -horizon_bars
    )

    reference["expected_raw_target"] = np.log(
        ordered["close"].shift(
            -horizon_bars
        )
        / ordered["close"]
    )

    reference[
        "expected_execution_return"
    ] = np.log(
        ordered["close"].shift(
            -exit_offset_bars
        )
        / ordered["open"].shift(
            -entry_offset_bars
        )
    )

    return reference


def cross_sectionally_demean(
    frame: pd.DataFrame,
    value_column: str,
    timestamp_column: str = "timestamp",
) -> np.ndarray:
    required_columns = {
        timestamp_column,
        value_column,
    }

    missing_columns = required_columns - set(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing demeaning columns: "
            f"{sorted(missing_columns)}"
        )

    values = frame[value_column].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Values to demean contain "
            "non-finite values"
        )

    cross_sectional_mean = frame.groupby(
        timestamp_column,
        sort=False,
    )[value_column].transform("mean")

    result = (
        frame[value_column]
        - cross_sectional_mean
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(result).all():
        raise ValueError(
            "Demeaned values contain "
            "non-finite values"
        )

    return result


def maximum_absolute_error(
    actual: np.ndarray,
    expected: np.ndarray,
) -> float:
    actual_values = np.asarray(
        actual,
        dtype=np.float64,
    )

    expected_values = np.asarray(
        expected,
        dtype=np.float64,
    )

    if actual_values.shape != expected_values.shape:
        raise ValueError(
            "Numeric alignment shapes differ"
        )

    if not np.isfinite(actual_values).all():
        raise ValueError(
            "Actual values contain non-finite values"
        )

    if not np.isfinite(expected_values).all():
        raise ValueError(
            "Expected values contain non-finite values"
        )

    return float(
        np.max(
            np.abs(
                actual_values
                - expected_values
            )
        )
    )


def maximum_timestamp_error_nanoseconds(
    actual: pd.Series,
    expected: pd.Series,
) -> int:
    actual_index = pd.DatetimeIndex(actual)
    expected_index = pd.DatetimeIndex(expected)

    if len(actual_index) != len(expected_index):
        raise ValueError(
            "Timestamp alignment lengths differ"
        )

    if actual_index.hasnans:
        raise ValueError(
            "Actual timestamps contain NaT"
        )

    if expected_index.hasnans:
        raise ValueError(
            "Expected timestamps contain NaT"
        )

    return int(
        np.abs(
            actual_index.asi8
            - expected_index.asi8
        ).max()
    )


def assert_key_alignment(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
) -> None:
    key_columns = [
        "timestamp",
        "symbol",
    ]

    for column in key_columns:
        if column not in expected.columns:
            raise ValueError(
                f"Expected data lacks {column}"
            )

        if column not in actual.columns:
            raise ValueError(
                f"Actual data lacks {column}"
            )

    if len(expected) != len(actual):
        raise ValueError(
            "Aligned row counts differ"
        )

    expected_timestamps = pd.DatetimeIndex(
        expected["timestamp"]
    )

    actual_timestamps = pd.DatetimeIndex(
        actual["timestamp"]
    )

    if not np.array_equal(
        expected_timestamps.asi8,
        actual_timestamps.asi8,
    ):
        raise ValueError(
            "Aligned timestamps differ"
        )

    if not np.array_equal(
        expected["symbol"].to_numpy(),
        actual["symbol"].to_numpy(),
    ):
        raise ValueError(
            "Aligned symbols differ"
        )
