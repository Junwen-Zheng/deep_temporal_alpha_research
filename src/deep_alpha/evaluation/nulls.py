from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deep_alpha.evaluation.metrics import (
    validate_prediction_frame,
)


@dataclass(frozen=True)
class RankedPredictionPanel:
    timestamps: pd.DatetimeIndex
    symbols: tuple[str, ...]
    prediction_ranks: np.ndarray
    target_ranks: np.ndarray

    @property
    def timestamp_count(self) -> int:
        return len(self.timestamps)

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)


def rank_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            "Rank input must be two-dimensional"
        )

    if array.shape[1] < 2:
        raise ValueError(
            "At least two cross-sectional values are required"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Rank input contains non-finite values"
        )

    return (
        pd.DataFrame(array)
        .rank(
            axis=1,
            method="average",
            ascending=True,
        )
        .to_numpy(dtype=np.float64)
    )


def cross_sectional_rank_ic(
    prediction_ranks: np.ndarray,
    target_ranks: np.ndarray,
) -> np.ndarray:
    predictions = np.asarray(
        prediction_ranks,
        dtype=np.float64,
    )

    targets = np.asarray(
        target_ranks,
        dtype=np.float64,
    )

    if predictions.shape != targets.shape:
        raise ValueError(
            "Prediction and target rank shapes differ"
        )

    if predictions.ndim != 2:
        raise ValueError(
            "Rank matrices must be two-dimensional"
        )

    centered_predictions = (
        predictions
        - predictions.mean(
            axis=1,
            keepdims=True,
        )
    )

    centered_targets = (
        targets
        - targets.mean(
            axis=1,
            keepdims=True,
        )
    )

    numerator = (
        centered_predictions
        * centered_targets
    ).sum(axis=1)

    denominator = np.sqrt(
        (
            centered_predictions**2
        ).sum(axis=1)
        * (
            centered_targets**2
        ).sum(axis=1)
    )

    correlations = np.full(
        len(numerator),
        np.nan,
        dtype=np.float64,
    )

    valid = denominator > 0

    correlations[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return correlations


def permute_rows(
    values: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            "Permutation input must be two-dimensional"
        )

    random_keys = generator.random(
        array.shape
    )

    permutation_indices = np.argsort(
        random_keys,
        axis=1,
        kind="stable",
    )

    return np.take_along_axis(
        array,
        permutation_indices,
        axis=1,
    )


def shifted_cross_sectional_rank_ic(
    prediction_ranks: np.ndarray,
    target_ranks: np.ndarray,
    shift_bars: int,
) -> np.ndarray:
    predictions = np.asarray(
        prediction_ranks,
        dtype=np.float64,
    )

    targets = np.asarray(
        target_ranks,
        dtype=np.float64,
    )

    if predictions.shape != targets.shape:
        raise ValueError(
            "Prediction and target rank shapes differ"
        )

    if abs(shift_bars) >= len(predictions):
        raise ValueError(
            "Time shift removes every timestamp"
        )

    if shift_bars > 0:
        aligned_predictions = predictions[
            :-shift_bars
        ]

        aligned_targets = targets[
            shift_bars:
        ]
    elif shift_bars < 0:
        offset = -shift_bars

        aligned_predictions = predictions[
            offset:
        ]

        aligned_targets = targets[
            :-offset
        ]
    else:
        aligned_predictions = predictions
        aligned_targets = targets

    return cross_sectional_rank_ic(
        aligned_predictions,
        aligned_targets,
    )


def empirical_two_sided_p_value(
    observed_value: float,
    null_values: np.ndarray,
) -> float:
    values = np.asarray(
        null_values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "Null values must be one-dimensional"
        )

    if len(values) == 0:
        raise ValueError(
            "At least one null value is required"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Null values contain non-finite values"
        )

    exceedances = int(
        (
            np.abs(values)
            >= abs(observed_value)
        ).sum()
    )

    return (
        exceedances + 1
    ) / (
        len(values) + 1
    )


def build_ranked_prediction_panel(
    frame: pd.DataFrame,
    expected_symbol_count: int,
) -> RankedPredictionPanel:
    validate_prediction_frame(frame)

    if expected_symbol_count <= 1:
        raise ValueError(
            "expected_symbol_count must exceed one"
        )

    ordered = (
        frame.sort_values(
            ["timestamp", "symbol"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    group_sizes = ordered.groupby(
        "timestamp",
        sort=False,
    ).size()

    if not group_sizes.eq(
        expected_symbol_count
    ).all():
        invalid = group_sizes[
            group_sizes
            != expected_symbol_count
        ]

        raise ValueError(
            "Incomplete prediction cross-sections: "
            f"{invalid.head().to_dict()}"
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
            f"Expected {expected_symbol_count} symbols, "
            f"found {len(symbols)}"
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
            "Symbol ordering differs across timestamps"
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
            "Timestamp values differ within a cross-section"
        )

    predictions = (
        ordered["prediction"]
        .to_numpy(
            dtype=np.float64,
        )
        .reshape(
            timestamp_count,
            expected_symbol_count,
        )
    )

    targets = (
        ordered["target"]
        .to_numpy(
            dtype=np.float64,
        )
        .reshape(
            timestamp_count,
            expected_symbol_count,
        )
    )

    timestamps = pd.to_datetime(
        timestamp_matrix[:, 0],
        utc=True,
    )

    return RankedPredictionPanel(
        timestamps=pd.DatetimeIndex(
            timestamps
        ),
        symbols=symbols,
        prediction_ranks=rank_rows(
            predictions
        ),
        target_ranks=rank_rows(targets),
    )
