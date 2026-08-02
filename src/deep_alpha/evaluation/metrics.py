from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_PREDICTION_COLUMNS = {
    "timestamp",
    "symbol",
    "prediction",
    "target",
}


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    missing_columns = (
        REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing prediction columns: "
            f"{sorted(missing_columns)}"
        )

    if frame.empty:
        raise ValueError("Prediction frame is empty")

    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError(
            "Duplicate timestamp/symbol prediction rows"
        )

    numeric_values = frame[
        ["prediction", "target"]
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Predictions or targets contain non-finite values"
        )


def rank_ic_series(frame: pd.DataFrame) -> pd.Series:
    validate_prediction_frame(frame)

    timestamp = frame["timestamp"]

    prediction_rank = frame.groupby(
        "timestamp",
        sort=False,
    )["prediction"].rank(method="average")

    target_rank = frame.groupby(
        "timestamp",
        sort=False,
    )["target"].rank(method="average")

    prediction_mean = prediction_rank.groupby(
        timestamp,
        sort=False,
    ).transform("mean")

    target_mean = target_rank.groupby(
        timestamp,
        sort=False,
    ).transform("mean")

    prediction_centered = (
        prediction_rank - prediction_mean
    )

    target_centered = target_rank - target_mean

    numerator = (
        prediction_centered * target_centered
    ).groupby(
        timestamp,
        sort=False,
    ).sum()

    prediction_sum_of_squares = (
        prediction_centered.pow(2)
        .groupby(timestamp, sort=False)
        .sum()
    )

    target_sum_of_squares = (
        target_centered.pow(2)
        .groupby(timestamp, sort=False)
        .sum()
    )

    denominator = np.sqrt(
        prediction_sum_of_squares
        * target_sum_of_squares
    )

    rank_ic = numerator / denominator.replace(
        0.0,
        np.nan,
    )

    rank_ic.name = "rank_ic"
    return rank_ic


def summarize_predictions(
    frame: pd.DataFrame,
) -> dict[str, float | int]:
    validate_prediction_frame(frame)

    rank_ic = rank_ic_series(frame).dropna()

    if rank_ic.empty:
        raise ValueError(
            "No valid cross-sectional Rank IC observations"
        )

    errors = (
        frame["prediction"].to_numpy(dtype=np.float64)
        - frame["target"].to_numpy(dtype=np.float64)
    )

    rank_ic_standard_deviation = float(
        rank_ic.std(ddof=1)
    )

    rank_ic_ir = np.nan

    if rank_ic_standard_deviation > 0:
        rank_ic_ir = float(
            rank_ic.mean()
            / rank_ic_standard_deviation
        )

    return {
        "mean_rank_ic": float(rank_ic.mean()),
        "median_rank_ic": float(rank_ic.median()),
        "rank_ic_std": rank_ic_standard_deviation,
        "rank_ic_ir": rank_ic_ir,
        "positive_timestamp_fraction": float(
            (rank_ic > 0).mean()
        ),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "n_rows": len(frame),
        "n_timestamps": int(
            frame["timestamp"].nunique()
        ),
    }
