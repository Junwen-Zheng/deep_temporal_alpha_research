from __future__ import annotations

import numpy as np
import pandas as pd

from deep_alpha.evaluation.metrics import (
    rank_ic_series,
    summarize_predictions,
)


def prediction_frame(
    reverse: bool = False,
) -> pd.DataFrame:
    rows = []

    for timestamp in pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=3,
        freq="5min",
    ):
        targets = [-1.0, 0.0, 1.0]

        predictions = (
            [1.0, 0.0, -1.0]
            if reverse
            else targets
        )

        for symbol, target, prediction in zip(
            ["A", "B", "C"],
            targets,
            predictions,
            strict=True,
        ):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "target": target,
                    "prediction": prediction,
                }
            )

    return pd.DataFrame(rows)


def test_perfect_rank_ic_is_one() -> None:
    rank_ic = rank_ic_series(
        prediction_frame()
    )

    np.testing.assert_allclose(
        rank_ic.to_numpy(),
        np.ones(3),
    )


def test_reverse_rank_ic_is_negative_one() -> None:
    rank_ic = rank_ic_series(
        prediction_frame(reverse=True)
    )

    np.testing.assert_allclose(
        rank_ic.to_numpy(),
        -np.ones(3),
    )


def test_prediction_summary_counts_rows() -> None:
    summary = summarize_predictions(
        prediction_frame()
    )

    assert summary["mean_rank_ic"] == 1.0
    assert summary["n_rows"] == 9
    assert summary["n_timestamps"] == 3
