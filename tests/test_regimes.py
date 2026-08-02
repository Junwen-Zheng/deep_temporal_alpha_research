from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from deep_alpha.evaluation.regimes import (
    assign_regime_labels,
    build_fold_regime_assignments,
    build_market_state_panel,
    build_regime_model_summary,
    melt_regime_assignments,
    validate_regime_quantiles,
)


def sample_market_bars() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=5,
        freq="5min",
    )

    rows = []

    closes = {
        "A": [
            100.0,
            101.0,
            102.0,
            104.0,
            108.0,
        ],
        "B": [
            100.0,
            99.0,
            98.0,
            97.0,
            96.0,
        ],
    }

    for symbol, values in (
        closes.items()
    ):
        for timestamp, close in zip(
            timestamps,
            values,
            strict=True,
        ):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "close": close,
                }
            )

    return pd.DataFrame(rows)


def test_market_state_panel_is_causal() -> None:
    result = build_market_state_panel(
        bars=sample_market_bars(),
        window_bars=2,
        expected_symbol_count=2,
    )

    assert len(result) == 3

    expected_market_return = np.mean(
        [
            np.log(102.0 / 100.0),
            np.log(98.0 / 100.0),
        ]
    )

    assert np.isclose(
        result.loc[
            0,
            "market_return",
        ],
        expected_market_return,
    )

    assert (
        result.loc[
            0,
            "timestamp",
        ]
        == pd.Timestamp(
            "2026-01-01T00:10:00Z"
        )
    )


def test_regime_assignment_uses_boundaries() -> None:
    result = assign_regime_labels(
        values=np.asarray(
            [
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
            ]
        ),
        lower_threshold=-1.0,
        upper_threshold=1.0,
        labels=(
            "low",
            "medium",
            "high",
        ),
    )

    assert result.tolist() == [
        "low",
        "low",
        "medium",
        "medium",
        "high",
    ]


def test_fold_thresholds_use_pretest_data() -> None:
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=12,
        freq="1h",
    )

    states = pd.DataFrame(
        {
            "timestamp": timestamps,
            "market_return": np.arange(
                12,
                dtype=np.float64,
            ),
            "market_realized_volatility": (
                np.arange(
                    12,
                    dtype=np.float64,
                )
                + 10.0
            ),
            "cross_sectional_dispersion": (
                np.arange(
                    12,
                    dtype=np.float64,
                )
                + 20.0
            ),
        }
    )

    fold = SimpleNamespace(
        fold_id=1,
        validation_end=timestamps[7],
        test_end=timestamps[11],
    )

    assignments, thresholds = (
        build_fold_regime_assignments(
            market_states=states,
            fold=fold,
            test_timestamps=timestamps[
                8:
            ],
            lower_quantile=1.0 / 3.0,
            upper_quantile=2.0 / 3.0,
        )
    )

    assert len(assignments) == 4

    expected_lower = np.quantile(
        np.arange(
            8,
            dtype=np.float64,
        ),
        1.0 / 3.0,
    )

    assert np.isclose(
        thresholds[
            "market_return_lower"
        ],
        expected_lower,
    )

    assert (
        thresholds[
            "calibration_end_timestamp"
        ]
        == timestamps[7]
    )


def test_melt_regimes_creates_three_dimensions() -> None:
    frame = pd.DataFrame(
        {
            "fold": [
                1,
                1,
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:05:00Z",
                ]
            ),
            "direction_regime": [
                "down",
                "up",
            ],
            "volatility_regime": [
                "low",
                "high",
            ],
            "dispersion_regime": [
                "medium",
                "high",
            ],
        }
    )

    result = melt_regime_assignments(
        frame
    )

    assert len(result) == 6

    assert set(
        result["regime_dimension"]
    ) == {
        "direction",
        "volatility",
        "dispersion",
    }


def test_regime_summary_uses_focus_cost() -> None:
    prediction_rows = []
    portfolio_rows = []

    for fold in [
        1,
        2,
        3,
    ]:
        for regime in [
            "low",
            "medium",
            "high",
        ]:
            prediction_rows.append(
                {
                    "model": "example",
                    "fold": fold,
                    "regime_dimension": (
                        "volatility"
                    ),
                    "regime": regime,
                    "mean_rank_ic": (
                        0.01
                        if regime != "high"
                        else -0.01
                    ),
                }
            )

            for cost in [
                0.0,
                1.0,
            ]:
                portfolio_rows.append(
                    {
                        "model": "example",
                        "fold": fold,
                        "regime_dimension": (
                            "volatility"
                        ),
                        "regime": regime,
                        "cost_bps": cost,
                        "break_even_cost_bps": (
                            0.5
                        ),
                        "pooled_annualized_net_return": (
                            1.0
                            if cost == 0.0
                            else -1.0
                        ),
                    }
                )

    result = build_regime_model_summary(
        prediction_metrics=pd.DataFrame(
            prediction_rows
        ),
        portfolio_metrics=pd.DataFrame(
            portfolio_rows
        ),
        focus_cost_bps=1.0,
    )

    assert len(result) == 1

    assert np.isclose(
        result.loc[
            0,
            "positive_rank_ic_cell_fraction",
        ],
        2.0 / 3.0,
    )

    assert np.isclose(
        result.loc[
            0,
            "positive_cell_fraction_at_focus_cost",
        ],
        0.0,
    )


def test_invalid_regime_quantiles_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="0 < lower",
    ):
        validate_regime_quantiles(
            lower_quantile=0.8,
            upper_quantile=0.2,
        )
