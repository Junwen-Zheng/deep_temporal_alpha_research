from __future__ import annotations

import numpy as np
import pandas as pd

from deep_alpha.data.features import (
    FEATURE_COLUMNS,
    add_forward_targets,
    build_symbol_features,
    finalize_research_panel,
)


def sample_bars(
    symbol: str,
    periods: int = 100,
) -> pd.DataFrame:
    timestamp = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=periods,
        freq="5min",
    )

    trend = np.arange(periods, dtype=float)

    close = 100.0 + trend * 0.1

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "available_time": (
                timestamp
                + pd.Timedelta(minutes=5)
                - pd.Timedelta(milliseconds=1)
            ),
            "symbol": symbol,
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 100.0 + trend,
            "quote_volume": (
                (100.0 + trend) * close
            ),
            "trade_count": (
                50 + np.arange(periods)
            ),
            "taker_buy_volume": (
                50.0 + trend * 0.4
            ),
            "taker_buy_quote_volume": (
                (50.0 + trend * 0.4) * close
            ),
        }
    )


def test_features_do_not_depend_on_future_rows() -> None:
    original = sample_bars("BTCUSDT")
    modified = original.copy()

    modified.loc[
        modified.index > 70,
        [
            "open",
            "high",
            "low",
            "close",
            "quote_volume",
            "trade_count",
            "taker_buy_quote_volume",
        ],
    ] *= 10.0

    original_features = build_symbol_features(original)
    modified_features = build_symbol_features(modified)

    original_row = original_features.loc[
        70,
        FEATURE_COLUMNS,
    ].to_numpy(dtype=np.float64)

    modified_row = modified_features.loc[
        70,
        FEATURE_COLUMNS,
    ].to_numpy(dtype=np.float64)

    np.testing.assert_allclose(
        original_row,
        modified_row,
        rtol=0.0,
        atol=0.0,
    )


def test_forward_target_uses_configured_horizon() -> None:
    first = sample_bars("A", periods=80)
    second = sample_bars("B", periods=80)

    second["close"] = 200.0

    panel = pd.concat(
        [
            build_symbol_features(first),
            build_symbol_features(second),
        ],
        ignore_index=True,
    )

    result = add_forward_targets(
        panel,
        horizon_bars=12,
        interval_minutes=5,
    )

    row = result[
        (result["symbol"] == "A")
        & (
            result["timestamp"]
            == pd.Timestamp("2026-01-01T00:00:00Z")
        )
    ].iloc[0]

    expected_raw = np.log(
        first.loc[12, "close"]
        / first.loc[0, "close"]
    )

    assert np.isclose(row["raw_target"], expected_raw)

    assert row["label_end_timestamp"] == pd.Timestamp(
        "2026-01-01T01:00:00Z"
    )


def test_cross_sectional_target_mean_is_zero() -> None:
    first = build_symbol_features(
        sample_bars("A", periods=100)
    )
    second = build_symbol_features(
        sample_bars("B", periods=100)
    )

    second["close"] *= 1.01

    panel = add_forward_targets(
        pd.concat([first, second], ignore_index=True),
        horizon_bars=12,
        interval_minutes=5,
    )

    finalized = finalize_research_panel(
        panel,
        expected_symbol_count=2,
    )

    target_means = finalized.groupby("timestamp")[
        "target"
    ].mean()

    assert target_means.abs().max() < 1e-12


def test_first_valid_row_reflects_longest_window() -> None:
    features = build_symbol_features(
        sample_bars("BTCUSDT", periods=100)
    )

    assert features.loc[:47, "return_48"].isna().all()
    assert np.isfinite(features.loc[48, "return_48"])


def test_zero_volume_bar_uses_neutral_taker_ratio() -> None:
    bars = sample_bars("BTCUSDT")

    bars.loc[70, "quote_volume"] = 0.0
    bars.loc[70, "taker_buy_quote_volume"] = 0.0

    features = build_symbol_features(bars)

    assert features.loc[70, "taker_buy_ratio"] == 0.5
