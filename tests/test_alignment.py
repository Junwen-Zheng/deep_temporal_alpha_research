from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_alpha.evaluation.alignment import (
    assert_key_alignment,
    build_bar_alignment_reference,
    cross_sectionally_demean,
    maximum_absolute_error,
    maximum_timestamp_error_nanoseconds,
)


def sample_bars() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=6,
        freq="5min",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
            ],
            "close": [
                10.5,
                11.5,
                12.5,
                13.5,
                14.5,
                15.5,
            ],
        }
    )


def test_raw_label_reconstruction() -> None:
    bars = sample_bars()

    reference = (
        build_bar_alignment_reference(
            bars=bars,
            horizon_bars=2,
            entry_offset_bars=1,
            exit_offset_bars=2,
            interval_minutes=5,
        )
    )

    expected = np.log(
        bars.loc[2, "close"]
        / bars.loc[0, "close"]
    )

    assert np.isclose(
        reference.loc[
            0,
            "expected_raw_target",
        ],
        expected,
    )

    assert (
        reference.loc[
            0,
            "expected_label_end_timestamp",
        ]
        == bars.loc[2, "timestamp"]
    )


def test_execution_uses_next_open() -> None:
    bars = sample_bars()

    reference = (
        build_bar_alignment_reference(
            bars=bars,
            horizon_bars=2,
            entry_offset_bars=1,
            exit_offset_bars=2,
            interval_minutes=5,
        )
    )

    expected = np.log(
        bars.loc[2, "close"]
        / bars.loc[1, "open"]
    )

    assert np.isclose(
        reference.loc[
            0,
            "expected_execution_return",
        ],
        expected,
    )


def test_cross_sectional_demeaning() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                ]
            ),
            "raw_target": [
                1.0,
                2.0,
                6.0,
            ],
        }
    )

    result = cross_sectionally_demean(
        frame=frame,
        value_column="raw_target",
    )

    np.testing.assert_allclose(
        result.mean(),
        0.0,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        result,
        np.asarray(
            [-2.0, -1.0, 3.0]
        ),
    )


def test_timestamp_error_is_exact() -> None:
    actual = pd.Series(
        pd.date_range(
            "2026-01-01T00:00:00Z",
            periods=3,
            freq="5min",
        )
    )

    expected = actual.copy()

    assert (
        maximum_timestamp_error_nanoseconds(
            actual,
            expected,
        )
        == 0
    )


def test_numeric_shape_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="shapes differ",
    ):
        maximum_absolute_error(
            np.asarray([1.0, 2.0]),
            np.asarray([1.0]),
        )


def test_key_mismatch_is_rejected() -> None:
    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                ]
            ),
            "symbol": ["A"],
        }
    )

    actual = expected.copy()
    actual["symbol"] = "B"

    with pytest.raises(
        ValueError,
        match="symbols differ",
    ):
        assert_key_alignment(
            expected=expected,
            actual=actual,
        )
