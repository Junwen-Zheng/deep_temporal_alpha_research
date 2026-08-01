from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from deep_alpha.data.normalize import (
    read_kline_csv,
    validate_bars,
)


def write_sample_csv(
    path: Path,
    include_header: bool = True,
) -> None:
    header = (
        "open_time,open,high,low,close,volume,close_time,"
        "quote_volume,trade_count,taker_buy_volume,"
        "taker_buy_quote_volume,ignore\n"
    )

    rows = [
        (
            "1735689600000,100,102,99,101,10,"
            "1735689899999,1000,5,6,600,0\n"
        ),
        (
            "1735689900000,101,103,100,102,11,"
            "1735690199999,1100,6,7,700,0\n"
        ),
        (
            "1735690200000,102,104,101,103,12,"
            "1735690499999,1200,7,8,800,0\n"
        ),
    ]

    contents = "".join(rows)

    if include_header:
        contents = header + contents

    path.write_text(contents, encoding="utf-8")


def test_csv_header_is_removed(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    write_sample_csv(path)

    frame = read_kline_csv(path, symbol="BTCUSDT")

    assert len(frame) == 3
    assert frame.index.tolist() == [0, 1, 2]
    assert list(frame["symbol"].unique()) == ["BTCUSDT"]
    assert str(frame["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert frame["trade_count"].dtype == "int64"


def test_csv_without_header_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    write_sample_csv(path, include_header=False)

    frame = read_kline_csv(path, symbol="BTCUSDT")

    assert len(frame) == 3


def test_unexpected_nonnumeric_row_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.csv"
    write_sample_csv(path)

    with path.open("a", encoding="utf-8") as handle:
        handle.write("invalid,row\n")

    with pytest.raises(
        ValueError,
        match="Unexpected nonnumeric open_time",
    ):
        read_kline_csv(path, symbol="BTCUSDT")


def test_valid_bars_have_no_missing_intervals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.csv"
    write_sample_csv(path)

    frame = read_kline_csv(path, symbol="BTCUSDT")

    summary = validate_bars(
        frame=frame,
        symbol="BTCUSDT",
        interval_minutes=5,
    )

    assert summary.duplicate_rows == 0
    assert summary.missing_intervals == 0


def test_duplicate_timestamp_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.csv"
    write_sample_csv(path)

    frame = read_kline_csv(path, symbol="BTCUSDT")
    frame = pd.concat(
        [frame, frame.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate timestamp rows",
    ):
        validate_bars(
            frame=frame,
            symbol="BTCUSDT",
            interval_minutes=5,
        )


def test_invalid_high_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    write_sample_csv(path)

    frame = read_kline_csv(path, symbol="BTCUSDT")
    frame.loc[0, "high"] = 98.0

    with pytest.raises(
        ValueError,
        match="Invalid high prices",
    ):
        validate_bars(
            frame=frame,
            symbol="BTCUSDT",
            interval_minutes=5,
        )
