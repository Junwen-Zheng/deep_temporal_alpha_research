from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from deep_alpha.config import load_yaml
from deep_alpha.data.download import iter_months, sha256_file

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

FLOAT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]

OUTPUT_COLUMNS = [
    "timestamp",
    "available_time",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]


@dataclass(frozen=True)
class ValidationSummary:
    duplicate_rows: int
    missing_intervals: int


@dataclass(frozen=True)
class NormalizedRecord:
    symbol: str
    interval: str
    source_files: list[str]
    source_file_count: int
    rows: int
    start_timestamp: str
    end_timestamp: str
    duplicate_rows: int
    missing_intervals: int
    output_path: str
    output_bytes: int
    output_sha256: str


def _epoch_unit(values: pd.Series) -> str:
    median = float(values.median())

    if median >= 100_000_000_000_000:
        return "us"

    return "ms"


def _drop_header_row(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    numeric_open_time = pd.to_numeric(
        frame["open_time"],
        errors="coerce",
    )
    invalid_mask = numeric_open_time.isna()

    if not invalid_mask.any():
        return frame

    invalid_indices = frame.index[invalid_mask].tolist()

    first_value = str(frame.loc[0, "open_time"])
    normalized_first_value = (
        first_value.lstrip("\ufeff").strip().lower()
    )

    if invalid_indices != [0] or normalized_first_value != "open_time":
        raise ValueError(
            f"Unexpected nonnumeric open_time rows in {path}: "
            f"{invalid_indices}"
        )

    return frame.loc[~invalid_mask].reset_index(drop=True)


def read_kline_csv(path: Path, symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=KLINE_COLUMNS,
        dtype=str,
    )

    frame = _drop_header_row(frame, path)

    if frame.empty:
        raise ValueError(f"No data rows found in {path}")

    for column in [
        "open_time",
        "close_time",
        "trade_count",
        *FLOAT_COLUMNS,
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    frame["open_time"] = frame["open_time"].astype("int64")
    frame["close_time"] = frame["close_time"].astype("int64")
    frame["trade_count"] = frame["trade_count"].astype("int64")

    for column in FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float64")

    open_time_unit = _epoch_unit(frame["open_time"])
    close_time_unit = _epoch_unit(frame["close_time"])

    if open_time_unit != close_time_unit:
        raise ValueError(
            f"Timestamp units differ in {path}: "
            f"{open_time_unit} and {close_time_unit}"
        )

    frame["timestamp"] = pd.to_datetime(
        frame["open_time"],
        unit=open_time_unit,
        utc=True,
    )

    frame["available_time"] = pd.to_datetime(
        frame["close_time"],
        unit=close_time_unit,
        utc=True,
    )

    frame["symbol"] = symbol

    return frame[OUTPUT_COLUMNS]


def validate_bars(
    frame: pd.DataFrame,
    symbol: str,
    interval_minutes: int,
) -> ValidationSummary:
    if frame.empty:
        raise ValueError(f"No normalized rows found for {symbol}")

    missing_columns = set(OUTPUT_COLUMNS) - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing normalized columns: {sorted(missing_columns)}"
        )

    if frame[OUTPUT_COLUMNS].isna().any().any():
        raise ValueError(f"Null values found for {symbol}")

    symbols = set(frame["symbol"].unique())

    if symbols != {symbol}:
        raise ValueError(
            f"Unexpected symbols for {symbol}: {sorted(symbols)}"
        )

    duplicate_rows = int(
        frame.duplicated(["symbol", "timestamp"]).sum()
    )

    if duplicate_rows:
        raise ValueError(
            f"Duplicate timestamp rows for {symbol}: "
            f"{duplicate_rows}"
        )

    numeric_values = frame[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ]
    ].to_numpy()

    if not np.isfinite(numeric_values).all():
        raise ValueError(f"Non-finite numeric values found for {symbol}")

    if (
        frame[["open", "high", "low", "close"]] <= 0
    ).any().any():
        raise ValueError(f"Non-positive prices found for {symbol}")

    if (
        frame[
            [
                "volume",
                "quote_volume",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "trade_count",
            ]
        ]
        < 0
    ).any().any():
        raise ValueError(f"Negative activity values found for {symbol}")

    if (
        frame["high"]
        < frame[["open", "close", "low"]].max(axis=1)
    ).any():
        raise ValueError(f"Invalid high prices found for {symbol}")

    if (
        frame["low"]
        > frame[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError(f"Invalid low prices found for {symbol}")

    if (frame["available_time"] < frame["timestamp"]).any():
        raise ValueError(
            f"available_time precedes timestamp for {symbol}"
        )

    expected_interval = pd.Timedelta(minutes=interval_minutes)

    if (
        frame["available_time"] - frame["timestamp"]
        > expected_interval
    ).any():
        raise ValueError(
            f"Bar duration exceeds configured interval for {symbol}"
        )

    timestamps = frame["timestamp"].array.asi8
    differences = np.diff(timestamps)
    expected_nanoseconds = expected_interval.value

    if (differences <= 0).any():
        raise ValueError(
            f"Timestamps are not strictly increasing for {symbol}"
        )

    if (differences % expected_nanoseconds != 0).any():
        raise ValueError(
            f"Timestamp spacing is not aligned for {symbol}"
        )

    missing_intervals = int(
        ((differences // expected_nanoseconds) - 1).sum()
    )

    return ValidationSummary(
        duplicate_rows=duplicate_rows,
        missing_intervals=missing_intervals,
    )


def write_parquet(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    table = pa.Table.from_pandas(
        frame,
        preserve_index=False,
    )

    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=["symbol"],
            write_statistics=True,
        )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def normalize_symbol(
    symbol: str,
    interval: str,
    raw_directory: Path,
    processed_directory: Path,
    expected_months: set[str],
) -> NormalizedRecord:
    source_directory = (
        raw_directory
        / "futures_um"
        / "monthly"
        / "klines"
        / symbol
        / interval
        / "extracted"
    )

    source_paths = sorted(source_directory.glob("*/*.csv"))

    found_months = {path.parent.name for path in source_paths}

    if found_months != expected_months:
        missing = sorted(expected_months - found_months)
        unexpected = sorted(found_months - expected_months)

        raise ValueError(
            f"Source months differ for {symbol}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    frames = [
        read_kline_csv(path=path, symbol=symbol)
        for path in source_paths
    ]

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    interval_minutes = int(interval.removesuffix("m"))

    summary = validate_bars(
        frame=frame,
        symbol=symbol,
        interval_minutes=interval_minutes,
    )

    destination = (
        processed_directory
        / "bars"
        / f"symbol={symbol}"
        / "part-00000.parquet"
    )

    write_parquet(frame, destination)

    return NormalizedRecord(
        symbol=symbol,
        interval=interval,
        source_files=[str(path) for path in source_paths],
        source_file_count=len(source_paths),
        rows=len(frame),
        start_timestamp=frame["timestamp"].iloc[0].isoformat(),
        end_timestamp=frame["timestamp"].iloc[-1].isoformat(),
        duplicate_rows=summary.duplicate_rows,
        missing_intervals=summary.missing_intervals,
        output_path=str(destination),
        output_bytes=destination.stat().st_size,
        output_sha256=sha256_file(destination),
    )


def write_normalized_manifest(
    records: list[NormalizedRecord],
    destination: Path,
) -> None:
    ordered = sorted(records, key=lambda record: record.symbol)

    payload = {
        "schema_version": 1,
        "record_count": len(ordered),
        "total_rows": sum(record.rows for record in ordered),
        "records": [asdict(record) for record in ordered],
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def normalize_dataset(
    config_path: str | Path,
    manifest_path: str | Path | None = None,
    symbol_filter: str | None = None,
) -> list[NormalizedRecord]:
    config = load_yaml(config_path)
    data_config = config["data"]

    symbols = list(data_config["symbols"])

    if symbol_filter is not None:
        if symbol_filter not in symbols:
            raise ValueError(
                f"Symbol is not configured: {symbol_filter}"
            )

        symbols = [symbol_filter]

    expected_months = set(
        iter_months(
            data_config["start_month"],
            data_config["end_month"],
        )
    )

    raw_directory = Path(data_config["raw_dir"])
    processed_directory = Path(data_config["processed_dir"])
    interval = str(data_config["interval"])

    destination = Path(
        manifest_path
        or data_config["normalized_manifest_path"]
    )

    records: list[NormalizedRecord] = []

    for index, symbol in enumerate(symbols, start=1):
        print(
            f"[{index:02d}/{len(symbols):02d}] "
            f"Normalizing {symbol}",
            flush=True,
        )

        record = normalize_symbol(
            symbol=symbol,
            interval=interval,
            raw_directory=raw_directory,
            processed_directory=processed_directory,
            expected_months=expected_months,
        )

        records.append(record)
        write_normalized_manifest(records, destination)

    return records
