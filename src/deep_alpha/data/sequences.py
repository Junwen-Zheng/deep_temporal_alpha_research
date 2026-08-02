from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TemporalPanel:
    timestamps: pd.DatetimeIndex
    symbols: tuple[str, ...]
    features: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray

    @property
    def timestamp_count(self) -> int:
        return len(self.timestamps)

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)

    @property
    def feature_count(self) -> int:
        return int(self.features.shape[2])


@dataclass(frozen=True)
class SequenceSplit:
    split: str
    endpoint_positions: np.ndarray
    timestamp_count: int
    row_count: int
    excluded_history_timestamps: int
    excluded_history_rows: int
    first_endpoint_timestamp: str
    last_endpoint_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("endpoint_positions")
        return payload


def build_temporal_panel(
    frame: pd.DataFrame,
    feature_columns: list[str],
    expected_symbol_count: int,
    interval_minutes: int,
) -> TemporalPanel:
    required_columns = {
        "timestamp",
        "symbol",
        "target",
        "raw_target",
        *feature_columns,
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing temporal-panel columns: "
            f"{sorted(missing_columns)}"
        )

    if frame.empty:
        raise ValueError("Temporal panel source is empty")

    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError(
            "Duplicate timestamp/symbol rows in temporal panel"
        )

    ordered = (
        frame.sort_values(
            ["timestamp", "symbol"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    symbols = tuple(
        sorted(
            str(symbol)
            for symbol in ordered["symbol"].unique()
        )
    )

    if len(symbols) != expected_symbol_count:
        raise ValueError(
            f"Expected {expected_symbol_count} symbols, "
            f"found {len(symbols)}"
        )

    rows_per_timestamp = ordered.groupby(
        "timestamp",
        sort=False,
    ).size()

    if not rows_per_timestamp.eq(expected_symbol_count).all():
        invalid = rows_per_timestamp[
            rows_per_timestamp != expected_symbol_count
        ]

        raise ValueError(
            "Incomplete timestamp cross-sections: "
            f"{invalid.head().to_dict()}"
        )

    timestamp_count = len(rows_per_timestamp)

    if len(ordered) != timestamp_count * expected_symbol_count:
        raise ValueError(
            "Temporal panel dimensions are inconsistent"
        )

    symbol_matrix = ordered["symbol"].to_numpy().reshape(
        timestamp_count,
        expected_symbol_count,
    )

    expected_symbol_row = np.asarray(
        symbols,
        dtype=object,
    )

    if not np.all(
        symbol_matrix == expected_symbol_row[None, :]
    ):
        raise ValueError(
            "Symbol ordering differs across timestamps"
        )

    timestamp_nanoseconds = (
        ordered["timestamp"]
        .array.asi8
        .reshape(
            timestamp_count,
            expected_symbol_count,
        )
    )

    if not np.all(
        timestamp_nanoseconds
        == timestamp_nanoseconds[:, [0]]
    ):
        raise ValueError(
            "Timestamp values differ within a cross-section"
        )

    unique_timestamp_nanoseconds = timestamp_nanoseconds[:, 0]

    expected_interval_nanoseconds = pd.Timedelta(
        minutes=interval_minutes
    ).value

    differences = np.diff(unique_timestamp_nanoseconds)

    if not np.all(
        differences == expected_interval_nanoseconds
    ):
        raise ValueError(
            "Temporal panel timestamps are not contiguous"
        )

    features = ordered[
        feature_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    ).reshape(
        timestamp_count,
        expected_symbol_count,
        len(feature_columns),
    )

    targets = ordered["target"].to_numpy(
        dtype=np.float32,
        copy=True,
    ).reshape(
        timestamp_count,
        expected_symbol_count,
    )

    raw_targets = ordered["raw_target"].to_numpy(
        dtype=np.float32,
        copy=True,
    ).reshape(
        timestamp_count,
        expected_symbol_count,
    )

    if not np.isfinite(features).all():
        raise ValueError(
            "Temporal features contain non-finite values"
        )

    if not np.isfinite(targets).all():
        raise ValueError(
            "Temporal targets contain non-finite values"
        )

    if not np.isfinite(raw_targets).all():
        raise ValueError(
            "Temporal raw targets contain non-finite values"
        )

    timestamps = pd.to_datetime(
        unique_timestamp_nanoseconds,
        utc=True,
    )

    return TemporalPanel(
        timestamps=pd.DatetimeIndex(timestamps),
        symbols=symbols,
        features=features,
        targets=targets,
        raw_targets=raw_targets,
    )


def build_sequence_split(
    frame: pd.DataFrame,
    row_mask: pd.Series,
    temporal_panel: TemporalPanel,
    sequence_length: int,
    split_name: str,
) -> SequenceSplit:
    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    if len(row_mask) != len(frame):
        raise ValueError(
            "Row mask length does not match frame"
        )

    ordered = frame.sort_values(
        ["timestamp", "symbol"],
        kind="mergesort",
    )

    aligned_mask = row_mask.reindex(ordered.index)

    if aligned_mask.isna().any():
        raise ValueError(
            "Row mask could not be aligned to the frame"
        )

    mask_matrix = aligned_mask.to_numpy(
        dtype=bool,
    ).reshape(
        temporal_panel.timestamp_count,
        temporal_panel.symbol_count,
    )

    if not np.all(
        mask_matrix == mask_matrix[:, [0]]
    ):
        raise ValueError(
            f"{split_name} mask is not uniform "
            "across the symbol cross-section"
        )

    timestamp_mask = mask_matrix[:, 0]

    positions = np.arange(
        temporal_panel.timestamp_count,
        dtype=np.int64,
    )

    sufficient_history = positions >= (
        sequence_length - 1
    )

    excluded_history_mask = (
        timestamp_mask & ~sufficient_history
    )

    endpoint_positions = positions[
        timestamp_mask & sufficient_history
    ]

    if len(endpoint_positions) == 0:
        raise ValueError(
            f"No usable sequence endpoints for {split_name}"
        )

    first_position = int(endpoint_positions[0])
    last_position = int(endpoint_positions[-1])

    return SequenceSplit(
        split=split_name,
        endpoint_positions=endpoint_positions,
        timestamp_count=len(endpoint_positions),
        row_count=(
            len(endpoint_positions)
            * temporal_panel.symbol_count
        ),
        excluded_history_timestamps=int(
            excluded_history_mask.sum()
        ),
        excluded_history_rows=int(
            excluded_history_mask.sum()
            * temporal_panel.symbol_count
        ),
        first_endpoint_timestamp=(
            temporal_panel.timestamps[
                first_position
            ].isoformat()
        ),
        last_endpoint_timestamp=(
            temporal_panel.timestamps[
                last_position
            ].isoformat()
        ),
    )


class CrossSectionalSequenceDataset(Dataset):
    def __init__(
        self,
        temporal_panel: TemporalPanel,
        endpoint_positions: np.ndarray,
        sequence_length: int,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

        endpoints = np.asarray(
            endpoint_positions,
            dtype=np.int64,
        )

        if endpoints.ndim != 1:
            raise ValueError(
                "endpoint_positions must be one-dimensional"
            )

        if len(endpoints) == 0:
            raise ValueError(
                "At least one endpoint is required"
            )

        if np.any(endpoints < sequence_length - 1):
            raise ValueError(
                "A sequence endpoint lacks sufficient history"
            )

        if np.any(
            endpoints >= temporal_panel.timestamp_count
        ):
            raise ValueError(
                "A sequence endpoint exceeds the panel"
            )

        if np.any(np.diff(endpoints) <= 0):
            raise ValueError(
                "Sequence endpoints must be strictly increasing"
            )

        self.temporal_panel = temporal_panel
        self.endpoint_positions = endpoints
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return (
            len(self.endpoint_positions)
            * self.temporal_panel.symbol_count
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        int,
        int,
    ]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        endpoint_offset = (
            index // self.temporal_panel.symbol_count
        )

        symbol_index = (
            index % self.temporal_panel.symbol_count
        )

        endpoint_position = int(
            self.endpoint_positions[endpoint_offset]
        )

        start_position = (
            endpoint_position
            - self.sequence_length
            + 1
        )

        sequence = self.temporal_panel.features[
            start_position : endpoint_position + 1,
            symbol_index,
            :,
        ]

        if len(sequence) != self.sequence_length:
            raise RuntimeError(
                "Constructed sequence has incorrect length"
            )

        contiguous_sequence = np.ascontiguousarray(
            sequence,
            dtype=np.float32,
        )

        target = np.float32(
            self.temporal_panel.targets[
                endpoint_position,
                symbol_index,
            ]
        )

        return (
            torch.from_numpy(contiguous_sequence),
            torch.tensor(target, dtype=torch.float32),
            endpoint_position,
            symbol_index,
        )
