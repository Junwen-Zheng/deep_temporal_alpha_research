from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FoldDefinition:
    fold_id: int
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class FoldSummary:
    fold_id: int
    train_end: str
    validation_end: str
    test_end: str
    train_rows: int
    validation_rows: int
    test_rows: int
    purged_train_boundary_rows: int
    purged_validation_boundary_rows: int
    purged_test_boundary_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FoldMasks:
    train: pd.Series
    validation: pd.Series
    test: pd.Series


def _parse_utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        raise ValueError(
            f"Split timestamp must include a timezone: {value}"
        )

    return timestamp.tz_convert("UTC")


def parse_fold_definition(
    configuration: dict[str, Any],
) -> FoldDefinition:
    fold = FoldDefinition(
        fold_id=int(configuration["fold_id"]),
        train_end=_parse_utc_timestamp(
            str(configuration["train_end"])
        ),
        validation_end=_parse_utc_timestamp(
            str(configuration["validation_end"])
        ),
        test_end=_parse_utc_timestamp(
            str(configuration["test_end"])
        ),
    )

    if not (
        fold.train_end
        < fold.validation_end
        < fold.test_end
    ):
        raise ValueError(
            f"Invalid boundary ordering for fold {fold.fold_id}"
        )

    return fold


def build_fold_masks(
    frame: pd.DataFrame,
    fold: FoldDefinition,
) -> FoldMasks:
    timestamp = frame["timestamp"]
    label_end_timestamp = frame["label_end_timestamp"]

    train = (
        timestamp.le(fold.train_end)
        & label_end_timestamp.le(fold.train_end)
    )

    validation = (
        timestamp.gt(fold.train_end)
        & timestamp.le(fold.validation_end)
        & label_end_timestamp.le(fold.validation_end)
    )

    test = (
        timestamp.gt(fold.validation_end)
        & timestamp.le(fold.test_end)
        & label_end_timestamp.le(fold.test_end)
    )

    if (train & validation).any():
        raise ValueError("Train and validation masks overlap")

    if (train & test).any():
        raise ValueError("Train and test masks overlap")

    if (validation & test).any():
        raise ValueError("Validation and test masks overlap")

    return FoldMasks(
        train=train,
        validation=validation,
        test=test,
    )


def summarize_fold(
    frame: pd.DataFrame,
    fold: FoldDefinition,
) -> FoldSummary:
    masks = build_fold_masks(frame, fold)

    timestamp = frame["timestamp"]
    label_end_timestamp = frame["label_end_timestamp"]

    unpurged_train = timestamp.le(fold.train_end)

    unpurged_validation = (
        timestamp.gt(fold.train_end)
        & timestamp.le(fold.validation_end)
    )

    unpurged_test = (
        timestamp.gt(fold.validation_end)
        & timestamp.le(fold.test_end)
    )

    return FoldSummary(
        fold_id=fold.fold_id,
        train_end=fold.train_end.isoformat(),
        validation_end=fold.validation_end.isoformat(),
        test_end=fold.test_end.isoformat(),
        train_rows=int(masks.train.sum()),
        validation_rows=int(masks.validation.sum()),
        test_rows=int(masks.test.sum()),
        purged_train_boundary_rows=int(
            (
                unpurged_train
                & label_end_timestamp.gt(fold.train_end)
            ).sum()
        ),
        purged_validation_boundary_rows=int(
            (
                unpurged_validation
                & label_end_timestamp.gt(
                    fold.validation_end
                )
            ).sum()
        ),
        purged_test_boundary_rows=int(
            (
                unpurged_test
                & label_end_timestamp.gt(fold.test_end)
            ).sum()
        ),
    )
