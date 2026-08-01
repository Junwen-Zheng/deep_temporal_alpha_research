from __future__ import annotations

import pandas as pd
import pytest

from deep_alpha.data.splits import (
    build_fold_masks,
    parse_fold_definition,
    summarize_fold,
)


def sample_panel() -> pd.DataFrame:
    timestamp = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=48,
        freq="5min",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "label_end_timestamp": (
                timestamp + pd.Timedelta(minutes=60)
            ),
        }
    )


def test_split_purges_labels_crossing_boundaries() -> None:
    frame = sample_panel()

    fold = parse_fold_definition(
        {
            "fold_id": 1,
            "train_end": "2026-01-01T00:55:00Z",
            "validation_end": "2026-01-01T01:55:00Z",
            "test_end": "2026-01-01T03:55:00Z",
        }
    )

    masks = build_fold_masks(frame, fold)

    assert (
        frame.loc[masks.train, "label_end_timestamp"]
        <= fold.train_end
    ).all()

    assert (
        frame.loc[
            masks.validation,
            "label_end_timestamp",
        ]
        <= fold.validation_end
    ).all()

    assert (
        frame.loc[masks.test, "label_end_timestamp"]
        <= fold.test_end
    ).all()


def test_split_masks_do_not_overlap() -> None:
    frame = sample_panel()

    fold = parse_fold_definition(
        {
            "fold_id": 1,
            "train_end": "2026-01-01T00:55:00Z",
            "validation_end": "2026-01-01T01:55:00Z",
            "test_end": "2026-01-01T03:55:00Z",
        }
    )

    masks = build_fold_masks(frame, fold)

    assert not (masks.train & masks.validation).any()
    assert not (masks.train & masks.test).any()
    assert not (masks.validation & masks.test).any()


def test_fold_summary_reports_purged_rows() -> None:
    frame = sample_panel()

    fold = parse_fold_definition(
        {
            "fold_id": 1,
            "train_end": "2026-01-01T00:55:00Z",
            "validation_end": "2026-01-01T01:55:00Z",
            "test_end": "2026-01-01T03:55:00Z",
        }
    )

    summary = summarize_fold(frame, fold)

    assert summary.purged_train_boundary_rows == 12
    assert summary.purged_validation_boundary_rows == 12
    assert summary.purged_test_boundary_rows == 12


def test_fold_boundaries_must_be_ordered() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid boundary ordering",
    ):
        parse_fold_definition(
            {
                "fold_id": 1,
                "train_end": "2026-03-01T00:00:00Z",
                "validation_end": "2026-02-01T00:00:00Z",
                "test_end": "2026-04-01T00:00:00Z",
            }
        )
