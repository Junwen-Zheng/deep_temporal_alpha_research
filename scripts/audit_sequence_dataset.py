from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.data.features import FEATURE_COLUMNS
from deep_alpha.data.sequences import (
    build_sequence_split,
    build_temporal_panel,
)
from deep_alpha.data.splits import (
    build_fold_masks,
    parse_fold_definition,
)


def main() -> None:
    research_config_path = Path(
        "configs/research.yaml"
    )

    research_config = load_yaml(
        research_config_path
    )["research"]

    research_manifest_path = Path(
        research_config["manifest_path"]
    )

    research_manifest = json.loads(
        research_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    panel_path = Path(
        research_manifest["output_path"]
    )

    columns = [
        "timestamp",
        "label_end_timestamp",
        "symbol",
        "raw_target",
        "target",
        *FEATURE_COLUMNS,
    ]

    print("Loading research panel", flush=True)

    panel = pd.read_parquet(
        panel_path,
        columns=columns,
    )

    sequence_length = int(
        research_config["sequence_length"]
    )

    interval_minutes = int(
        research_config["interval_minutes"]
    )

    expected_symbol_count = int(
        research_config["expected_symbol_count"]
    )

    print("Building temporal tensor layout", flush=True)

    temporal_panel = build_temporal_panel(
        frame=panel,
        feature_columns=FEATURE_COLUMNS,
        expected_symbol_count=expected_symbol_count,
        interval_minutes=interval_minutes,
    )

    fold_records: list[dict[str, Any]] = []

    for fold_configuration in research_config["folds"]:
        fold = parse_fold_definition(
            fold_configuration
        )

        masks = build_fold_masks(panel, fold)

        split_records = []

        for split_name, row_mask in [
            ("train", masks.train),
            ("validation", masks.validation),
            ("test", masks.test),
        ]:
            split = build_sequence_split(
                frame=panel,
                row_mask=row_mask,
                temporal_panel=temporal_panel,
                sequence_length=sequence_length,
                split_name=split_name,
            )

            split_records.append(
                split.to_dict()
            )

        fold_records.append(
            {
                "fold": fold.fold_id,
                "train_end": fold.train_end.isoformat(),
                "validation_end": (
                    fold.validation_end.isoformat()
                ),
                "test_end": fold.test_end.isoformat(),
                "splits": split_records,
            }
        )

    manifest = {
        "schema_version": 1,
        "source_research_manifest": str(
            research_manifest_path
        ),
        "source_research_manifest_sha256": (
            sha256_file(research_manifest_path)
        ),
        "research_panel": str(panel_path),
        "research_panel_sha256": sha256_file(
            panel_path
        ),
        "research_config": str(
            research_config_path
        ),
        "research_config_sha256": sha256_file(
            research_config_path
        ),
        "sequence_length": sequence_length,
        "history_span_minutes": (
            (sequence_length - 1)
            * interval_minutes
        ),
        "interval_minutes": interval_minutes,
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "timestamp_count": (
            temporal_panel.timestamp_count
        ),
        "symbol_count": temporal_panel.symbol_count,
        "symbols": list(temporal_panel.symbols),
        "first_panel_timestamp": (
            temporal_panel.timestamps[0].isoformat()
        ),
        "first_usable_endpoint_timestamp": (
            temporal_panel.timestamps[
                sequence_length - 1
            ].isoformat()
        ),
        "last_panel_timestamp": (
            temporal_panel.timestamps[-1].isoformat()
        ),
        "folds": fold_records,
    }

    destination = Path(
        "artifacts/evidence/"
        "sequence_dataset_manifest.json"
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Sequence dataset audit complete")
    print(
        "Panel shape:",
        (
            temporal_panel.timestamp_count,
            temporal_panel.symbol_count,
            temporal_panel.feature_count,
        ),
    )

    print(
        "First usable endpoint:",
        manifest["first_usable_endpoint_timestamp"],
    )

    for fold_record in fold_records:
        print(f"Fold {fold_record['fold']}")

        for split in fold_record["splits"]:
            print(
                f"  {split['split']}: "
                f"timestamps={split['timestamp_count']} "
                f"rows={split['row_count']} "
                f"history_excluded="
                f"{split['excluded_history_rows']}"
            )


if __name__ == "__main__":
    main()
