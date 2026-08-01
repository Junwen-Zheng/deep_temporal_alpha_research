from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.data.features import (
    FEATURE_COLUMNS,
    add_forward_targets,
    build_symbol_features,
    finalize_research_panel,
)
from deep_alpha.data.splits import (
    parse_fold_definition,
    summarize_fold,
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
            row_group_size=200_000,
        )

        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_manifest(
    panel: pd.DataFrame,
    output_path: Path,
    source_manifest_path: Path,
    research_config: dict[str, Any],
) -> dict[str, Any]:
    fold_summaries = []

    for fold_configuration in research_config["folds"]:
        fold = parse_fold_definition(fold_configuration)
        summary = summarize_fold(panel, fold)
        fold_summaries.append(summary.to_dict())

    rows_per_symbol = {
        str(symbol): int(count)
        for symbol, count in panel.groupby("symbol").size().items()
    }

    return {
        "schema_version": 1,
        "source_normalized_manifest": str(
            source_manifest_path
        ),
        "source_normalized_manifest_sha256": sha256_file(
            source_manifest_path
        ),
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
        "total_rows": len(panel),
        "timestamp_count": int(
            panel["timestamp"].nunique()
        ),
        "symbol_count": int(panel["symbol"].nunique()),
        "rows_per_symbol": rows_per_symbol,
        "start_timestamp": panel[
            "timestamp"
        ].min().isoformat(),
        "end_timestamp": panel[
            "timestamp"
        ].max().isoformat(),
        "sequence_length": int(
            research_config["sequence_length"]
        ),
        "target_horizon_bars": int(
            research_config["target_horizon_bars"]
        ),
        "interval_minutes": int(
            research_config["interval_minutes"]
        ),
        "feature_columns": FEATURE_COLUMNS,
        "folds": fold_summaries,
    }


def main() -> None:
    data_config = load_yaml("configs/data.yaml")["data"]
    research_config = load_yaml(
        "configs/research.yaml"
    )["research"]

    symbols = list(data_config["symbols"])
    processed_directory = Path(
        data_config["processed_dir"]
    )

    feature_frames = []

    for index, symbol in enumerate(symbols, start=1):
        print(
            f"[{index:02d}/{len(symbols):02d}] "
            f"Building features for {symbol}",
            flush=True,
        )

        input_path = (
            processed_directory
            / "bars"
            / f"symbol={symbol}"
            / "part-00000.parquet"
        )

        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        bars = pd.read_parquet(input_path)

        feature_frames.append(
            build_symbol_features(bars)
        )

    print("Combining symbol panels", flush=True)

    panel = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    print("Building forward targets", flush=True)

    panel = add_forward_targets(
        panel=panel,
        horizon_bars=int(
            research_config["target_horizon_bars"]
        ),
        interval_minutes=int(
            research_config["interval_minutes"]
        ),
    )

    print("Finalizing research panel", flush=True)

    panel = finalize_research_panel(
        panel=panel,
        expected_symbol_count=int(
            research_config["expected_symbol_count"]
        ),
    )

    output_path = Path(
        research_config["output_path"]
    )

    print(f"Writing {output_path}", flush=True)
    write_parquet(panel, output_path)

    source_manifest_path = Path(
        data_config["normalized_manifest_path"]
    )

    manifest = build_manifest(
        panel=panel,
        output_path=output_path,
        source_manifest_path=source_manifest_path,
        research_config=research_config,
    )

    manifest_path = Path(
        research_config["manifest_path"]
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Research dataset complete")
    print("Rows:", manifest["total_rows"])
    print(
        "Timestamps:",
        manifest["timestamp_count"],
    )
    print("Symbols:", manifest["symbol_count"])
    print("Features:", len(FEATURE_COLUMNS))

    for fold in manifest["folds"]:
        print(
            f"Fold {fold['fold_id']}: "
            f"train={fold['train_rows']}, "
            f"validation={fold['validation_rows']}, "
            f"test={fold['test_rows']}"
        )


if __name__ == "__main__":
    main()
