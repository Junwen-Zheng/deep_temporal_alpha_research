from __future__ import annotations

import argparse
from pathlib import Path

from deep_alpha.data.normalize import normalize_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize verified Binance kline CSV files to Parquet."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        type=Path,
    )
    parser.add_argument("--symbol")
    parser.add_argument("--manifest", type=Path)

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    records = normalize_dataset(
        config_path=arguments.config,
        manifest_path=arguments.manifest,
        symbol_filter=arguments.symbol,
    )

    print()
    print("Normalization complete")
    print(f"Symbols: {len(records)}")
    print(f"Rows: {sum(record.rows for record in records)}")
    print(
        "Missing intervals:",
        sum(record.missing_intervals for record in records),
    )


if __name__ == "__main__":
    main()
