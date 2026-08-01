from __future__ import annotations

import argparse
from pathlib import Path

from deep_alpha.data.download import download_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify Binance monthly futures klines."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        type=Path,
    )
    parser.add_argument("--symbol")
    parser.add_argument("--month")
    parser.add_argument("--manifest", type=Path)

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    records = download_dataset(
        config_path=arguments.config,
        manifest_path=arguments.manifest,
        symbol_filter=arguments.symbol,
        month_filter=arguments.month,
    )

    total_archive_bytes = sum(
        record.archive_bytes for record in records
    )
    total_csv_lines = sum(
        record.csv_lines for record in records
    )

    print()
    print("Download complete")
    print(f"Verified archives: {len(records)}")
    print(f"Archive bytes: {total_archive_bytes}")
    print(f"CSV lines: {total_csv_lines}")


if __name__ == "__main__":
    main()
