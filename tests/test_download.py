from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from deep_alpha.data.download import (
    ArchiveSpec,
    extract_single_csv,
    iter_months,
    parse_checksum,
    sha256_file,
)


def test_month_range_is_inclusive() -> None:
    months = list(iter_months("2025-11", "2026-02"))

    assert months == [
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
    ]


def test_archive_url() -> None:
    spec = ArchiveSpec(
        symbol="BTCUSDT",
        interval="5m",
        month="2025-07",
    )

    assert spec.filename == "BTCUSDT-5m-2025-07.zip"
    assert spec.url.endswith(
        "/BTCUSDT/5m/BTCUSDT-5m-2025-07.zip"
    )
    assert spec.checksum_url.endswith(".zip.CHECKSUM")


def test_checksum_parsing_and_file_hash(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"deep-alpha")

    expected = hashlib.sha256(b"deep-alpha").hexdigest()

    assert sha256_file(file_path) == expected
    assert parse_checksum(f"{expected}  sample.bin\n") == expected


def test_invalid_checksum_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid SHA-256"):
        parse_checksum("not-a-valid-checksum")


def test_safe_csv_extraction(tmp_path: Path) -> None:
    archive_path = tmp_path / "valid.zip"

    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(
            "BTCUSDT-5m-2025-07.csv",
            "1,2,3\n4,5,6\n",
        )

    output = extract_single_csv(
        archive_path,
        tmp_path / "extracted",
    )

    assert output.name == "BTCUSDT-5m-2025-07.csv"
    assert output.read_text(encoding="utf-8") == (
        "1,2,3\n4,5,6\n"
    )


def test_unsafe_zip_path_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"

    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("../escape.csv", "unsafe")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        extract_single_csv(
            archive_path,
            tmp_path / "extracted",
        )
