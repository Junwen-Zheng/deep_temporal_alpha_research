from __future__ import annotations

import hashlib
import json
import shutil
import string
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from deep_alpha.config import load_yaml

BASE_URL = (
    "https://data.binance.vision/data/futures/um/monthly/klines"
)


@dataclass(frozen=True)
class ArchiveSpec:
    symbol: str
    interval: str
    month: str

    @property
    def filename(self) -> str:
        return f"{self.symbol}-{self.interval}-{self.month}.zip"

    @property
    def checksum_filename(self) -> str:
        return f"{self.filename}.CHECKSUM"

    @property
    def url(self) -> str:
        return (
            f"{BASE_URL}/{self.symbol}/{self.interval}/"
            f"{self.filename}"
        )

    @property
    def checksum_url(self) -> str:
        return f"{self.url}.CHECKSUM"


@dataclass(frozen=True)
class DownloadRecord:
    symbol: str
    interval: str
    month: str
    archive_url: str
    archive_path: str
    checksum_path: str
    extracted_csv_path: str
    sha256: str
    archive_bytes: int
    csv_bytes: int
    csv_lines: int


def _parse_month(value: str) -> tuple[int, int]:
    try:
        year_text, month_text = value.split("-", maxsplit=1)
        year = int(year_text)
        month = int(month_text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid month: {value!r}") from error

    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month: {value!r}")

    return year, month


def iter_months(start_month: str, end_month: str) -> Iterator[str]:
    start_year, start_number = _parse_month(start_month)
    end_year, end_number = _parse_month(end_month)

    current_year = start_year
    current_month = start_number

    while (current_year, current_month) <= (end_year, end_number):
        yield f"{current_year:04d}-{current_month:02d}"

        current_month += 1

        if current_month == 13:
            current_year += 1
            current_month = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_checksum(contents: str) -> str:
    tokens = contents.strip().split()

    if not tokens:
        raise ValueError("Checksum file is empty")

    digest = tokens[0].lower()

    if len(digest) != 64:
        raise ValueError(f"Invalid SHA-256 digest length: {digest!r}")

    if any(character not in string.hexdigits for character in digest):
        raise ValueError(f"Invalid SHA-256 digest: {digest!r}")

    return digest


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        return

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    try:
        with session.get(
            url,
            stream=True,
            timeout=(15, 180),
        ) as response:
            if response.status_code == 404:
                raise FileNotFoundError(
                    f"Remote archive does not exist: {url}"
                )

            response.raise_for_status()

            with temporary.open("wb") as output:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        output.write(chunk)

        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def verify_archive(
    archive_path: Path,
    checksum_path: Path,
) -> str:
    expected = parse_checksum(
        checksum_path.read_text(encoding="utf-8")
    )
    actual = sha256_file(archive_path)

    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {archive_path}: "
            f"expected {expected}, found {actual}"
        )

    return actual


def extract_single_csv(
    archive_path: Path,
    destination_directory: Path,
) -> Path:
    destination_directory.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()

        if bad_member is not None:
            raise ValueError(
                f"ZIP CRC validation failed for {bad_member}"
            )

        members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
        ]

        if len(members) != 1:
            raise ValueError(
                f"{archive_path} must contain exactly one file; "
                f"found {len(members)}"
            )

        member = members[0]
        member_path = Path(member.filename)

        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(
                f"Unsafe ZIP member path: {member.filename}"
            )

        if member_path.suffix.lower() != ".csv":
            raise ValueError(
                f"Expected CSV member, found: {member.filename}"
            )

        output_path = destination_directory / member_path.name

        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        temporary = output_path.with_suffix(".csv.part")

        try:
            with (
                archive.open(member) as source,
                temporary.open("wb") as destination,
            ):
                shutil.copyfileobj(source, destination)

            temporary.replace(output_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    return output_path


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def download_archive(
    session: requests.Session,
    spec: ArchiveSpec,
    raw_directory: Path,
) -> DownloadRecord:
    archive_directory = (
        raw_directory
        / "futures_um"
        / "monthly"
        / "klines"
        / spec.symbol
        / spec.interval
    )

    archive_path = archive_directory / spec.filename
    checksum_path = archive_directory / spec.checksum_filename

    download_file(session, spec.url, archive_path)
    download_file(session, spec.checksum_url, checksum_path)

    digest = verify_archive(archive_path, checksum_path)

    extracted_path = extract_single_csv(
        archive_path,
        archive_directory / "extracted" / spec.month,
    )

    return DownloadRecord(
        symbol=spec.symbol,
        interval=spec.interval,
        month=spec.month,
        archive_url=spec.url,
        archive_path=str(archive_path),
        checksum_path=str(checksum_path),
        extracted_csv_path=str(extracted_path),
        sha256=digest,
        archive_bytes=archive_path.stat().st_size,
        csv_bytes=extracted_path.stat().st_size,
        csv_lines=count_lines(extracted_path),
    )


def write_manifest(
    records: list[DownloadRecord],
    destination: Path,
) -> None:
    ordered = sorted(
        records,
        key=lambda record: (
            record.symbol,
            record.interval,
            record.month,
        ),
    )

    payload = {
        "schema_version": 1,
        "record_count": len(ordered),
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


def download_dataset(
    config_path: str | Path,
    manifest_path: str | Path | None = None,
    symbol_filter: str | None = None,
    month_filter: str | None = None,
) -> list[DownloadRecord]:
    config = load_yaml(config_path)
    data_config = config["data"]

    symbols = list(data_config["symbols"])
    months = list(
        iter_months(
            data_config["start_month"],
            data_config["end_month"],
        )
    )

    if symbol_filter is not None:
        if symbol_filter not in symbols:
            raise ValueError(
                f"Symbol is not configured: {symbol_filter}"
            )

        symbols = [symbol_filter]

    if month_filter is not None:
        if month_filter not in months:
            raise ValueError(
                f"Month is outside the configured range: "
                f"{month_filter}"
            )

        months = [month_filter]

    destination = Path(
        manifest_path or data_config["manifest_path"]
    )
    raw_directory = Path(data_config["raw_dir"])
    interval = str(data_config["interval"])

    specs = [
        ArchiveSpec(
            symbol=symbol,
            interval=interval,
            month=month,
        )
        for symbol in symbols
        for month in months
    ]

    records: list[DownloadRecord] = []

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": (
                    "deep-temporal-alpha-research/0.1"
                )
            }
        )

        for index, spec in enumerate(specs, start=1):
            print(
                f"[{index:03d}/{len(specs):03d}] "
                f"{spec.symbol} {spec.interval} {spec.month}",
                flush=True,
            )

            record = download_archive(
                session=session,
                spec=spec,
                raw_directory=raw_directory,
            )
            records.append(record)

            # Preserve completed work if a later download fails.
            write_manifest(records, destination)

    return records
