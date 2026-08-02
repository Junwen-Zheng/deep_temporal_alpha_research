from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from deep_alpha.evaluation.release_audit import (
    build_inventory_digest,
    build_release_inventory,
    extract_hash_references,
    inspect_text_files,
    parse_porcelain_paths,
    verify_manifest_hashes,
)


def test_porcelain_paths_are_parsed() -> None:
    output = (
        " M README.md\n"
        "?? configs/new.yaml\n"
    )

    assert parse_porcelain_paths(
        output
    ) == [
        "README.md",
        "configs/new.yaml",
    ]


def test_release_inventory_is_sorted(
    tmp_path: Path,
) -> None:
    second = tmp_path / "b.txt"
    first = tmp_path / "a.txt"

    second.write_text(
        "second\n",
        encoding="utf-8",
    )

    first.write_text(
        "first\n",
        encoding="utf-8",
    )

    inventory = (
        build_release_inventory(
            [
                second,
                first,
            ]
        )
    )

    assert inventory[
        "path"
    ].tolist() == [
        first.as_posix(),
        second.as_posix(),
    ]

    assert inventory[
        "sha256"
    ].str.len().eq(64).all()


def test_inventory_digest_is_stable() -> None:
    frame = pd.DataFrame(
        {
            "path": [
                "b",
                "a",
            ],
            "sha256": [
                "2" * 64,
                "1" * 64,
            ],
        }
    )

    first = build_inventory_digest(
        frame
    )

    second = build_inventory_digest(
        frame.iloc[::-1]
    )

    assert first == second
    assert len(first) == 64


def test_hash_references_are_extracted() -> None:
    payload = {
        "config": "configs/test.yaml",
        "config_sha256": "a" * 64,
        "source_files": {
            "data.csv": "b" * 64,
        },
        "records": [
            {
                "path": "model.bin",
                "sha256": "c" * 64,
            }
        ],
    }

    references = (
        extract_hash_references(
            payload
        )
    )

    assert references == [
        (
            "configs/test.yaml",
            "a" * 64,
        ),
        (
            "data.csv",
            "b" * 64,
        ),
        (
            "model.bin",
            "c" * 64,
        ),
    ]


def test_manifest_hash_verification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"

    source.write_text(
        "evidence\n",
        encoding="utf-8",
    )

    import hashlib

    digest = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    manifest = tmp_path / "manifest.json"

    manifest.write_text(
        json.dumps(
            {
                "source_files": {
                    source.as_posix(): digest,
                }
            }
        ),
        encoding="utf-8",
    )

    result = verify_manifest_hashes(
        [
            manifest,
        ]
    )

    assert len(result) == 1

    assert result.loc[
        0,
        "status",
    ] == "pass"


def test_text_inspection_detects_crlf(
    tmp_path: Path,
) -> None:
    path = tmp_path / "example.txt"

    path.write_bytes(
        b"first\r\nsecond\r\n"
    )

    result = inspect_text_files(
        paths=[
            path,
        ],
        text_extensions=[
            ".txt",
        ],
    )

    assert len(result) == 1
    assert bool(result.loc[0, "utf8"])

    assert not bool(
        result.loc[
            0,
            "lf_only",
        ]
    )

def test_hash_inference_ignores_nonlocal_values() -> None:
    payload = {
        "source": "binance",
        "source_sha256": "a" * 64,
        "remote": (
            "https://example.com/archive.zip"
        ),
        "remote_sha256": "b" * 64,
        "config": "configs/test.yaml",
        "config_sha256": "c" * 64,
    }

    references = (
        extract_hash_references(
            payload
        )
    )

    assert references == [
        (
            "configs/test.yaml",
            "c" * 64,
        )
    ]

def test_manifest_hash_can_resolve_git_history(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"

    repository.mkdir()

    subprocess.run(
        [
            "git",
            "init",
            "-q",
        ],
        cwd=repository,
        check=True,
    )

    subprocess.run(
        [
            "git",
            "config",
            "user.name",
            "Test User",
        ],
        cwd=repository,
        check=True,
    )

    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "test@example.com",
        ],
        cwd=repository,
        check=True,
    )

    config_path = (
        repository
        / "configs"
        / "neural.yaml"
    )

    config_path.parent.mkdir(
        parents=True
    )

    config_path.write_text(
        "version: one\n",
        encoding="utf-8",
    )

    import hashlib

    historical_sha256 = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()

    subprocess.run(
        [
            "git",
            "add",
            "configs/neural.yaml",
        ],
        cwd=repository,
        check=True,
    )

    subprocess.run(
        [
            "git",
            "commit",
            "-q",
            "-m",
            "Add initial neural config",
        ],
        cwd=repository,
        check=True,
    )

    config_path.write_text(
        "version: two\n",
        encoding="utf-8",
    )

    manifest_path = (
        repository
        / "artifacts"
        / "evidence"
        / "model_manifest.json"
    )

    manifest_path.parent.mkdir(
        parents=True
    )

    manifest_path.write_text(
        json.dumps(
            {
                "config": (
                    "configs/neural.yaml"
                ),
                "config_sha256": (
                    historical_sha256
                ),
            }
        ),
        encoding="utf-8",
    )

    result = verify_manifest_hashes(
        [
            manifest_path,
        ],
        repository_root=repository,
    )

    assert len(result) == 1

    assert result.loc[
        0,
        "status",
    ] == "pass"

    assert result.loc[
        0,
        "resolution",
    ] == "git_history"

    assert len(
        result.loc[
            0,
            "matched_revision",
        ]
    ) == 40
