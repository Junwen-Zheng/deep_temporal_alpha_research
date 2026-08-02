from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def calculate_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_porcelain_paths(
    output: str,
) -> list[str]:
    paths = []

    for line in output.splitlines():
        if not line:
            continue

        if len(line) < 4:
            raise ValueError(
                "Malformed git status line"
            )

        path = line[3:]

        if " -> " in path:
            path = path.split(
                " -> ",
                maxsplit=1,
            )[1]

        paths.append(path)

    return sorted(set(paths))


def classify_release_path(
    path: Path,
) -> str:
    parts = path.parts

    if not parts:
        return "other"

    if parts[0] == "artifacts":
        if (
            len(parts) > 1
            and parts[1] == "evidence"
        ):
            return "evidence"

        if (
            len(parts) > 1
            and parts[1] == "reports"
        ):
            return "report"

        return "artifact"

    if parts[0] == "src":
        return "source"

    if parts[0] == "tests":
        return "test"

    if parts[0] == "scripts":
        return "script"

    if parts[0] == "configs":
        return "configuration"

    if parts[0] == "docs":
        return "documentation"

    return "repository"


def build_release_inventory(
    paths: Sequence[Path],
) -> pd.DataFrame:
    unique_paths = sorted(
        {
            Path(path)
            for path in paths
        },
        key=lambda value: (
            value.as_posix()
        ),
    )

    if not unique_paths:
        raise ValueError(
            "Release inventory is empty"
        )

    rows = []

    for path in unique_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        rows.append(
            {
                "path": path.as_posix(),
                "category": (
                    classify_release_path(
                        path
                    )
                ),
                "bytes": (
                    path.stat().st_size
                ),
                "sha256": (
                    calculate_sha256(path)
                ),
            }
        )

    return pd.DataFrame(rows)


def build_inventory_digest(
    inventory: pd.DataFrame,
) -> str:
    required_columns = {
        "path",
        "sha256",
    }

    missing = required_columns - set(
        inventory.columns
    )

    if missing:
        raise ValueError(
            "Inventory digest columns "
            f"are missing: {sorted(missing)}"
        )

    ordered = inventory.sort_values(
        "path"
    )

    payload = "".join(
        f"{row.path}\0{row.sha256}\n"
        for row in ordered.itertuples(
            index=False
        )
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()



def _looks_like_local_file_reference(
    value: Any,
) -> bool:
    if not isinstance(value, str):
        return False

    if not value:
        return False

    lowered = value.lower()

    if (
        "://" in value
        or lowered.startswith(
            (
                "mailto:",
                "urn:",
            )
        )
    ):
        return False

    candidate = Path(value)

    return (
        "/" in value
        or "\\" in value
        or bool(candidate.suffix)
        or value.startswith(".")
    )


def extract_hash_references(
    payload: Any,
) -> list[
    tuple[
        str,
        str,
    ]
]:
    references: list[
        tuple[
            str,
            str,
        ]
    ] = []

    def add_reference(
        path_value: Any,
        hash_value: Any,
    ) -> None:
        if not isinstance(
            path_value,
            str,
        ):
            return

        if not isinstance(
            hash_value,
            str,
        ):
            return

        if len(hash_value) != 64:
            return

        if (
            not path_value
            or "://" in path_value
            or path_value.lower().startswith(
                (
                    "mailto:",
                    "urn:",
                )
            )
        ):
            return

        references.append(
            (
                path_value,
                hash_value,
            )
        )

    def walk(
        value: Any,
    ) -> None:
        if isinstance(value, Mapping):
            if (
                "path" in value
                and "sha256" in value
            ):
                add_reference(
                    value["path"],
                    value["sha256"],
                )

            for mapping_key in [
                "source_files",
                "evidence_files",
            ]:
                mapping = value.get(
                    mapping_key
                )

                if isinstance(
                    mapping,
                    Mapping,
                ):
                    for path_value, (
                        hash_value
                    ) in mapping.items():
                        add_reference(
                            path_value,
                            hash_value,
                        )

            for key, hash_value in (
                value.items()
            ):
                if not isinstance(
                    key,
                    str,
                ):
                    continue

                if not key.endswith(
                    "_sha256"
                ):
                    continue

                base = key.removesuffix(
                    "_sha256"
                )

                for candidate in [
                    base,
                    f"{base}_path",
                ]:
                    candidate_value = value.get(
                        candidate
                    )

                    if (
                        _looks_like_local_file_reference(
                            candidate_value
                        )
                    ):
                        add_reference(
                            candidate_value,
                            hash_value,
                        )
                        break

            for nested in value.values():
                walk(nested)

        elif isinstance(
            value,
            Sequence,
        ) and not isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            for nested in value:
                walk(nested)

    walk(payload)

    return sorted(
        set(references),
        key=lambda value: (
            value[0],
            value[1],
        ),
    )



def find_git_revision_with_hash(
    path: Path,
    expected_sha256: str,
    repository_root: Path,
) -> str | None:
    if path.is_absolute():
        return None

    history = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "log",
            "--format=%H",
            "--all",
            "--",
            path.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if history.returncode != 0:
        return None

    revisions = [
        value.strip()
        for value in history.stdout.splitlines()
        if value.strip()
    ]

    for revision in revisions:
        blob = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "show",
                (
                    f"{revision}:"
                    f"{path.as_posix()}"
                ),
            ],
            check=False,
            capture_output=True,
        )

        if blob.returncode != 0:
            continue

        actual_sha256 = hashlib.sha256(
            blob.stdout
        ).hexdigest()

        if actual_sha256 == expected_sha256:
            return revision

    return None


def verify_manifest_hashes(
    manifest_paths: Sequence[Path],
    repository_root: Path | None = None,
) -> pd.DataFrame:
    rows = []

    root = (
        Path(repository_root)
        if repository_root is not None
        else None
    )

    for manifest_path in sorted(
        manifest_paths,
        key=lambda value: (
            value.as_posix()
        ),
    ):
        if not manifest_path.is_file():
            raise FileNotFoundError(
                manifest_path
            )

        payload = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        references = (
            extract_hash_references(
                payload
            )
        )

        for path_text, expected in (
            references
        ):
            path = Path(path_text)

            current_path = (
                path
                if path.is_absolute()
                or root is None
                else root / path
            )

            if current_path.is_file():
                actual = calculate_sha256(
                    current_path
                )
            else:
                actual = ""

            matched_revision = ""
            resolution = "missing"
            status = "fail"

            if actual == expected:
                status = "pass"
                resolution = "current"
            elif root is not None:
                historical_revision = (
                    find_git_revision_with_hash(
                        path=path,
                        expected_sha256=expected,
                        repository_root=root,
                    )
                )

                if historical_revision is not None:
                    status = "pass"
                    resolution = "git_history"
                    matched_revision = (
                        historical_revision
                    )
                elif actual:
                    resolution = (
                        "current_hash_mismatch"
                    )

            rows.append(
                {
                    "manifest": (
                        manifest_path.as_posix()
                    ),
                    "path": path_text,
                    "expected_sha256": (
                        expected
                    ),
                    "actual_sha256": (
                        actual
                    ),
                    "resolution": resolution,
                    "matched_revision": (
                        matched_revision
                    ),
                    "status": status,
                }
            )

    if not rows:
        raise ValueError(
            "No manifest hash references "
            "were found"
        )

    return pd.DataFrame(rows)


def inspect_text_files(
    paths: Sequence[Path],
    text_extensions: Sequence[str],
) -> pd.DataFrame:
    extensions = {
        value.lower()
        for value in text_extensions
    }

    absolute_tokens = [
        "/" + "Users" + "/",
        (
            "C:"
            + "\\"
            + "Users"
            + "\\"
        ),
    ]

    rows = []

    for path in sorted(
        {
            Path(value)
            for value in paths
        },
        key=lambda value: (
            value.as_posix()
        ),
    ):
        if path.suffix.lower() not in (
            extensions
        ):
            continue

        raw = path.read_bytes()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            rows.append(
                {
                    "path": path.as_posix(),
                    "utf8": False,
                    "lf_only": False,
                    "absolute_user_path": (
                        False
                    ),
                }
            )
            continue

        rows.append(
            {
                "path": path.as_posix(),
                "utf8": True,
                "lf_only": (
                    b"\r\n" not in raw
                    and b"\r" not in raw
                ),
                "absolute_user_path": (
                    any(
                        token in text
                        for token in (
                            absolute_tokens
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(rows)
