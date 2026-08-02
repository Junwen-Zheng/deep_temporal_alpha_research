from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.evaluation.release_audit import (
    build_inventory_digest,
    build_release_inventory,
    calculate_sha256,
    inspect_text_files,
    parse_porcelain_paths,
    verify_manifest_hashes,
)


def run_command(
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )


def write_csv(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        destination,
        index=False,
        lineterminator="\n",
    )


def main() -> None:
    config_path = Path(
        "configs/release_audit.yaml"
    )

    config = load_yaml(
        config_path
    )["release_audit"]

    required_test_count = int(
        config[
            "required_test_count"
        ]
    )

    maximum_file_bytes = int(
        config[
            "maximum_file_bytes"
        ]
    )

    report_path = Path(
        config["report_html"]
    )

    report_manifest_path = Path(
        config["report_manifest"]
    )

    report_rebuild_path = Path(
        config[
            "report_rebuild_record"
        ]
    )

    audit_path = Path(
        config["audit_output"]
    )

    inventory_path = Path(
        config["inventory_output"]
    )

    environment_path = Path(
        config["environment_output"]
    )

    manifest_path = Path(
        config["manifest_output"]
    )

    exclusions = {
        str(value)
        for value in config[
            "inventory_exclusions"
        ]
    }

    allowed_worktree = {
        str(value)
        for value in config[
            "allowed_worktree_paths"
        ]
    }

    required_packages = [
        str(value)
        for value in config[
            "required_packages"
        ]
    ]

    package_versions = {}

    for package in sorted(
        required_packages
    ):
        package_versions[package] = (
            importlib.metadata.version(
                package
            )
        )

    environment = {
        "schema_version": 1,
        "python_implementation": (
            platform.python_implementation()
        ),
        "python_version": (
            platform.python_version()
        ),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": package_versions,
    }

    environment_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    environment_path.write_text(
        json.dumps(
            environment,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    audit_rows: list[
        dict[str, Any]
    ] = []

    def add_check(
        check: str,
        passed: bool,
        detail: str,
    ) -> None:
        audit_rows.append(
            {
                "check": check,
                "status": (
                    "pass"
                    if passed
                    else "fail"
                ),
                "detail": detail,
            }
        )

    pytest_result = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ]
    )

    test_match = re.search(
        r"(\d+) passed",
        (
            pytest_result.stdout
            + pytest_result.stderr
        ),
    )

    test_count = (
        int(test_match.group(1))
        if test_match
        else -1
    )

    add_check(
        "pytest",
        (
            pytest_result.returncode == 0
            and test_count
            == required_test_count
        ),
        f"{test_count} passed",
    )

    ruff_result = run_command(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            ".",
        ]
    )

    add_check(
        "ruff",
        ruff_result.returncode == 0,
        "Ruff checks passed",
    )

    pip_result = run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "check",
        ]
    )

    add_check(
        "pip_check",
        pip_result.returncode == 0,
        "Dependency consistency passed",
    )

    compile_result = run_command(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "src",
            "scripts",
            "tests",
        ]
    )

    add_check(
        "compileall",
        compile_result.returncode == 0,
        "Python compilation passed",
    )

    diff_result = run_command(
        [
            "git",
            "diff",
            "--check",
        ]
    )

    add_check(
        "git_diff_check",
        diff_result.returncode == 0,
        "Git whitespace check passed",
    )

    status_result = run_command(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    )

    changed_paths = set(
        parse_porcelain_paths(
            status_result.stdout
        )
    )

    unexpected_paths = sorted(
        changed_paths
        - allowed_worktree
    )

    add_check(
        "worktree_scope",
        (
            status_result.returncode == 0
            and not unexpected_paths
        ),
        (
            "Only configured release-audit "
            "paths are modified or untracked"
            if not unexpected_paths
            else (
                "Unexpected paths: "
                + ",".join(
                    unexpected_paths
                )
            )
        ),
    )

    if not report_rebuild_path.is_file():
        raise FileNotFoundError(
            report_rebuild_path
        )

    report_rebuild = json.loads(
        report_rebuild_path.read_text(
            encoding="utf-8"
        )
    )

    rebuild_artifacts = (
        report_rebuild[
            "artifacts"
        ]
    )

    rebuild_stable = bool(
        report_rebuild["stable"]
    )

    rebuild_hashes_current = all(
        Path(path_text).is_file()
        and calculate_sha256(
            Path(path_text)
        )
        == expected_hash
        for path_text, expected_hash in (
            rebuild_artifacts.items()
        )
    )

    add_check(
        "report_rebuild",
        (
            rebuild_stable
            and rebuild_hashes_current
            and int(
                report_rebuild[
                    "build_count"
                ]
            )
            == 2
        ),
        (
            "Frozen report was rebuilt "
            "twice with identical hashes"
        ),
    )

    if not report_manifest_path.is_file():
        raise FileNotFoundError(
            report_manifest_path
        )

    report_manifest = json.loads(
        report_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    report_hash_matches = (
        report_path.is_file()
        and calculate_sha256(
            report_path
        )
        == report_manifest[
            "report_sha256"
        ]
        == rebuild_artifacts[
            report_path.as_posix()
        ]
    )

    add_check(
        "report_manifest",
        report_hash_matches,
        (
            "Report hash matches report "
            "manifest and rebuild record"
        ),
    )

    excluded_manifests = {
        str(value)
        for value in config[
            "excluded_manifests"
        ]
    }

    manifest_paths = [
        Path(path_text)
        for path_text in glob(
            str(
                config[
                    "manifest_glob"
                ]
            )
        )
        if path_text
        not in excluded_manifests
    ]

    hash_references = (
        verify_manifest_hashes(
            manifest_paths,
            repository_root=Path("."),
        )
    )

    failed_hash_references = (
        hash_references.loc[
            hash_references[
                "status"
            ]
            != "pass"
        ]
    )

    manifest_hashes_pass = (
        failed_hash_references.empty
    )

    if manifest_hashes_pass:
        manifest_hash_detail = (
            f"{len(hash_references)} "
            "historical manifest references "
            "verified"
        )
    else:
        examples = "; ".join(
            (
                f"{row.manifest} -> "
                f"{row.path}"
            )
            for row in (
                failed_hash_references
                .head(5)
                .itertuples(index=False)
            )
        )

        manifest_hash_detail = (
            f"{len(failed_hash_references)} "
            f"of {len(hash_references)} "
            "manifest references failed: "
            f"{examples}"
        )

    add_check(
        "manifest_hash_references",
        manifest_hashes_pass,
        manifest_hash_detail,
    )

    tracked_result = run_command(
        [
            "git",
            "ls-files",
            "-z",
        ]
    )

    if tracked_result.returncode != 0:
        raise RuntimeError(
            "Unable to enumerate tracked files"
        )

    tracked_paths = {
        value
        for value in (
            tracked_result.stdout.split(
                "\0"
            )
        )
        if value
    }

    pending_paths = {
        path
        for path in allowed_worktree
        if Path(path).is_file()
    }

    release_paths = sorted(
        {
            Path(path)
            for path in (
                tracked_paths
                | pending_paths
            )
            if path not in exclusions
        },
        key=lambda value: (
            value.as_posix()
        ),
    )

    inventory = (
        build_release_inventory(
            release_paths
        )
    )

    inventory_sorted = (
        inventory["path"]
        .is_monotonic_increasing
    )

    inventory_unique = (
        not inventory[
            "path"
        ].duplicated().any()
    )

    add_check(
        "release_inventory",
        (
            inventory_sorted
            and inventory_unique
        ),
        (
            f"{len(inventory)} release files "
            "inventoried"
        ),
    )

    oversized = inventory.loc[
        inventory["bytes"]
        > maximum_file_bytes
    ]

    add_check(
        "file_size_limit",
        oversized.empty,
        (
            "No release file exceeds "
            f"{maximum_file_bytes} bytes"
            if oversized.empty
            else (
                "Oversized files: "
                + ",".join(
                    oversized["path"]
                )
            )
        ),
    )

    symlinks = [
        path.as_posix()
        for path in release_paths
        if path.is_symlink()
    ]

    add_check(
        "symlinks",
        not symlinks,
        (
            "No release symlinks found"
            if not symlinks
            else (
                "Symlinks: "
                + ",".join(symlinks)
            )
        ),
    )

    text_audit = inspect_text_files(
        paths=release_paths,
        text_extensions=[
            str(value)
            for value in config[
                "text_extensions"
            ]
        ],
    )

    add_check(
        "utf8_text",
        (
            text_audit.empty
            or text_audit[
                "utf8"
            ].all()
        ),
        "Release text files decode as UTF-8",
    )

    add_check(
        "line_endings",
        (
            text_audit.empty
            or text_audit[
                "lf_only"
            ].all()
        ),
        "Release text files use LF endings",
    )

    add_check(
        "absolute_user_paths",
        (
            text_audit.empty
            or not text_audit[
                "absolute_user_path"
            ].any()
        ),
        (
            "No machine-specific user paths "
            "found in release text"
        ),
    )

    add_check(
        "environment_packages",
        (
            set(package_versions)
            == set(required_packages)
            and all(
                package_versions.values()
            )
        ),
        (
            f"{len(package_versions)} required "
            "package versions recorded"
        ),
    )

    audit = pd.DataFrame(
        audit_rows
    )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        audit,
        audit_path,
    )

    write_csv(
        inventory,
        inventory_path,
    )

    release_ready = (
        audit["status"]
        .eq("pass")
        .all()
    )

    release_manifest = {
        "schema_version": 1,
        "release_ready": (
            bool(release_ready)
        ),
        "required_test_count": (
            required_test_count
        ),
        "verified_test_count": (
            test_count
        ),
        "audit_check_count": (
            len(audit)
        ),
        "inventory_file_count": (
            len(inventory)
        ),
        "inventory_digest": (
            build_inventory_digest(
                inventory
            )
        ),
        "historical_manifest_count": (
            len(manifest_paths)
        ),
        "manifest_reference_count": (
            len(hash_references)
        ),
        "maximum_file_bytes": (
            maximum_file_bytes
        ),
        "report_path": (
            report_path.as_posix()
        ),
        "report_sha256": (
            calculate_sha256(
                report_path
            )
        ),
        "report_bytes": (
            report_path.stat().st_size
        ),
        "report_rebuild_count": int(
            report_rebuild[
                "build_count"
            ]
        ),
        "report_rebuild_stable": (
            rebuild_stable
        ),
        "python_version": (
            environment[
                "python_version"
            ]
        ),
        "python_implementation": (
            environment[
                "python_implementation"
            ]
        ),
        "system": environment["system"],
        "machine": environment["machine"],
        "packages": package_versions,
        "audit_path": (
            audit_path.as_posix()
        ),
        "audit_sha256": (
            calculate_sha256(
                audit_path
            )
        ),
        "inventory_path": (
            inventory_path.as_posix()
        ),
        "inventory_sha256": (
            calculate_sha256(
                inventory_path
            )
        ),
        "environment_path": (
            environment_path.as_posix()
        ),
        "environment_sha256": (
            calculate_sha256(
                environment_path
            )
        ),
        "report_rebuild_record": (
            report_rebuild_path.as_posix()
        ),
        "report_rebuild_sha256": (
            calculate_sha256(
                report_rebuild_path
            )
        ),
        "inventory_exclusions": (
            sorted(exclusions)
        ),
        "aggregate_outputs_excluded_from_inventory": (
            True
        ),
        "generation_timestamp_embedded": (
            False
        ),
    }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path.write_text(
        json.dumps(
            release_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Reproducibility and release audit "
        "complete"
    )

    print()
    print(
        audit.to_string(
            index=False
        )
    )

    print()
    print(
        "Release files:",
        len(inventory),
    )

    print(
        "Historical manifests:",
        len(manifest_paths),
    )

    print(
        "Manifest references:",
        len(hash_references),
    )

    print(
        "Inventory digest:",
        release_manifest[
            "inventory_digest"
        ],
    )

    if not release_ready:
        failed = audit.loc[
            audit["status"] != "pass"
        ]

        raise RuntimeError(
            "Release audit failed:\n"
            + failed.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
