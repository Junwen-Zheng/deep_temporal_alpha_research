from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.reporting.research_report import (
    assert_self_contained_html,
    build_key_findings,
    build_report_html,
    build_source_inventory,
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
        float_format="%.12g",
        lineterminator="\n",
    )


def resolve_source_paths(
    patterns: list[str],
    excluded_prefixes: list[str],
) -> list[Path]:
    candidates = set()

    for pattern in patterns:
        for match in glob(
            pattern,
            recursive=True,
        ):
            path = Path(match)

            if path.is_file():
                candidates.add(path)

    result = []

    for path in sorted(
        candidates,
        key=lambda value: (
            value.as_posix()
        ),
    ):
        path_text = path.as_posix()

        if any(
            path_text.startswith(prefix)
            for prefix in excluded_prefixes
        ):
            continue

        result.append(path)

    if not result:
        raise ValueError(
            "No report source files matched"
        )

    return result


def main() -> None:
    config_path = Path(
        "configs/report.yaml"
    )

    config = load_yaml(
        config_path
    )["report"]

    source_table_paths = {
        str(name): Path(path)
        for name, path in (
            config[
                "source_tables"
            ].items()
        )
    }

    tables = {}

    for name, path in (
        source_table_paths.items()
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

        tables[name] = pd.read_csv(path)

    focus_cost_bps = float(
        config["focus_cost_bps"]
    )

    source_paths = (
        resolve_source_paths(
            patterns=[
                str(value)
                for value in config[
                    "source_globs"
                ]
            ],
            excluded_prefixes=[
                str(value)
                for value in config[
                    "excluded_source_prefixes"
                ]
            ],
        )
    )

    inventory = build_source_inventory(
        source_paths
    )

    findings = build_key_findings(
        tables=tables,
        focus_cost_bps=focus_cost_bps,
    )

    html = build_report_html(
        title=str(config["title"]),
        subtitle=str(
            config["subtitle"]
        ),
        report_version=int(
            config["report_version"]
        ),
        milestones_covered=int(
            config[
                "milestones_covered"
            ]
        ),
        focus_cost_bps=(
            focus_cost_bps
        ),
        tables=tables,
        findings=findings,
        inventory=inventory,
    )

    assert_self_contained_html(html)

    output_path = Path(
        config["output_html"]
    )

    findings_path = Path(
        config["findings_output"]
    )

    inventory_path = Path(
        config["inventory_output"]
    )

    manifest_path = Path(
        config["manifest_output"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        html,
        encoding="utf-8",
        newline="\n",
    )

    write_csv(
        findings,
        findings_path,
    )

    write_csv(
        inventory,
        inventory_path,
    )

    manifest = {
        "schema_version": 1,
        "report_version": int(
            config["report_version"]
        ),
        "milestones_covered": int(
            config[
                "milestones_covered"
            ]
        ),
        "focus_cost_bps": (
            focus_cost_bps
        ),
        "report_config": (
            config_path.as_posix()
        ),
        "report_config_sha256": (
            sha256_file(config_path)
        ),
        "report_path": (
            output_path.as_posix()
        ),
        "report_sha256": (
            sha256_file(output_path)
        ),
        "report_bytes": (
            output_path.stat().st_size
        ),
        "findings_path": (
            findings_path.as_posix()
        ),
        "findings_sha256": (
            sha256_file(findings_path)
        ),
        "inventory_path": (
            inventory_path.as_posix()
        ),
        "inventory_sha256": (
            sha256_file(inventory_path)
        ),
        "self_contained": True,
        "external_resources": False,
        "generation_timestamp_embedded": (
            False
        ),
        "source_table_paths": {
            name: path.as_posix()
            for name, path in sorted(
                source_table_paths.items()
            )
        },
        "source_files": {
            str(row.path): str(row.sha256)
            for row in (
                inventory.itertuples(
                    index=False
                )
            )
        },
        "source_file_count": len(
            inventory
        ),
        "finding_count": len(
            findings
        ),
        "primary_conclusion": (
            "Frozen models contain "
            "cross-sectional predictive "
            "information but do not support "
            "one-basis-point transaction cost."
        ),
        "exploratory_conclusion": (
            "LightGBM without calendar "
            "features is a post-hoc follow-up "
            "hypothesis requiring an untouched "
            "evaluation period."
        ),
    }

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
        newline="\n",
    )

    print(
        "Deterministic research report complete"
    )
    print(
        "Report:",
        output_path,
    )
    print(
        "Report bytes:",
        output_path.stat().st_size,
    )
    print(
        "Report SHA-256:",
        sha256_file(output_path),
    )
    print(
        "Source files:",
        len(inventory),
    )
    print(
        "Key findings:",
        len(findings),
    )

    print()
    print("Primary findings")

    print(
        findings[
            [
                "status",
                "category",
                "finding",
                "model",
                "variant",
                "value",
                "unit",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
