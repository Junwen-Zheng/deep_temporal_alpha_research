from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.evaluation.costs import (
    build_break_even_cohort_metrics,
    build_break_even_cost_curve,
    build_break_even_model_summary,
    build_cost_grid,
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
    )


def main() -> None:
    break_even_config_path = Path(
        "configs/break_even.yaml"
    )

    break_even_config = load_yaml(
        break_even_config_path
    )["break_even"]

    evaluation_config_path = Path(
        break_even_config[
            "evaluation_config"
        ]
    )

    evaluation_config = load_yaml(
        evaluation_config_path
    )["evaluation"]

    source_cohort_path = Path(
        break_even_config[
            "source_cohort_metrics"
        ]
    )

    source_summary_path = Path(
        break_even_config[
            "source_portfolio_summary"
        ]
    )

    source_manifest_path = Path(
        break_even_config[
            "source_portfolio_manifest"
        ]
    )

    source_cohorts = pd.read_csv(
        source_cohort_path
    )

    source_summary = pd.read_csv(
        source_summary_path
    )

    source_manifest = json.loads(
        source_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    annualization_periods = int(
        evaluation_config[
            "annualization_periods"
        ]
    )

    robustness_costs = [
        float(value)
        for value in break_even_config[
            "robustness_costs_bps"
        ]
    ]

    curve_configuration = (
        break_even_config[
            "cost_curve"
        ]
    )

    cost_grid = build_cost_grid(
        start_bps=float(
            curve_configuration[
                "start_bps"
            ]
        ),
        end_bps=float(
            curve_configuration[
                "end_bps"
            ]
        ),
        step_bps=float(
            curve_configuration[
                "step_bps"
            ]
        ),
    )

    cohort_metrics = (
        build_break_even_cohort_metrics(
            cohort_metrics=source_cohorts,
            annualization_periods=(
                annualization_periods
            ),
        )
    )

    model_summary = (
        build_break_even_model_summary(
            cohort_metrics=cohort_metrics,
            annualization_periods=(
                annualization_periods
            ),
            robustness_costs_bps=(
                robustness_costs
            ),
        )
    )

    cost_curve = build_break_even_cost_curve(
        cohort_metrics=cohort_metrics,
        cost_grid_bps=cost_grid,
        annualization_periods=(
            annualization_periods
        ),
    )

    print(
        "Verifying dense curve against "
        "committed cost scenarios",
        flush=True,
    )

    for source_row in (
        source_summary.itertuples(
            index=False
        )
    ):
        candidate = cost_curve.loc[
            (
                cost_curve["model"]
                == source_row.model
            )
            & np.isclose(
                cost_curve[
                    "cost_bps"
                ].to_numpy(
                    dtype=np.float64
                ),
                float(
                    source_row.cost_bps
                ),
                rtol=0.0,
                atol=1e-12,
            )
        ]

        if len(candidate) != 1:
            raise ValueError(
                "Could not uniquely match "
                f"{source_row.model} at "
                f"{source_row.cost_bps} bps"
            )

        curve_row = candidate.iloc[0]

        comparisons = {
            (
                "mean annualized "
                "net return"
            ): (
                curve_row[
                    "mean_cohort_"
                    "annualized_net_return"
                ],
                source_row.mean_annualized_net_return,
            ),
            (
                "worst annualized "
                "net return"
            ): (
                curve_row[
                    "worst_cohort_"
                    "annualized_net_return"
                ],
                source_row.worst_annualized_net_return,
            ),
            "mean turnover": (
                curve_row[
                    "mean_turnover"
                ],
                source_row.mean_turnover,
            ),
        }

        for name, (
            actual,
            expected,
        ) in comparisons.items():
            if not np.isclose(
                float(actual),
                float(expected),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"Curve {name} mismatch "
                    f"for {source_row.model} "
                    f"at {source_row.cost_bps} "
                    f"bps: {actual} versus "
                    f"{expected}"
                )

    expected_models = [
        str(model)
        for model in evaluation_config[
            "models"
        ]
    ]

    if set(
        cohort_metrics["model"]
    ) != set(expected_models):
        raise ValueError(
            "Break-even models differ "
            "from evaluation models"
        )

    if (
        source_manifest[
            "annualization_periods"
        ]
        != annualization_periods
    ):
        raise ValueError(
            "Annualization differs from "
            "the portfolio manifest"
        )

    evidence_directory = Path(
        break_even_config[
            "evidence_dir"
        ]
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort_path = (
        evidence_directory
        / "break_even_cohort_metrics.csv"
    )

    summary_path = (
        evidence_directory
        / "break_even_model_summary.csv"
    )

    curve_path = (
        evidence_directory
        / "break_even_cost_curve.csv"
    )

    write_csv(
        cohort_metrics,
        cohort_path,
    )

    write_csv(
        model_summary,
        summary_path,
    )

    write_csv(
        cost_curve,
        curve_path,
    )

    manifest_path = (
        evidence_directory
        / "break_even_run_manifest.json"
    )

    evidence_files = [
        cohort_path,
        summary_path,
        curve_path,
    ]

    manifest = {
        "schema_version": 1,
        "break_even_config": str(
            break_even_config_path
        ),
        "break_even_config_sha256": (
            sha256_file(
                break_even_config_path
            )
        ),
        "evaluation_config": str(
            evaluation_config_path
        ),
        "evaluation_config_sha256": (
            sha256_file(
                evaluation_config_path
            )
        ),
        "source_cohort_metrics": str(
            source_cohort_path
        ),
        "source_cohort_metrics_sha256": (
            sha256_file(
                source_cohort_path
            )
        ),
        "source_portfolio_summary": str(
            source_summary_path
        ),
        "source_portfolio_summary_sha256": (
            sha256_file(
                source_summary_path
            )
        ),
        "source_portfolio_manifest": str(
            source_manifest_path
        ),
        "source_portfolio_manifest_sha256": (
            sha256_file(
                source_manifest_path
            )
        ),
        "models": expected_models,
        "annualization_periods": (
            annualization_periods
        ),
        "cohort_count_per_model": (
            int(
                cohort_metrics.groupby(
                    "model"
                )["cohort"].nunique().iloc[0]
            )
        ),
        "cost_curve_start_bps": float(
            cost_grid[0]
        ),
        "cost_curve_end_bps": float(
            cost_grid[-1]
        ),
        "cost_curve_step_bps": float(
            curve_configuration[
                "step_bps"
            ]
        ),
        "cost_curve_points": len(
            cost_grid
        ),
        "robustness_costs_bps": (
            robustness_costs
        ),
        "break_even_definition": (
            "transaction cost per unit of "
            "one-way traded notional at "
            "which mean net log return is zero"
        ),
        "evidence_files": {
            str(path): sha256_file(path)
            for path in evidence_files
        },
    }

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
    print(
        "Break-even cost analysis complete"
    )

    print()
    print("Model break-even summary")

    print(
        model_summary[
            [
                "model",
                "pooled_break_even_cost_bps",
                (
                    "equal_weighted_"
                    "break_even_cost_bps"
                ),
                (
                    "worst_individual_"
                    "break_even_cost_bps"
                ),
                (
                    "median_individual_"
                    "break_even_cost_bps"
                ),
                (
                    "positive_cohort_"
                    "fraction_at_1bps"
                ),
                (
                    "positive_cohort_"
                    "fraction_at_5bps"
                ),
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
