from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.evaluation.rebalance import (
    build_rebalance_cadence_summary,
    build_rebalance_phase_map,
    build_rebalance_phase_metrics,
    validate_rebalance_cadences,
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
    sensitivity_config_path = Path(
        "configs/rebalance_sensitivity.yaml"
    )

    sensitivity_config = load_yaml(
        sensitivity_config_path
    )["rebalance_sensitivity"]

    evaluation_config_path = Path(
        sensitivity_config[
            "evaluation_config"
        ]
    )

    evaluation_config = load_yaml(
        evaluation_config_path
    )["evaluation"]

    source_cohort_path = Path(
        sensitivity_config[
            "source_cohort_metrics"
        ]
    )

    source_summary_path = Path(
        sensitivity_config[
            "source_model_summary"
        ]
    )

    source_manifest_path = Path(
        sensitivity_config[
            "source_break_even_manifest"
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

    holding_bars = int(
        evaluation_config[
            "holding_bars"
        ]
    )

    interval_minutes = int(
        evaluation_config[
            "interval_minutes"
        ]
    )

    annualization_periods = int(
        evaluation_config[
            "annualization_periods"
        ]
    )

    cadences = validate_rebalance_cadences(
        cadence_bars=[
            int(value)
            for value in sensitivity_config[
                "cadence_bars"
            ]
        ],
        holding_bars=holding_bars,
    )

    cost_levels = [
        float(value)
        for value in sensitivity_config[
            "cost_bps"
        ]
    ]

    phase_map = build_rebalance_phase_map(
        cadence_bars=cadences,
        holding_bars=holding_bars,
        interval_minutes=interval_minutes,
    )

    phase_metrics = (
        build_rebalance_phase_metrics(
            cohort_metrics=source_cohorts,
            cadence_bars=cadences,
            cost_levels_bps=cost_levels,
            holding_bars=holding_bars,
            interval_minutes=(
                interval_minutes
            ),
            annualization_periods=(
                annualization_periods
            ),
        )
    )

    cadence_summary = (
        build_rebalance_cadence_summary(
            phase_metrics
        )
    )

    print(
        "Verifying five-minute aggregation "
        "against break-even summary",
        flush=True,
    )

    five_minute = phase_metrics.loc[
        (
            phase_metrics[
                "cadence_bars"
            ]
            == 1
        )
        & np.isclose(
            phase_metrics[
                "cost_bps"
            ].to_numpy(
                dtype=np.float64
            ),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
    ][
        [
            "model",
            "break_even_cost_bps",
        ]
    ].rename(
        columns={
            "break_even_cost_bps": (
                "reconstructed_break_even"
            )
        }
    )

    comparison = source_summary[
        [
            "model",
            "pooled_break_even_cost_bps",
        ]
    ].merge(
        five_minute,
        on="model",
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != len(
        source_summary
    ):
        raise ValueError(
            "Five-minute model coverage differs "
            "from the source summary"
        )

    if not np.allclose(
        comparison[
            "pooled_break_even_cost_bps"
        ].to_numpy(),
        comparison[
            "reconstructed_break_even"
        ].to_numpy(),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Five-minute pooled break-even "
            "does not reproduce source evidence"
        )

    print(
        "Verifying hourly phases against "
        "individual cohort thresholds",
        flush=True,
    )

    hourly = phase_metrics.loc[
        (
            phase_metrics[
                "cadence_bars"
            ]
            == holding_bars
        )
        & np.isclose(
            phase_metrics[
                "cost_bps"
            ].to_numpy(
                dtype=np.float64
            ),
            0.0,
            rtol=0.0,
            atol=1e-12,
        ),
        [
            "model",
            "phase",
            "break_even_cost_bps",
        ],
    ].rename(
        columns={
            "phase": "cohort",
            "break_even_cost_bps": (
                "reconstructed_break_even"
            ),
        }
    )

    hourly_comparison = source_cohorts[
        [
            "model",
            "cohort",
            "break_even_cost_bps",
        ]
    ].merge(
        hourly,
        on=[
            "model",
            "cohort",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(hourly_comparison) != len(
        source_cohorts
    ):
        raise ValueError(
            "Hourly phase coverage differs "
            "from source cohort evidence"
        )

    if not np.allclose(
        hourly_comparison[
            "break_even_cost_bps"
        ].to_numpy(),
        hourly_comparison[
            "reconstructed_break_even"
        ].to_numpy(),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Hourly phases do not reproduce "
            "individual cohort thresholds"
        )

    for _, group in phase_metrics.groupby(
        [
            "model",
            "cadence_bars",
            "phase",
        ],
        sort=True,
    ):
        ordered = group.sort_values(
            "cost_bps"
        )

        net_returns = ordered[
            "pooled_net_mean_return"
        ].to_numpy(dtype=np.float64)

        if np.any(
            np.diff(net_returns) > 1e-15
        ):
            raise ValueError(
                "Net return increases as "
                "transaction cost rises"
            )

    expected_models = [
        str(model)
        for model in evaluation_config[
            "models"
        ]
    ]

    if set(
        phase_metrics["model"]
    ) != set(expected_models):
        raise ValueError(
            "Sensitivity model coverage differs "
            "from evaluation configuration"
        )

    if (
        source_manifest[
            "cohort_count_per_model"
        ]
        != holding_bars
    ):
        raise ValueError(
            "Source cohort count differs "
            "from the holding period"
        )

    evidence_directory = Path(
        sensitivity_config[
            "evidence_dir"
        ]
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase_map_path = (
        evidence_directory
        / "rebalance_phase_map.csv"
    )

    phase_metrics_path = (
        evidence_directory
        / "rebalance_phase_metrics.csv"
    )

    cadence_summary_path = (
        evidence_directory
        / "rebalance_cadence_summary.csv"
    )

    write_csv(
        phase_map,
        phase_map_path,
    )

    write_csv(
        phase_metrics,
        phase_metrics_path,
    )

    write_csv(
        cadence_summary,
        cadence_summary_path,
    )

    manifest_path = (
        evidence_directory
        / "rebalance_sensitivity_manifest.json"
    )

    evidence_files = [
        phase_map_path,
        phase_metrics_path,
        cadence_summary_path,
    ]

    manifest = {
        "schema_version": 1,
        "sensitivity_config": str(
            sensitivity_config_path
        ),
        "sensitivity_config_sha256": (
            sha256_file(
                sensitivity_config_path
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
        "source_model_summary": str(
            source_summary_path
        ),
        "source_model_summary_sha256": (
            sha256_file(
                source_summary_path
            )
        ),
        "source_break_even_manifest": str(
            source_manifest_path
        ),
        "source_break_even_manifest_sha256": (
            sha256_file(
                source_manifest_path
            )
        ),
        "models": expected_models,
        "holding_bars": holding_bars,
        "holding_minutes": (
            holding_bars
            * interval_minutes
        ),
        "interval_minutes": (
            interval_minutes
        ),
        "annualization_periods": (
            annualization_periods
        ),
        "cadence_bars": list(
            cadences
        ),
        "cadence_minutes": [
            value * interval_minutes
            for value in cadences
        ],
        "cost_bps": cost_levels,
        "phase_count_total": int(
            phase_map["phase"].count()
        ),
        "method": (
            "equal-capital staggered hourly "
            "sleeves grouped by signal-refresh "
            "cadence and clock phase"
        ),
        "sharpe_aggregation_performed": False,
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
        "Rebalance sensitivity analysis "
        "complete"
    )

    print()
    print(
        "One-basis-point cadence summary"
    )

    print(
        cadence_summary.loc[
            np.isclose(
                cadence_summary[
                    "cost_bps"
                ].to_numpy(
                    dtype=np.float64
                ),
                1.0,
                rtol=0.0,
                atol=1e-12,
            ),
            [
                "model",
                "cadence_minutes",
                "sleeve_count",
                "mean_phase_break_even_cost_bps",
                "worst_phase_break_even_cost_bps",
                "mean_pooled_annualized_net_return",
                "worst_pooled_annualized_net_return",
                "positive_phase_fraction",
            ],
        ]
        .sort_values(
            [
                "cadence_minutes",
                "mean_phase_break_even_cost_bps",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
