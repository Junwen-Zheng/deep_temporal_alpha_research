from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.evaluation.nulls import (
    RankedPredictionPanel,
    build_ranked_prediction_panel,
    cross_sectional_rank_ic,
    empirical_two_sided_p_value,
    permute_rows,
    shifted_cross_sectional_rank_ic,
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


def load_model_folds(
    model_name: str,
    prediction_directory: Path,
    expected_symbol_count: int,
) -> tuple[
    dict[int, RankedPredictionPanel],
    dict[int, pd.DataFrame],
    list[dict[str, Any]],
]:
    panels = {}
    frames = {}
    source_records = []

    for fold in [1, 2, 3]:
        path = (
            prediction_directory
            / (
                f"{model_name}_fold_{fold}_"
                "test.parquet"
            )
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_parquet(path)

        panel = build_ranked_prediction_panel(
            frame=frame,
            expected_symbol_count=(
                expected_symbol_count
            ),
        )

        frames[fold] = (
            frame.sort_values(
                ["timestamp", "symbol"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        panels[fold] = panel

        source_records.append(
            {
                "model": model_name,
                "fold": fold,
                "path": str(path),
                "rows": len(frame),
                "timestamps": (
                    panel.timestamp_count
                ),
                "sha256": sha256_file(path),
            }
        )

    return panels, frames, source_records


def weighted_fold_mean(
    fold_values: list[float],
    fold_counts: list[int],
) -> float:
    return float(
        np.average(
            np.asarray(
                fold_values,
                dtype=np.float64,
            ),
            weights=np.asarray(
                fold_counts,
                dtype=np.float64,
            ),
        )
    )



def finite_rank_ic_mean(
    values: np.ndarray,
    context: str,
) -> tuple[float, int]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = array[
        np.isfinite(array)
    ]

    if len(valid) == 0:
        raise ValueError(
            "No valid Rank IC timestamps for "
            f"{context}"
        )

    return float(valid.mean()), len(valid)

def main() -> None:
    null_config_path = Path(
        "configs/null_tests.yaml"
    )

    null_config = load_yaml(
        null_config_path
    )["null_tests"]

    evaluation_config_path = Path(
        null_config["evaluation_config"]
    )

    evaluation_config = load_yaml(
        evaluation_config_path
    )["evaluation"]

    source_metrics_path = Path(
        null_config["source_oos_metrics"]
    )

    source_metrics = pd.read_csv(
        source_metrics_path
    )

    expected_symbol_count = int(
        null_config[
            "expected_symbol_count"
        ]
    )

    permutation_count = int(
        null_config["permutations"]
    )

    seed = int(null_config["seed"])

    time_shifts = [
        int(value)
        for value in null_config[
            "time_shifts_bars"
        ]
    ]

    interval_minutes = int(
        evaluation_config[
            "interval_minutes"
        ]
    )

    evidence_directory = Path(
        null_config["evidence_dir"]
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_names = [
        str(name)
        for name in evaluation_config[
            "models"
        ]
    ]

    model_panels: dict[
        str,
        dict[int, RankedPredictionPanel],
    ] = {}

    model_frames: dict[
        str,
        dict[int, pd.DataFrame],
    ] = {}

    prediction_records = []

    for model_name in model_names:
        print(
            f"Loading {model_name}",
            flush=True,
        )

        model_configuration = (
            evaluation_config["models"][
                model_name
            ]
        )

        (
            panels,
            frames,
            records,
        ) = load_model_folds(
            model_name=model_name,
            prediction_directory=Path(
                model_configuration["directory"]
            ),
            expected_symbol_count=(
                expected_symbol_count
            ),
        )

        model_panels[model_name] = panels
        model_frames[model_name] = frames
        prediction_records.extend(records)

    reference_model = model_names[0]

    for model_name in model_names[1:]:
        for fold in [1, 2, 3]:
            reference = model_panels[
                reference_model
            ][fold]

            candidate = model_panels[
                model_name
            ][fold]

            if not reference.timestamps.equals(
                candidate.timestamps
            ):
                raise ValueError(
                    f"Timestamp alignment differs "
                    f"for {model_name}, fold {fold}"
                )

            if reference.symbols != candidate.symbols:
                raise ValueError(
                    f"Symbol alignment differs "
                    f"for {model_name}, fold {fold}"
                )

            if not np.array_equal(
                reference.target_ranks,
                candidate.target_ranks,
            ):
                raise ValueError(
                    f"Target ranks differ "
                    f"for {model_name}, fold {fold}"
                )

    observed_values = {}
    observed_fold_rows = []

    source_metric_lookup = (
        source_metrics.set_index("model")[
            "mean_rank_ic"
        ].to_dict()
    )

    for model_name in model_names:
        fold_means = []
        fold_counts = []

        for fold in [1, 2, 3]:
            panel = model_panels[
                model_name
            ][fold]

            rank_ic = cross_sectional_rank_ic(
                panel.prediction_ranks,
                panel.target_ranks,
            )

            fold_mean, valid_count = (
                finite_rank_ic_mean(
                    rank_ic,
                    context=(
                        f"{model_name} fold {fold} "
                        "observed evaluation"
                    ),
                )
            )

            fold_means.append(fold_mean)
            fold_counts.append(valid_count)

            observed_fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "mean_rank_ic": fold_mean,
                    "n_timestamps": len(rank_ic),
                }
            )

        observed = weighted_fold_mean(
            fold_means,
            fold_counts,
        )

        expected_observed = float(
            source_metric_lookup[model_name]
        )

        if not np.isclose(
            observed,
            expected_observed,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"Observed IC differs from source "
                f"metrics for {model_name}: "
                f"{observed} versus "
                f"{expected_observed}"
            )

        observed_values[model_name] = observed

    print(
        "Running target-permutation nulls",
        flush=True,
    )

    permutation_rows = []

    for permutation in range(
        1,
        permutation_count + 1,
    ):
        permuted_targets = {}

        for fold in [1, 2, 3]:
            reference = model_panels[
                reference_model
            ][fold]

            generator = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        seed,
                        permutation,
                        fold,
                    ]
                )
            )

            permuted_targets[fold] = (
                permute_rows(
                    reference.target_ranks,
                    generator=generator,
                )
            )

        for model_name in model_names:
            fold_means = []
            fold_counts = []

            for fold in [1, 2, 3]:
                panel = model_panels[
                    model_name
                ][fold]

                rank_ic = (
                    cross_sectional_rank_ic(
                        panel.prediction_ranks,
                        permuted_targets[fold],
                    )
                )

                (
                    fold_mean,
                    valid_count,
                ) = finite_rank_ic_mean(
                    rank_ic,
                    context=(
                        f"{model_name} fold {fold} "
                        "null evaluation"
                    ),
                )

                fold_means.append(fold_mean)
                fold_counts.append(valid_count)

            permutation_rows.append(
                {
                    "model": model_name,
                    "permutation": permutation,
                    "mean_rank_ic": (
                        weighted_fold_mean(
                            fold_means,
                            fold_counts,
                        )
                    ),
                    "n_timestamps": int(
                        sum(fold_counts)
                    ),
                }
            )

    permutation_metrics = pd.DataFrame(
        permutation_rows
    ).sort_values(
        ["model", "permutation"]
    ).reset_index(drop=True)

    summary_rows = []

    for model_name in model_names:
        null_values = (
            permutation_metrics.loc[
                permutation_metrics["model"]
                == model_name,
                "mean_rank_ic",
            ]
            .to_numpy(dtype=np.float64)
        )

        observed = observed_values[
            model_name
        ]

        null_mean = float(
            null_values.mean()
        )

        null_standard_deviation = float(
            null_values.std(ddof=1)
        )

        if null_standard_deviation <= 0:
            raise ValueError(
                f"Degenerate null distribution "
                f"for {model_name}"
            )

        summary_rows.append(
            {
                "model": model_name,
                "observed_mean_rank_ic": (
                    observed
                ),
                "null_mean_rank_ic": (
                    null_mean
                ),
                "null_rank_ic_std": (
                    null_standard_deviation
                ),
                "null_min_rank_ic": float(
                    null_values.min()
                ),
                "null_max_rank_ic": float(
                    null_values.max()
                ),
                "observed_null_zscore": (
                    (
                        observed
                        - null_mean
                    )
                    / null_standard_deviation
                ),
                "empirical_two_sided_p_value": (
                    empirical_two_sided_p_value(
                        observed_value=observed,
                        null_values=null_values,
                    )
                ),
                "permutations": (
                    permutation_count
                ),
                "n_timestamps": int(
                    permutation_metrics.loc[
                        permutation_metrics[
                            "model"
                        ]
                        == model_name,
                        "n_timestamps",
                    ].iloc[0]
                ),
            }
        )

    null_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            "observed_mean_rank_ic",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "Running temporal alignment shifts",
        flush=True,
    )

    shift_rows = []

    for model_name in model_names:
        for shift_bars in time_shifts:
            fold_means = []
            fold_counts = []

            for fold in [1, 2, 3]:
                panel = model_panels[
                    model_name
                ][fold]

                rank_ic = (
                    shifted_cross_sectional_rank_ic(
                        panel.prediction_ranks,
                        panel.target_ranks,
                        shift_bars=shift_bars,
                    )
                )

                (
                    fold_mean,
                    valid_count,
                ) = finite_rank_ic_mean(
                    rank_ic,
                    context=(
                        f"{model_name} fold {fold} "
                        "null evaluation"
                    ),
                )

                fold_means.append(fold_mean)
                fold_counts.append(valid_count)

            shift_rows.append(
                {
                    "model": model_name,
                    "shift_bars": shift_bars,
                    "shift_minutes": (
                        shift_bars
                        * interval_minutes
                    ),
                    "mean_rank_ic": (
                        weighted_fold_mean(
                            fold_means,
                            fold_counts,
                        )
                    ),
                    "mean_fold_rank_ic": (
                        float(
                            np.mean(fold_means)
                        )
                    ),
                    "worst_fold_rank_ic": (
                        float(
                            np.min(fold_means)
                        )
                    ),
                    "best_fold_rank_ic": (
                        float(
                            np.max(fold_means)
                        )
                    ),
                    "n_timestamps": int(
                        sum(fold_counts)
                    ),
                }
            )

    time_shift_metrics = (
        pd.DataFrame(shift_rows)
        .sort_values(
            ["model", "shift_bars"]
        )
        .reset_index(drop=True)
    )

    zero_shift = (
        time_shift_metrics.loc[
            time_shift_metrics[
                "shift_bars"
            ]
            == 0,
            [
                "model",
                "mean_rank_ic",
            ],
        ]
        .set_index("model")[
            "mean_rank_ic"
        ]
        .to_dict()
    )

    for model_name in model_names:
        if not np.isclose(
            zero_shift[model_name],
            observed_values[model_name],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Zero-shift IC mismatch "
                f"for {model_name}"
            )

    momentum_maximum_error = 0.0

    for fold in [1, 2, 3]:
        momentum = model_frames[
            "momentum_12"
        ][fold]

        reversal = model_frames[
            "reversal_12"
        ][fold]

        for column in [
            "timestamp",
            "symbol",
        ]:
            if not np.array_equal(
                momentum[column].to_numpy(),
                reversal[column].to_numpy(),
            ):
                raise ValueError(
                    "Momentum/reversal alignment "
                    f"differs in {column}"
                )

        prediction_sum = (
            momentum["prediction"].to_numpy(
                dtype=np.float64
            )
            + reversal["prediction"].to_numpy(
                dtype=np.float64
            )
        )

        momentum_maximum_error = max(
            momentum_maximum_error,
            float(
                np.abs(
                    prediction_sum
                ).max()
            ),
        )

    alignment_checks = {
        "schema_version": 1,
        "models": model_names,
        "fold_count": 3,
        "expected_symbol_count": (
            expected_symbol_count
        ),
        "prediction_rows_per_model": (
            int(
                sum(
                    record["rows"]
                    for record
                    in prediction_records
                    if record["model"]
                    == reference_model
                )
            )
        ),
        "prediction_timestamps_per_model": (
            int(
                sum(
                    model_panels[
                        reference_model
                    ][fold].timestamp_count
                    for fold in [1, 2, 3]
                )
            )
        ),
        "targets_identical_across_models": True,
        "timestamps_identical_across_models": True,
        "symbols_identical_across_models": True,
        "momentum_reversal_max_abs_sum": (
            momentum_maximum_error
        ),
        "momentum_observed_mean_rank_ic": (
            observed_values["momentum_12"]
        ),
        "reversal_observed_mean_rank_ic": (
            observed_values["reversal_12"]
        ),
    }

    if momentum_maximum_error > 1e-7:
        raise ValueError(
            "Momentum and reversal predictions "
            "are not exact opposites"
        )

    if not np.isclose(
        alignment_checks[
            "momentum_observed_mean_rank_ic"
        ],
        -alignment_checks[
            "reversal_observed_mean_rank_ic"
        ],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Momentum and reversal IC values "
            "are not exact opposites"
        )

    observed_fold_metrics = (
        pd.DataFrame(
            observed_fold_rows
        )
        .sort_values(
            ["model", "fold"]
        )
        .reset_index(drop=True)
    )

    observed_fold_path = (
        evidence_directory
        / "null_observed_fold_metrics.csv"
    )

    permutation_path = (
        evidence_directory
        / "null_permutation_metrics.csv"
    )

    summary_path = (
        evidence_directory
        / "null_test_summary.csv"
    )

    time_shift_path = (
        evidence_directory
        / "null_time_shift_metrics.csv"
    )

    alignment_path = (
        evidence_directory
        / "null_alignment_checks.json"
    )

    write_csv(
        observed_fold_metrics,
        observed_fold_path,
    )

    write_csv(
        permutation_metrics,
        permutation_path,
    )

    write_csv(
        null_summary,
        summary_path,
    )

    write_csv(
        time_shift_metrics,
        time_shift_path,
    )

    alignment_path.write_text(
        json.dumps(
            alignment_checks,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = (
        evidence_directory
        / "null_test_manifest.json"
    )

    evidence_files = [
        observed_fold_path,
        permutation_path,
        summary_path,
        time_shift_path,
        alignment_path,
    ]

    manifest = {
        "schema_version": 1,
        "null_config": str(
            null_config_path
        ),
        "null_config_sha256": sha256_file(
            null_config_path
        ),
        "evaluation_config": str(
            evaluation_config_path
        ),
        "evaluation_config_sha256": (
            sha256_file(
                evaluation_config_path
            )
        ),
        "source_oos_metrics": str(
            source_metrics_path
        ),
        "source_oos_metrics_sha256": (
            sha256_file(
                source_metrics_path
            )
        ),
        "seed": seed,
        "permutations": permutation_count,
        "expected_symbol_count": (
            expected_symbol_count
        ),
        "time_shifts_bars": time_shifts,
        "models": model_names,
        "prediction_files": sorted(
            prediction_records,
            key=lambda record: (
                record["model"],
                record["fold"],
            ),
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
    print("Null-test research complete")

    print()
    print("Permutation-null summary")

    print(
        null_summary[
            [
                "model",
                "observed_mean_rank_ic",
                "null_mean_rank_ic",
                "null_rank_ic_std",
                "observed_null_zscore",
                "empirical_two_sided_p_value",
            ]
        ].to_string(index=False)
    )

    print()
    print("Zero-shift alignment verified")
    print(
        "Momentum/reversal maximum error:",
        momentum_maximum_error,
    )


if __name__ == "__main__":
    main()
