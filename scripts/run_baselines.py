from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.data.features import FEATURE_COLUMNS
from deep_alpha.data.scaling import RobustFeatureScaler
from deep_alpha.data.splits import (
    build_fold_masks,
    parse_fold_definition,
)
from deep_alpha.evaluation.metrics import (
    summarize_predictions,
)
from deep_alpha.models.baselines import (
    fit_lightgbm,
    fit_ridge,
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


def build_prediction_frame(
    source: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    if len(source) != len(predictions):
        raise ValueError(
            "Prediction length does not match source rows"
        )

    result = source[
        [
            "timestamp",
            "symbol",
            "raw_target",
            "target",
        ]
    ].copy()

    result["prediction"] = np.asarray(
        predictions,
        dtype=np.float64,
    )

    return result


def evaluate_and_save(
    model_name: str,
    fold_id: int,
    split_name: str,
    source: pd.DataFrame,
    predictions: np.ndarray,
    prediction_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction_frame = build_prediction_frame(
        source=source,
        predictions=predictions,
    )

    metrics = summarize_predictions(
        prediction_frame
    )

    metrics.update(
        {
            "model": model_name,
            "fold": fold_id,
            "split": split_name,
        }
    )

    destination = (
        prediction_directory
        / (
            f"{model_name}_fold_{fold_id}_"
            f"{split_name}.parquet"
        )
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_frame.to_parquet(
        destination,
        index=False,
        compression="zstd",
    )

    prediction_record = {
        "model": model_name,
        "fold": fold_id,
        "split": split_name,
        "path": str(destination),
        "rows": len(prediction_frame),
        "sha256": sha256_file(destination),
    }

    return metrics, prediction_record


def select_ridge_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_frame: pd.DataFrame,
    alphas: list[float],
    fold_id: int,
) -> tuple[
    object,
    np.ndarray,
    float,
    list[dict[str, Any]],
]:
    candidates = []

    for alpha in alphas:
        model = fit_ridge(
            features=train_features,
            targets=train_targets,
            alpha=alpha,
        )

        validation_predictions = model.predict(
            validation_features
        )

        validation_prediction_frame = (
            build_prediction_frame(
                source=validation_frame,
                predictions=validation_predictions,
            )
        )

        validation_metrics = summarize_predictions(
            validation_prediction_frame
        )

        candidates.append(
            {
                "alpha": alpha,
                "model": model,
                "predictions": validation_predictions,
                "mean_rank_ic": validation_metrics[
                    "mean_rank_ic"
                ],
            }
        )

    selected = max(
        candidates,
        key=lambda candidate: candidate[
            "mean_rank_ic"
        ],
    )

    selection_rows = []

    for candidate in candidates:
        selection_rows.append(
            {
                "model": "ridge",
                "fold": fold_id,
                "candidate": (
                    f"alpha={candidate['alpha']}"
                ),
                "validation_mean_rank_ic": candidate[
                    "mean_rank_ic"
                ],
                "selected": (
                    candidate["alpha"]
                    == selected["alpha"]
                ),
            }
        )

    return (
        selected["model"],
        selected["predictions"],
        float(selected["alpha"]),
        selection_rows,
    )


def build_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    grouped = metrics.groupby(
        ["model", "split"],
        as_index=False,
        sort=True,
    )

    return grouped.agg(
        mean_fold_rank_ic=(
            "mean_rank_ic",
            "mean",
        ),
        worst_fold_rank_ic=(
            "mean_rank_ic",
            "min",
        ),
        best_fold_rank_ic=(
            "mean_rank_ic",
            "max",
        ),
        mean_rank_ic_ir=(
            "rank_ic_ir",
            "mean",
        ),
        mean_positive_timestamp_fraction=(
            "positive_timestamp_fraction",
            "mean",
        ),
        mean_rmse=("rmse", "mean"),
        mean_mae=("mae", "mean"),
        total_rows=("n_rows", "sum"),
    )


def main() -> None:
    research_config_path = Path(
        "configs/research.yaml"
    )

    baseline_config_path = Path(
        "configs/baselines.yaml"
    )

    research_config = load_yaml(
        research_config_path
    )["research"]

    baseline_config = load_yaml(
        baseline_config_path
    )["baselines"]

    source_manifest_path = Path(
        research_config["manifest_path"]
    )

    source_manifest = json.loads(
        source_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    panel_path = Path(
        source_manifest["output_path"]
    )

    required_columns = [
        "timestamp",
        "label_end_timestamp",
        "symbol",
        "raw_target",
        "target",
        *FEATURE_COLUMNS,
    ]

    print("Loading research panel", flush=True)

    panel = pd.read_parquet(
        panel_path,
        columns=required_columns,
    )

    prediction_directory = Path(
        baseline_config["prediction_dir"]
    )

    evidence_directory = Path(
        baseline_config["evidence_dir"]
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []

    ridge_alphas = [
        float(value)
        for value in baseline_config[
            "ridge"
        ]["alphas"]
    ]

    seed = int(baseline_config["seed"])

    for fold_configuration in research_config["folds"]:
        fold = parse_fold_definition(
            fold_configuration
        )

        print()
        print(
            f"=== Fold {fold.fold_id} ===",
            flush=True,
        )

        masks = build_fold_masks(panel, fold)

        train = panel.loc[masks.train].copy()
        validation = panel.loc[
            masks.validation
        ].copy()

        test = panel.loc[masks.test].copy()

        print(
            f"train={len(train)} "
            f"validation={len(validation)} "
            f"test={len(test)}",
            flush=True,
        )

        scaler = RobustFeatureScaler.fit(
            train,
            FEATURE_COLUMNS,
        )

        train_scaled = scaler.transform(train)
        validation_scaled = scaler.transform(
            validation
        )
        test_scaled = scaler.transform(test)

        train_raw = train[
            FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        validation_raw = validation[
            FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        test_raw = test[
            FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        train_targets = train[
            "target"
        ].to_numpy(dtype=np.float64)

        validation_targets = validation[
            "target"
        ].to_numpy(dtype=np.float64)

        print("Evaluating momentum baselines")

        reference_predictions = {
            "momentum_12": {
                "validation": validation[
                    "return_12"
                ].to_numpy(dtype=np.float64),
                "test": test[
                    "return_12"
                ].to_numpy(dtype=np.float64),
            },
            "reversal_12": {
                "validation": -validation[
                    "return_12"
                ].to_numpy(dtype=np.float64),
                "test": -test[
                    "return_12"
                ].to_numpy(dtype=np.float64),
            },
        }

        for model_name, split_predictions in (
            reference_predictions.items()
        ):
            for split_name, predictions in (
                split_predictions.items()
            ):
                source = (
                    validation
                    if split_name == "validation"
                    else test
                )

                metrics, prediction_record = (
                    evaluate_and_save(
                        model_name=model_name,
                        fold_id=fold.fold_id,
                        split_name=split_name,
                        source=source,
                        predictions=predictions,
                        prediction_directory=(
                            prediction_directory
                        ),
                    )
                )

                metrics_rows.append(metrics)
                prediction_records.append(
                    prediction_record
                )

        print("Selecting Ridge alpha")

        (
            ridge_model,
            ridge_validation_predictions,
            selected_alpha,
            ridge_selection_rows,
        ) = select_ridge_model(
            train_features=train_scaled,
            train_targets=train_targets,
            validation_features=validation_scaled,
            validation_frame=validation,
            alphas=ridge_alphas,
            fold_id=fold.fold_id,
        )

        selection_rows.extend(
            ridge_selection_rows
        )

        ridge_test_predictions = (
            ridge_model.predict(test_scaled)
        )

        for split_name, source, predictions in [
            (
                "validation",
                validation,
                ridge_validation_predictions,
            ),
            (
                "test",
                test,
                ridge_test_predictions,
            ),
        ]:
            metrics, prediction_record = (
                evaluate_and_save(
                    model_name="ridge",
                    fold_id=fold.fold_id,
                    split_name=split_name,
                    source=source,
                    predictions=predictions,
                    prediction_directory=(
                        prediction_directory
                    ),
                )
            )

            metrics_rows.append(metrics)
            prediction_records.append(
                prediction_record
            )

        print(
            "Training deterministic LightGBM",
            flush=True,
        )

        lightgbm_model = fit_lightgbm(
            train_features=train_raw,
            train_targets=train_targets,
            validation_features=validation_raw,
            validation_targets=validation_targets,
            configuration=baseline_config[
                "lightgbm"
            ],
            seed=seed + fold.fold_id,
        )

        best_iteration = int(
            lightgbm_model.best_iteration_
        )

        lightgbm_validation_predictions = (
            lightgbm_model.predict(
                validation_raw,
                num_iteration=best_iteration,
            )
        )

        lightgbm_test_predictions = (
            lightgbm_model.predict(
                test_raw,
                num_iteration=best_iteration,
            )
        )

        for split_name, source, predictions in [
            (
                "validation",
                validation,
                lightgbm_validation_predictions,
            ),
            (
                "test",
                test,
                lightgbm_test_predictions,
            ),
        ]:
            metrics, prediction_record = (
                evaluate_and_save(
                    model_name="lightgbm",
                    fold_id=fold.fold_id,
                    split_name=split_name,
                    source=source,
                    predictions=predictions,
                    prediction_directory=(
                        prediction_directory
                    ),
                )
            )

            metrics_rows.append(metrics)
            prediction_records.append(
                prediction_record
            )

        fold_records.append(
            {
                "fold": fold.fold_id,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "ridge_alpha": selected_alpha,
                "lightgbm_best_iteration": (
                    best_iteration
                ),
                "scaler": scaler.to_dict(),
            }
        )

    metrics_frame = pd.DataFrame(metrics_rows)

    metrics_frame = metrics_frame[
        [
            "model",
            "fold",
            "split",
            "mean_rank_ic",
            "median_rank_ic",
            "rank_ic_std",
            "rank_ic_ir",
            "positive_timestamp_fraction",
            "rmse",
            "mae",
            "n_rows",
            "n_timestamps",
        ]
    ].sort_values(
        ["model", "fold", "split"]
    ).reset_index(drop=True)

    selection_frame = pd.DataFrame(
        selection_rows
    ).sort_values(
        ["fold", "candidate"]
    ).reset_index(drop=True)

    summary_frame = build_summary(
        metrics_frame
    ).sort_values(
        ["model", "split"]
    ).reset_index(drop=True)

    metrics_path = (
        evidence_directory
        / "baseline_fold_metrics.csv"
    )

    selection_path = (
        evidence_directory
        / "baseline_model_selection.csv"
    )

    summary_path = (
        evidence_directory
        / "baseline_summary.csv"
    )

    write_csv(metrics_frame, metrics_path)
    write_csv(selection_frame, selection_path)
    write_csv(summary_frame, summary_path)

    evidence_manifest_path = (
        evidence_directory
        / "baseline_run_manifest.json"
    )

    evidence_manifest = {
        "schema_version": 1,
        "source_research_manifest": str(
            source_manifest_path
        ),
        "source_research_manifest_sha256": (
            sha256_file(source_manifest_path)
        ),
        "research_panel": str(panel_path),
        "research_panel_sha256": sha256_file(
            panel_path
        ),
        "research_config": str(
            research_config_path
        ),
        "research_config_sha256": sha256_file(
            research_config_path
        ),
        "baseline_config": str(
            baseline_config_path
        ),
        "baseline_config_sha256": sha256_file(
            baseline_config_path
        ),
        "feature_columns": FEATURE_COLUMNS,
        "models": [
            "momentum_12",
            "reversal_12",
            "ridge",
            "lightgbm",
        ],
        "folds": fold_records,
        "prediction_files": sorted(
            prediction_records,
            key=lambda record: (
                record["model"],
                record["fold"],
                record["split"],
            ),
        ),
        "evidence_files": {
            str(metrics_path): sha256_file(
                metrics_path
            ),
            str(selection_path): sha256_file(
                selection_path
            ),
            str(summary_path): sha256_file(
                summary_path
            ),
        },
    }

    evidence_manifest_path.write_text(
        json.dumps(
            evidence_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Baseline research complete")
    print(
        metrics_frame[
            [
                "model",
                "fold",
                "split",
                "mean_rank_ic",
            ]
        ].to_string(index=False)
    )

    print()
    print("Cross-fold summary")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
