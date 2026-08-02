from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn

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
from deep_alpha.models.mlp import FeatureMLP
from deep_alpha.training.engine import (
    iterate_minibatches,
    predict_in_batches,
    select_device,
    set_global_seed,
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

    result["prediction"] = predictions
    return result


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
    gradient_clip_norm: float,
) -> float:
    model.train()

    total_loss = 0.0
    total_rows = 0

    for batch_features, batch_targets in iterate_minibatches(
        features=features,
        targets=targets,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    ):
        feature_tensor = torch.from_numpy(
            batch_features
        ).to(device=device)

        target_tensor = torch.from_numpy(
            batch_targets
        ).to(device=device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(feature_tensor)
        loss = loss_function(predictions, target_tensor)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

        optimizer.step()

        batch_rows = len(batch_features)

        total_loss += float(loss.detach().cpu()) * batch_rows
        total_rows += batch_rows

    return total_loss / total_rows


def mean_squared_error(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    errors = predictions - targets

    return float(np.mean(errors**2))


def evaluate_and_save(
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
            "model": "mlp",
            "fold": fold_id,
            "split": split_name,
        }
    )

    destination = (
        prediction_directory
        / f"mlp_fold_{fold_id}_{split_name}.parquet"
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

    record = {
        "model": "mlp",
        "fold": fold_id,
        "split": split_name,
        "path": str(destination),
        "rows": len(prediction_frame),
        "sha256": sha256_file(destination),
    }

    return metrics, record


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


def build_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["model", "split"],
            as_index=False,
            sort=True,
        )
        .agg(
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
        .sort_values(["model", "split"])
        .reset_index(drop=True)
    )


def main() -> None:
    research_config_path = Path(
        "configs/research.yaml"
    )

    neural_config_path = Path(
        "configs/neural.yaml"
    )

    research_config = load_yaml(
        research_config_path
    )["research"]

    neural_config = load_yaml(
        neural_config_path
    )["neural"]

    mlp_config = neural_config["mlp"]

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

    panel_columns = [
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
        columns=panel_columns,
    )

    seed = int(neural_config["seed"])

    set_global_seed(seed)

    device = select_device(
        str(neural_config["device"])
    )

    print(f"Training device: {device}", flush=True)

    prediction_directory = Path(
        neural_config["prediction_dir"]
    )

    model_directory = Path(
        neural_config["model_dir"]
    )

    evidence_directory = Path(
        neural_config["evidence_dir"]
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracking_database = Path(
        neural_config["mlflow"]["database_path"]
    ).resolve()

    tracking_database.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_directory = Path(
        neural_config["mlflow"]["artifact_dir"]
    ).resolve()

    artifact_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracking_uri = (
        f"sqlite:///{tracking_database.as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = str(
        neural_config["mlflow"][
            "experiment_name"
        ]
    )

    existing_experiment = (
        mlflow.get_experiment_by_name(
            experiment_name
        )
    )

    if existing_experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=(
                artifact_directory.as_uri()
            ),
        )

    mlflow.set_experiment(experiment_name)

    hidden_dims = [
        int(value)
        for value in mlp_config["hidden_dims"]
    ]

    batch_size = int(mlp_config["batch_size"])
    max_epochs = int(mlp_config["max_epochs"])
    patience = int(mlp_config["patience"])

    minimum_improvement = float(
        mlp_config["minimum_improvement"]
    )

    learning_rate = float(
        mlp_config["learning_rate"]
    )

    weight_decay = float(
        mlp_config["weight_decay"]
    )

    dropout = float(mlp_config["dropout"])

    gradient_clip_norm = float(
        mlp_config["gradient_clip_norm"]
    )

    metrics_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []

    for fold_configuration in research_config["folds"]:
        fold = parse_fold_definition(
            fold_configuration
        )

        fold_seed = seed + fold.fold_id

        set_global_seed(fold_seed)

        print()
        print(
            f"=== MLP fold {fold.fold_id} ===",
            flush=True,
        )

        masks = build_fold_masks(panel, fold)

        train = panel.loc[masks.train]
        validation = panel.loc[masks.validation]
        test = panel.loc[masks.test]

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

        train_features = scaler.transform(train)
        validation_features = scaler.transform(
            validation
        )
        test_features = scaler.transform(test)

        train_targets = train[
            "target"
        ].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        validation_targets = validation[
            "target"
        ].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        model = FeatureMLP(
            feature_count=len(FEATURE_COLUMNS),
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        loss_function = nn.MSELoss()

        best_score = -np.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        fold_history: list[dict[str, Any]] = []

        with mlflow.start_run(
            run_name=f"mlp_fold_{fold.fold_id}"
        ) as run:
            run_id = run.info.run_id

            mlflow.log_params(
                {
                    "fold": fold.fold_id,
                    "seed": fold_seed,
                    "device": str(device),
                    "feature_count": len(
                        FEATURE_COLUMNS
                    ),
                    "hidden_dims": ",".join(
                        str(value)
                        for value in hidden_dims
                    ),
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "gradient_clip_norm": (
                        gradient_clip_norm
                    ),
                    "selection_metric": (
                        "validation_mean_rank_ic"
                    ),
                }
            )

            for epoch in range(1, max_epochs + 1):
                train_loss = train_epoch(
                    model=model,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    features=train_features,
                    targets=train_targets,
                    batch_size=batch_size,
                    device=device,
                    seed=fold_seed + epoch,
                    gradient_clip_norm=(
                        gradient_clip_norm
                    ),
                )

                validation_predictions = (
                    predict_in_batches(
                        model=model,
                        features=validation_features,
                        batch_size=batch_size,
                        device=device,
                    )
                )

                validation_frame = (
                    build_prediction_frame(
                        source=validation,
                        predictions=(
                            validation_predictions
                        ),
                    )
                )

                validation_metrics = (
                    summarize_predictions(
                        validation_frame
                    )
                )

                validation_loss = mean_squared_error(
                    predictions=validation_predictions,
                    targets=validation_targets,
                )

                validation_rank_ic = float(
                    validation_metrics[
                        "mean_rank_ic"
                    ]
                )

                row = {
                    "model": "mlp",
                    "fold": fold.fold_id,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": (
                        validation_loss
                    ),
                    "validation_mean_rank_ic": (
                        validation_rank_ic
                    ),
                    "selected": False,
                }

                fold_history.append(row)

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "validation_loss": (
                            validation_loss
                        ),
                        "validation_mean_rank_ic": (
                            validation_rank_ic
                        ),
                    },
                    step=epoch,
                )

                print(
                    f"epoch={epoch} "
                    f"train_loss={train_loss:.8f} "
                    f"validation_loss="
                    f"{validation_loss:.8f} "
                    f"validation_rank_ic="
                    f"{validation_rank_ic:.6f}",
                    flush=True,
                )

                improved = (
                    validation_rank_ic
                    > best_score + minimum_improvement
                )

                if improved:
                    best_score = validation_rank_ic
                    best_epoch = epoch

                    best_state = {
                        name: tensor.detach()
                        .cpu()
                        .clone()
                        for name, tensor
                        in model.state_dict().items()
                    }

                    stale_epochs = 0
                else:
                    stale_epochs += 1

                if stale_epochs >= patience:
                    print(
                        "Early stopping",
                        flush=True,
                    )
                    break

            if best_state is None:
                raise RuntimeError(
                    "No MLP checkpoint was selected"
                )

            for row in fold_history:
                row["selected"] = (
                    row["epoch"] == best_epoch
                )

            history_rows.extend(fold_history)

            model.load_state_dict(best_state)

            validation_predictions = (
                predict_in_batches(
                    model=model,
                    features=validation_features,
                    batch_size=batch_size,
                    device=device,
                )
            )

            test_predictions = predict_in_batches(
                model=model,
                features=test_features,
                batch_size=batch_size,
                device=device,
            )

            for split_name, source, predictions in [
                (
                    "validation",
                    validation,
                    validation_predictions,
                ),
                (
                    "test",
                    test,
                    test_predictions,
                ),
            ]:
                metrics, prediction_record = (
                    evaluate_and_save(
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

            checkpoint_path = (
                model_directory
                / f"mlp_fold_{fold.fold_id}.pt"
            )

            torch.save(
                {
                    "model": "mlp",
                    "fold": fold.fold_id,
                    "best_epoch": best_epoch,
                    "best_validation_mean_rank_ic": (
                        best_score
                    ),
                    "feature_columns": FEATURE_COLUMNS,
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                    "state_dict": best_state,
                    "scaler": scaler.to_dict(),
                },
                checkpoint_path,
            )

            model_records.append(
                {
                    "fold": fold.fold_id,
                    "path": str(checkpoint_path),
                    "sha256": sha256_file(
                        checkpoint_path
                    ),
                    "best_epoch": best_epoch,
                    "best_validation_mean_rank_ic": (
                        best_score
                    ),
                    "mlflow_run_id": run_id,
                    "train_rows": len(train),
                    "validation_rows": len(
                        validation
                    ),
                    "test_rows": len(test),
                    "scaler": scaler.to_dict(),
                }
            )

            mlflow.log_metrics(
                {
                    "selected_epoch": best_epoch,
                    "selected_validation_rank_ic": (
                        best_score
                    ),
                }
            )

        del (
            model,
            optimizer,
            train_features,
            validation_features,
            test_features,
            train_targets,
            validation_targets,
        )

        if device.type == "mps":
            torch.mps.empty_cache()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_frame = (
        pd.DataFrame(metrics_rows)
        [
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
        ]
        .sort_values(
            ["fold", "split"]
        )
        .reset_index(drop=True)
    )

    history_frame = (
        pd.DataFrame(history_rows)
        .sort_values(["fold", "epoch"])
        .reset_index(drop=True)
    )

    summary_frame = build_summary(
        metrics_frame
    )

    metrics_path = (
        evidence_directory
        / "mlp_fold_metrics.csv"
    )

    history_path = (
        evidence_directory
        / "mlp_training_history.csv"
    )

    summary_path = (
        evidence_directory
        / "mlp_summary.csv"
    )

    write_csv(metrics_frame, metrics_path)
    write_csv(history_frame, history_path)
    write_csv(summary_frame, summary_path)

    manifest_path = (
        evidence_directory
        / "mlp_run_manifest.json"
    )

    manifest = {
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
        "neural_config": str(
            neural_config_path
        ),
        "neural_config_sha256": sha256_file(
            neural_config_path
        ),
        "feature_columns": FEATURE_COLUMNS,
        "device": str(device),
        "model_files": model_records,
        "prediction_files": sorted(
            prediction_records,
            key=lambda record: (
                record["fold"],
                record["split"],
            ),
        ),
        "evidence_files": {
            str(metrics_path): sha256_file(
                metrics_path
            ),
            str(history_path): sha256_file(
                history_path
            ),
            str(summary_path): sha256_file(
                summary_path
            ),
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
    print("MLP research complete")
    print(
        metrics_frame[
            [
                "fold",
                "split",
                "mean_rank_ic",
                "rank_ic_ir",
            ]
        ].to_string(index=False)
    )

    print()
    print("Cross-fold MLP summary")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
