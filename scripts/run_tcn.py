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
from deep_alpha.data.sequences import (
    SequenceSplit,
    TemporalPanel,
    build_sequence_split,
    build_temporal_panel,
)
from deep_alpha.data.splits import (
    build_fold_masks,
    parse_fold_definition,
)
from deep_alpha.evaluation.metrics import (
    summarize_predictions,
)
from deep_alpha.models.tcn import TemporalConvNet
from deep_alpha.training.engine import (
    select_device,
    set_global_seed,
)
from deep_alpha.training.sequence_engine import (
    extract_cross_sectional_batch,
    iterate_endpoint_batches,
    predict_sequence_model,
    scale_temporal_panel,
    subsample_endpoints,
)


def build_prediction_source(
    ordered_panel: pd.DataFrame,
    temporal_panel: TemporalPanel,
    split: SequenceSplit,
) -> pd.DataFrame:
    symbol_indices = np.arange(
        temporal_panel.symbol_count,
        dtype=np.int64,
    )

    flat_indices = (
        split.endpoint_positions[:, None]
        * temporal_panel.symbol_count
        + symbol_indices[None, :]
    ).reshape(-1)

    source = ordered_panel.iloc[
        flat_indices
    ][
        [
            "timestamp",
            "symbol",
            "raw_target",
            "target",
        ]
    ].copy()

    expected_targets = temporal_panel.targets[
        split.endpoint_positions,
        :,
    ].reshape(-1)

    actual_targets = source["target"].to_numpy(
        dtype=np.float32,
    )

    if not np.allclose(
        actual_targets,
        expected_targets,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(
            "Prediction source targets are misaligned"
        )

    return source.reset_index(drop=True)


def build_prediction_frame(
    source: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    if len(source) != len(predictions):
        raise ValueError(
            "Prediction length does not match source rows"
        )

    result = source.copy()

    result["prediction"] = np.asarray(
        predictions,
        dtype=np.float64,
    )

    return result


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_scaler: torch.amp.GradScaler,
    temporal_panel: TemporalPanel,
    endpoint_positions: np.ndarray,
    sequence_length: int,
    endpoint_batch_size: int,
    device: torch.device,
    mixed_precision: bool,
    seed: int,
    gradient_clip_norm: float,
) -> float:
    model.train()

    loss_function = nn.MSELoss()

    total_loss = 0.0
    total_rows = 0

    for endpoint_batch in iterate_endpoint_batches(
        endpoint_positions=endpoint_positions,
        batch_size=endpoint_batch_size,
        shuffle=True,
        seed=seed,
    ):
        features, targets, _, _ = (
            extract_cross_sectional_batch(
                temporal_panel=temporal_panel,
                endpoint_positions=endpoint_batch,
                sequence_length=sequence_length,
            )
        )

        feature_tensor = torch.from_numpy(
            features
        ).to(device=device)

        target_tensor = torch.from_numpy(
            targets
        ).to(device=device)

        optimizer.zero_grad(set_to_none=True)

        if mixed_precision:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                predictions = model(feature_tensor)
                loss = loss_function(
                    predictions,
                    target_tensor,
                )

            gradient_scaler.scale(loss).backward()
            gradient_scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

            gradient_scaler.step(optimizer)
            gradient_scaler.update()
        else:
            predictions = model(feature_tensor)

            loss = loss_function(
                predictions,
                target_tensor,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

            optimizer.step()

        batch_rows = len(targets)

        total_loss += (
            float(loss.detach().cpu())
            * batch_rows
        )

        total_rows += batch_rows

    if total_rows == 0:
        raise RuntimeError(
            "No TCN training rows were processed"
        )

    return total_loss / total_rows


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
            "model": "tcn",
            "fold": fold_id,
            "split": split_name,
        }
    )

    destination = (
        prediction_directory
        / f"tcn_fold_{fold_id}_{split_name}.parquet"
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
        "model": "tcn",
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

    tcn_config = neural_config["tcn"]

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

    panel = (
        panel.sort_values(
            ["timestamp", "symbol"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    sequence_length = int(
        research_config["sequence_length"]
    )

    temporal_panel = build_temporal_panel(
        frame=panel,
        feature_columns=FEATURE_COLUMNS,
        expected_symbol_count=int(
            research_config[
                "expected_symbol_count"
            ]
        ),
        interval_minutes=int(
            research_config["interval_minutes"]
        ),
    )

    channel_count = int(
        tcn_config["channel_count"]
    )

    kernel_size = int(
        tcn_config["kernel_size"]
    )

    dilations = [
        int(value)
        for value in tcn_config["dilations"]
    ]

    dropout = float(tcn_config["dropout"])

    train_endpoint_stride = int(
        tcn_config["train_endpoint_stride"]
    )

    endpoint_batch_size = int(
        tcn_config["endpoint_batch_size"]
    )

    prediction_endpoint_batch_size = int(
        tcn_config[
            "prediction_endpoint_batch_size"
        ]
    )

    max_epochs = int(tcn_config["max_epochs"])
    patience = int(tcn_config["patience"])

    minimum_improvement = float(
        tcn_config["minimum_improvement"]
    )

    learning_rate = float(
        tcn_config["learning_rate"]
    )

    weight_decay = float(
        tcn_config["weight_decay"]
    )

    gradient_clip_norm = float(
        tcn_config["gradient_clip_norm"]
    )

    seed = int(neural_config["seed"])

    set_global_seed(seed)

    device = select_device(
        str(neural_config["device"])
    )

    mixed_precision_requested = bool(
        tcn_config["mixed_precision"]
    )

    mixed_precision_enabled = (
        mixed_precision_requested
        and device.type == "cuda"
    )

    print(f"Training device: {device}")
    print(
        "Mixed precision:",
        mixed_precision_enabled,
    )

    model_probe = TemporalConvNet(
        feature_count=len(FEATURE_COLUMNS),
        channel_count=channel_count,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
    )

    if model_probe.receptive_field != sequence_length:
        raise ValueError(
            "Configured TCN receptive field must equal "
            "the sequence length"
        )

    del model_probe

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
            "tcn_experiment_name"
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
            f"=== TCN fold {fold.fold_id} ===",
            flush=True,
        )

        masks = build_fold_masks(panel, fold)

        train_split = build_sequence_split(
            frame=panel,
            row_mask=masks.train,
            temporal_panel=temporal_panel,
            sequence_length=sequence_length,
            split_name="train",
        )

        validation_split = build_sequence_split(
            frame=panel,
            row_mask=masks.validation,
            temporal_panel=temporal_panel,
            sequence_length=sequence_length,
            split_name="validation",
        )

        test_split = build_sequence_split(
            frame=panel,
            row_mask=masks.test,
            temporal_panel=temporal_panel,
            sequence_length=sequence_length,
            split_name="test",
        )

        train_frame = panel.loc[
            masks.train
        ]

        scaler = RobustFeatureScaler.fit(
            train_frame,
            FEATURE_COLUMNS,
        )

        scaled_panel = scale_temporal_panel(
            temporal_panel=temporal_panel,
            scaler=scaler,
        )

        sampled_train_endpoints = (
            subsample_endpoints(
                train_split.endpoint_positions,
                stride=train_endpoint_stride,
            )
        )

        sampled_train_rows = (
            len(sampled_train_endpoints)
            * temporal_panel.symbol_count
        )

        validation_source = (
            build_prediction_source(
                ordered_panel=panel,
                temporal_panel=temporal_panel,
                split=validation_split,
            )
        )

        test_source = build_prediction_source(
            ordered_panel=panel,
            temporal_panel=temporal_panel,
            split=test_split,
        )

        print(
            f"available_train_rows={train_split.row_count} "
            f"sampled_train_rows={sampled_train_rows} "
            f"validation_rows={validation_split.row_count} "
            f"test_rows={test_split.row_count}",
            flush=True,
        )

        model = TemporalConvNet(
            feature_count=len(FEATURE_COLUMNS),
            channel_count=channel_count,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        ).to(device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        gradient_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=mixed_precision_enabled,
        )

        best_score = -np.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        fold_history: list[dict[str, Any]] = []

        with mlflow.start_run(
            run_name=f"tcn_fold_{fold.fold_id}"
        ) as run:
            run_id = run.info.run_id

            mlflow.log_params(
                {
                    "model": "tcn",
                    "fold": fold.fold_id,
                    "seed": fold_seed,
                    "device": str(device),
                    "mixed_precision": (
                        mixed_precision_enabled
                    ),
                    "sequence_length": sequence_length,
                    "receptive_field": (
                        model.receptive_field
                    ),
                    "feature_count": len(
                        FEATURE_COLUMNS
                    ),
                    "channel_count": channel_count,
                    "kernel_size": kernel_size,
                    "dilations": ",".join(
                        str(value)
                        for value in dilations
                    ),
                    "dropout": dropout,
                    "train_endpoint_stride": (
                        train_endpoint_stride
                    ),
                    "sampled_train_rows": (
                        sampled_train_rows
                    ),
                    "endpoint_batch_size": (
                        endpoint_batch_size
                    ),
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
                    gradient_scaler=(
                        gradient_scaler
                    ),
                    temporal_panel=scaled_panel,
                    endpoint_positions=(
                        sampled_train_endpoints
                    ),
                    sequence_length=sequence_length,
                    endpoint_batch_size=(
                        endpoint_batch_size
                    ),
                    device=device,
                    mixed_precision=(
                        mixed_precision_enabled
                    ),
                    seed=fold_seed + epoch,
                    gradient_clip_norm=(
                        gradient_clip_norm
                    ),
                )

                validation_predictions = (
                    predict_sequence_model(
                        model=model,
                        temporal_panel=scaled_panel,
                        endpoint_positions=(
                            validation_split
                            .endpoint_positions
                        ),
                        sequence_length=(
                            sequence_length
                        ),
                        endpoint_batch_size=(
                            prediction_endpoint_batch_size
                        ),
                        device=device,
                        mixed_precision=(
                            mixed_precision_enabled
                        ),
                    )
                )

                validation_frame = (
                    build_prediction_frame(
                        source=validation_source,
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

                validation_rank_ic = float(
                    validation_metrics[
                        "mean_rank_ic"
                    ]
                )

                validation_loss = float(
                    validation_metrics["rmse"]
                    ** 2
                )

                history_row = {
                    "model": "tcn",
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

                fold_history.append(history_row)

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
                    "No TCN checkpoint was selected"
                )

            for row in fold_history:
                row["selected"] = (
                    row["epoch"] == best_epoch
                )

            history_rows.extend(fold_history)

            model.load_state_dict(best_state)

            validation_predictions = (
                predict_sequence_model(
                    model=model,
                    temporal_panel=scaled_panel,
                    endpoint_positions=(
                        validation_split
                        .endpoint_positions
                    ),
                    sequence_length=sequence_length,
                    endpoint_batch_size=(
                        prediction_endpoint_batch_size
                    ),
                    device=device,
                    mixed_precision=(
                        mixed_precision_enabled
                    ),
                )
            )

            test_predictions = predict_sequence_model(
                model=model,
                temporal_panel=scaled_panel,
                endpoint_positions=(
                    test_split.endpoint_positions
                ),
                sequence_length=sequence_length,
                endpoint_batch_size=(
                    prediction_endpoint_batch_size
                ),
                device=device,
                mixed_precision=(
                    mixed_precision_enabled
                ),
            )

            for split_name, source, predictions in [
                (
                    "validation",
                    validation_source,
                    validation_predictions,
                ),
                (
                    "test",
                    test_source,
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
                / f"tcn_fold_{fold.fold_id}.pt"
            )

            torch.save(
                {
                    "model": "tcn",
                    "fold": fold.fold_id,
                    "best_epoch": best_epoch,
                    "best_validation_mean_rank_ic": (
                        best_score
                    ),
                    "feature_columns": FEATURE_COLUMNS,
                    "sequence_length": sequence_length,
                    "receptive_field": (
                        model.receptive_field
                    ),
                    "channel_count": channel_count,
                    "kernel_size": kernel_size,
                    "dilations": dilations,
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
                    "available_train_rows": (
                        train_split.row_count
                    ),
                    "sampled_train_rows": (
                        sampled_train_rows
                    ),
                    "sampled_train_timestamps": (
                        len(sampled_train_endpoints)
                    ),
                    "validation_rows": (
                        validation_split.row_count
                    ),
                    "test_rows": (
                        test_split.row_count
                    ),
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
            gradient_scaler,
            scaled_panel,
            validation_predictions,
            test_predictions,
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
        / "tcn_fold_metrics.csv"
    )

    history_path = (
        evidence_directory
        / "tcn_training_history.csv"
    )

    summary_path = (
        evidence_directory
        / "tcn_summary.csv"
    )

    write_csv(metrics_frame, metrics_path)
    write_csv(history_frame, history_path)
    write_csv(summary_frame, summary_path)

    manifest_path = (
        evidence_directory
        / "tcn_run_manifest.json"
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
        "mixed_precision_requested": (
            mixed_precision_requested
        ),
        "mixed_precision_enabled": (
            mixed_precision_enabled
        ),
        "sequence_length": sequence_length,
        "receptive_field": sequence_length,
        "train_endpoint_stride": (
            train_endpoint_stride
        ),
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
    print("TCN research complete")

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
    print("Cross-fold TCN summary")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
