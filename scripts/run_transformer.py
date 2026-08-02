from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.data.features import FEATURE_COLUMNS
from deep_alpha.data.scaling import RobustFeatureScaler
from deep_alpha.data.sequences import (
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
from deep_alpha.models.transformer import (
    TemporalPatchTransformer,
)
from deep_alpha.training.engine import (
    select_device,
    set_global_seed,
)
from deep_alpha.training.sequence_engine import (
    predict_sequence_model,
    scale_temporal_panel,
    subsample_endpoints,
)
from deep_alpha.training.sequence_research import (
    build_prediction_frame,
    build_prediction_source,
    build_summary,
    evaluate_and_save,
    train_sequence_epoch,
    write_csv,
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

    transformer_config = neural_config[
        "transformer"
    ]

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

    patch_size = int(
        transformer_config["patch_size"]
    )

    model_dimension = int(
        transformer_config["model_dimension"]
    )

    attention_heads = int(
        transformer_config["attention_heads"]
    )

    layer_count = int(
        transformer_config["layer_count"]
    )

    feedforward_dimension = int(
        transformer_config[
            "feedforward_dimension"
        ]
    )

    dropout = float(
        transformer_config["dropout"]
    )

    train_endpoint_stride = int(
        transformer_config[
            "train_endpoint_stride"
        ]
    )

    endpoint_batch_size = int(
        transformer_config[
            "endpoint_batch_size"
        ]
    )

    prediction_endpoint_batch_size = int(
        transformer_config[
            "prediction_endpoint_batch_size"
        ]
    )

    max_epochs = int(
        transformer_config["max_epochs"]
    )

    patience = int(
        transformer_config["patience"]
    )

    minimum_improvement = float(
        transformer_config[
            "minimum_improvement"
        ]
    )

    learning_rate = float(
        transformer_config["learning_rate"]
    )

    weight_decay = float(
        transformer_config["weight_decay"]
    )

    gradient_clip_norm = float(
        transformer_config[
            "gradient_clip_norm"
        ]
    )

    seed = int(neural_config["seed"])

    set_global_seed(seed)

    device = select_device(
        str(neural_config["device"])
    )

    mixed_precision_requested = bool(
        transformer_config["mixed_precision"]
    )

    mixed_precision_enabled = (
        mixed_precision_requested
        and device.type == "cuda"
    )

    model_probe = TemporalPatchTransformer(
        feature_count=len(FEATURE_COLUMNS),
        sequence_length=sequence_length,
        patch_size=patch_size,
        model_dimension=model_dimension,
        attention_heads=attention_heads,
        layer_count=layer_count,
        feedforward_dimension=(
            feedforward_dimension
        ),
        dropout=dropout,
    )

    token_count = model_probe.patch_count

    del model_probe

    print(f"Training device: {device}")
    print(
        "Mixed precision:",
        mixed_precision_enabled,
    )
    print("Transformer tokens:", token_count)

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

    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_database.as_posix()}"
    )

    experiment_name = str(
        neural_config["mlflow"][
            "transformer_experiment_name"
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
            f"=== Transformer fold {fold.fold_id} ===",
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

        train_frame = panel.loc[masks.train]

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

        model = TemporalPatchTransformer(
            feature_count=len(FEATURE_COLUMNS),
            sequence_length=sequence_length,
            patch_size=patch_size,
            model_dimension=model_dimension,
            attention_heads=attention_heads,
            layer_count=layer_count,
            feedforward_dimension=(
                feedforward_dimension
            ),
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
            run_name=(
                f"transformer_fold_{fold.fold_id}"
            )
        ) as run:
            run_id = run.info.run_id

            mlflow.log_params(
                {
                    "model": "transformer",
                    "fold": fold.fold_id,
                    "seed": fold_seed,
                    "device": str(device),
                    "mixed_precision": (
                        mixed_precision_enabled
                    ),
                    "sequence_length": sequence_length,
                    "patch_size": patch_size,
                    "token_count": token_count,
                    "feature_count": len(
                        FEATURE_COLUMNS
                    ),
                    "model_dimension": (
                        model_dimension
                    ),
                    "attention_heads": (
                        attention_heads
                    ),
                    "layer_count": layer_count,
                    "feedforward_dimension": (
                        feedforward_dimension
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
                train_loss = train_sequence_epoch(
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
                    "model": "transformer",
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
                    "No Transformer checkpoint "
                    "was selected"
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
                        model_name="transformer",
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
                / (
                    "transformer_fold_"
                    f"{fold.fold_id}.pt"
                )
            )

            torch.save(
                {
                    "model": "transformer",
                    "fold": fold.fold_id,
                    "best_epoch": best_epoch,
                    "best_validation_mean_rank_ic": (
                        best_score
                    ),
                    "feature_columns": FEATURE_COLUMNS,
                    "sequence_length": sequence_length,
                    "patch_size": patch_size,
                    "token_count": token_count,
                    "model_dimension": (
                        model_dimension
                    ),
                    "attention_heads": (
                        attention_heads
                    ),
                    "layer_count": layer_count,
                    "feedforward_dimension": (
                        feedforward_dimension
                    ),
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
        / "transformer_fold_metrics.csv"
    )

    history_path = (
        evidence_directory
        / "transformer_training_history.csv"
    )

    summary_path = (
        evidence_directory
        / "transformer_summary.csv"
    )

    write_csv(metrics_frame, metrics_path)
    write_csv(history_frame, history_path)
    write_csv(summary_frame, summary_path)

    manifest_path = (
        evidence_directory
        / "transformer_run_manifest.json"
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
        "patch_size": patch_size,
        "token_count": token_count,
        "model_dimension": model_dimension,
        "attention_heads": attention_heads,
        "layer_count": layer_count,
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
    print("Transformer research complete")

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
    print("Cross-fold Transformer summary")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
