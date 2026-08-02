from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from deep_alpha.data.download import sha256_file
from deep_alpha.data.sequences import (
    SequenceSplit,
    TemporalPanel,
)
from deep_alpha.evaluation.metrics import (
    summarize_predictions,
)
from deep_alpha.training.sequence_engine import (
    extract_cross_sectional_batch,
    iterate_endpoint_batches,
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


def train_sequence_epoch(
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
            "No sequence training rows were processed"
        )

    return total_loss / total_rows


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

    record = {
        "model": model_name,
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
