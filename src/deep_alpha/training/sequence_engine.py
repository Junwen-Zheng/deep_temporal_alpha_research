from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch import nn

from deep_alpha.data.scaling import RobustFeatureScaler
from deep_alpha.data.sequences import TemporalPanel


def scale_temporal_panel(
    temporal_panel: TemporalPanel,
    scaler: RobustFeatureScaler,
) -> TemporalPanel:
    if (
        temporal_panel.feature_count
        != len(scaler.feature_columns)
    ):
        raise ValueError(
            "Scaler feature count differs from temporal panel"
        )

    median = scaler.median.astype(
        np.float32,
        copy=False,
    ).reshape(1, 1, -1)

    scale = scaler.scale.astype(
        np.float32,
        copy=False,
    ).reshape(1, 1, -1)

    features = (
        temporal_panel.features - median
    ) / scale

    features = np.ascontiguousarray(
        features,
        dtype=np.float32,
    )

    if not np.isfinite(features).all():
        raise ValueError(
            "Scaled temporal features contain "
            "non-finite values"
        )

    return TemporalPanel(
        timestamps=temporal_panel.timestamps,
        symbols=temporal_panel.symbols,
        features=features,
        targets=temporal_panel.targets,
        raw_targets=temporal_panel.raw_targets,
    )


def subsample_endpoints(
    endpoint_positions: np.ndarray,
    stride: int,
) -> np.ndarray:
    endpoints = np.asarray(
        endpoint_positions,
        dtype=np.int64,
    )

    if endpoints.ndim != 1:
        raise ValueError(
            "endpoint_positions must be one-dimensional"
        )

    if len(endpoints) == 0:
        raise ValueError(
            "At least one endpoint is required"
        )

    if stride <= 0:
        raise ValueError("stride must be positive")

    selected = endpoints[::stride]

    return np.ascontiguousarray(
        selected,
        dtype=np.int64,
    )


def iterate_endpoint_batches(
    endpoint_positions: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[np.ndarray]:
    endpoints = np.asarray(
        endpoint_positions,
        dtype=np.int64,
    ).copy()

    if endpoints.ndim != 1:
        raise ValueError(
            "endpoint_positions must be one-dimensional"
        )

    if len(endpoints) == 0:
        raise ValueError(
            "At least one endpoint is required"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if shuffle:
        generator = np.random.default_rng(seed)
        generator.shuffle(endpoints)

    for start in range(0, len(endpoints), batch_size):
        yield endpoints[start : start + batch_size]


def extract_cross_sectional_batch(
    temporal_panel: TemporalPanel,
    endpoint_positions: np.ndarray,
    sequence_length: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    endpoints = np.asarray(
        endpoint_positions,
        dtype=np.int64,
    )

    if endpoints.ndim != 1:
        raise ValueError(
            "endpoint_positions must be one-dimensional"
        )

    if len(endpoints) == 0:
        raise ValueError(
            "At least one endpoint is required"
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    if np.any(endpoints < sequence_length - 1):
        raise ValueError(
            "An endpoint lacks sufficient sequence history"
        )

    if np.any(
        endpoints >= temporal_panel.timestamp_count
    ):
        raise ValueError(
            "An endpoint exceeds the temporal panel"
        )

    offsets = np.arange(
        -sequence_length + 1,
        1,
        dtype=np.int64,
    )

    time_positions = (
        endpoints[:, None] + offsets[None, :]
    )

    windows = temporal_panel.features[
        time_positions,
        :,
        :,
    ]

    windows = windows.transpose(
        0,
        2,
        1,
        3,
    )

    feature_batch = np.ascontiguousarray(
        windows.reshape(
            -1,
            sequence_length,
            temporal_panel.feature_count,
        ),
        dtype=np.float32,
    )

    target_batch = np.ascontiguousarray(
        temporal_panel.targets[
            endpoints,
            :,
        ].reshape(-1),
        dtype=np.float32,
    )

    repeated_endpoints = np.repeat(
        endpoints,
        temporal_panel.symbol_count,
    )

    symbol_indices = np.tile(
        np.arange(
            temporal_panel.symbol_count,
            dtype=np.int64,
        ),
        len(endpoints),
    )

    return (
        feature_batch,
        target_batch,
        repeated_endpoints,
        symbol_indices,
    )


def predict_sequence_model(
    model: nn.Module,
    temporal_panel: TemporalPanel,
    endpoint_positions: np.ndarray,
    sequence_length: int,
    endpoint_batch_size: int,
    device: torch.device,
    mixed_precision: bool,
) -> np.ndarray:
    prediction_batches = []

    model.eval()

    with torch.inference_mode():
        for endpoint_batch in iterate_endpoint_batches(
            endpoint_positions=endpoint_positions,
            batch_size=endpoint_batch_size,
            shuffle=False,
            seed=0,
        ):
            features, _, _, _ = (
                extract_cross_sectional_batch(
                    temporal_panel=temporal_panel,
                    endpoint_positions=endpoint_batch,
                    sequence_length=sequence_length,
                )
            )

            feature_tensor = torch.from_numpy(
                features
            ).to(device=device)

            if mixed_precision:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    output = model(feature_tensor)
            else:
                output = model(feature_tensor)

            prediction_batches.append(
                output.detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )

    predictions = np.concatenate(
        prediction_batches
    )

    expected_rows = (
        len(endpoint_positions)
        * temporal_panel.symbol_count
    )

    if len(predictions) != expected_rows:
        raise RuntimeError(
            "Sequence prediction count is incorrect"
        )

    return predictions
