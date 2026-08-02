from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np
import torch
from torch import nn


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")

    return torch.device(normalized)


def iterate_minibatches(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if features.ndim != 2:
        raise ValueError("features must be two-dimensional")

    if targets.ndim != 1:
        raise ValueError("targets must be one-dimensional")

    if len(features) != len(targets):
        raise ValueError(
            "Feature and target lengths do not match"
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = np.arange(len(features))

    if shuffle:
        generator = np.random.default_rng(seed)
        generator.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]

        yield features[selected], targets[selected]


def predict_in_batches(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must be two-dimensional")

    predictions = np.empty(
        len(features),
        dtype=np.float64,
    )

    model.eval()

    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))

            batch = torch.from_numpy(
                features[start:end]
            ).to(device=device)

            output = model(batch)

            predictions[start:end] = (
                output.detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )

    return predictions
