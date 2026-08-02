from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustFeatureScaler:
    feature_columns: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> RobustFeatureScaler:
        columns = tuple(feature_columns)

        if not columns:
            raise ValueError("At least one feature is required")

        values = frame.loc[:, columns].to_numpy(
            dtype=np.float64,
        )

        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("Training feature matrix is empty")

        if not np.isfinite(values).all():
            raise ValueError(
                "Training features contain non-finite values"
            )

        median = np.median(values, axis=0)
        lower_quartile = np.quantile(values, 0.25, axis=0)
        upper_quartile = np.quantile(values, 0.75, axis=0)

        scale = upper_quartile - lower_quartile
        scale = np.where(scale > 1e-12, scale, 1.0)

        return cls(
            feature_columns=columns,
            median=median,
            scale=scale,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[
            :,
            self.feature_columns,
        ].to_numpy(dtype=np.float64)

        if values.ndim != 2:
            raise ValueError("Feature matrix must be two-dimensional")

        if not np.isfinite(values).all():
            raise ValueError(
                "Features contain non-finite values"
            )

        transformed = (
            values - self.median
        ) / self.scale

        if not np.isfinite(transformed).all():
            raise ValueError(
                "Scaled features contain non-finite values"
            )

        return transformed.astype(
            np.float32,
            copy=False,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_columns": list(self.feature_columns),
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
        }
