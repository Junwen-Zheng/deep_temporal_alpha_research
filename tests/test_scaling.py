from __future__ import annotations

import numpy as np
import pandas as pd

from deep_alpha.data.scaling import (
    RobustFeatureScaler,
)


def test_scaler_uses_training_data_only() -> None:
    training = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
        }
    )

    validation = pd.DataFrame(
        {
            "feature": [100.0],
        }
    )

    scaler = RobustFeatureScaler.fit(
        training,
        ["feature"],
    )

    transformed = scaler.transform(validation)

    assert scaler.median.tolist() == [2.0]
    assert transformed[0, 0] > 50.0


def test_constant_feature_uses_unit_scale() -> None:
    frame = pd.DataFrame(
        {
            "constant": [5.0, 5.0, 5.0],
        }
    )

    scaler = RobustFeatureScaler.fit(
        frame,
        ["constant"],
    )

    transformed = scaler.transform(frame)

    assert scaler.scale.tolist() == [1.0]
    np.testing.assert_allclose(
        transformed,
        np.zeros((3, 1), dtype=np.float32),
    )
