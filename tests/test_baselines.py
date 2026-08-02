from __future__ import annotations

import numpy as np

from deep_alpha.models.baselines import (
    fit_lightgbm,
    fit_ridge,
)


def sample_regression_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    generator = np.random.default_rng(42)

    features = generator.normal(
        size=(200, 4)
    ).astype(np.float32)

    targets = (
        features[:, 0] * 0.5
        - features[:, 1] * 0.25
    )

    return features, targets


def test_ridge_predicts_expected_shape() -> None:
    features, targets = sample_regression_data()

    model = fit_ridge(
        features=features,
        targets=targets,
        alpha=1.0,
    )

    predictions = model.predict(features)

    assert predictions.shape == (200,)
    assert np.isfinite(predictions).all()


def test_lightgbm_predicts_expected_shape() -> None:
    features, targets = sample_regression_data()

    model = fit_lightgbm(
        train_features=features[:150],
        train_targets=targets[:150],
        validation_features=features[150:],
        validation_targets=targets[150:],
        configuration={
            "objective": "regression",
            "n_estimators": 30,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "max_depth": -1,
            "min_child_samples": 10,
            "subsample": 1.0,
            "subsample_freq": 0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "early_stopping_rounds": 5,
            "n_jobs": 1,
        },
        seed=42,
    )

    predictions = model.predict(
        features[150:]
    )

    assert predictions.shape == (50,)
    assert np.isfinite(predictions).all()
