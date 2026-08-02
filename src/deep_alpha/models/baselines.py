from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> Ridge:
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")

    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
    )

    model.fit(features, targets)
    return model


def fit_lightgbm(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    configuration: Mapping[str, Any],
    seed: int,
) -> LGBMRegressor:
    parameters = dict(configuration)

    early_stopping_rounds = int(
        parameters.pop("early_stopping_rounds")
    )

    model = LGBMRegressor(
        **parameters,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )

    model.fit(
        train_features,
        train_targets,
        eval_set=[
            (
                validation_features,
                validation_targets,
            )
        ],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    return model
