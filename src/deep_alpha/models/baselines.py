from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> Ridge:
    from sklearn.linear_model import Ridge

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
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

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
        eval_X=validation_features,
        eval_y=validation_targets,
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
