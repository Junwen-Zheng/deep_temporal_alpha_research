from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run_isolated(source: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(source),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise AssertionError(
            "Isolated model test failed.\n"
            f"Return code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def test_ridge_predicts_expected_shape() -> None:
    run_isolated(
        """
        import numpy as np

        from deep_alpha.models.baselines import fit_ridge

        generator = np.random.default_rng(42)

        features = generator.normal(
            size=(200, 4)
        ).astype(np.float32)

        targets = (
            features[:, 0] * 0.5
            - features[:, 1] * 0.25
        )

        model = fit_ridge(
            features=features,
            targets=targets,
            alpha=1.0,
        )

        predictions = model.predict(features)

        assert predictions.shape == (200,)
        assert np.isfinite(predictions).all()
        """
    )


def test_lightgbm_predicts_expected_shape() -> None:
    run_isolated(
        """
        import numpy as np

        from deep_alpha.models.baselines import fit_lightgbm

        generator = np.random.default_rng(42)

        features = generator.normal(
            size=(200, 4)
        ).astype(np.float32)

        targets = (
            features[:, 0] * 0.5
            - features[:, 1] * 0.25
        )

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
        """
    )
