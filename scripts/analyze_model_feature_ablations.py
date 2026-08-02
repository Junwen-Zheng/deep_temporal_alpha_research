from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.data.features import (
    FEATURE_COLUMNS,
)
from deep_alpha.data.scaling import (
    RobustFeatureScaler,
)
from deep_alpha.data.splits import (
    build_fold_masks,
    parse_fold_definition,
)
from deep_alpha.evaluation.ablations import (
    add_full_variant_deltas,
    build_ablation_portfolio_summary,
    build_leave_one_family_out_sets,
)
from deep_alpha.evaluation.metrics import (
    summarize_predictions,
)
from deep_alpha.evaluation.portfolio import (
    build_portfolio_periods,
    summarize_portfolio_cohorts,
)
from deep_alpha.models.baselines import (
    fit_lightgbm,
    fit_ridge,
)


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


def build_prediction_frame(
    source: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    if len(source) != len(predictions):
        raise ValueError(
            "Prediction length differs "
            "from source rows"
        )

    result = source[
        [
            "timestamp",
            "symbol",
            "raw_target",
            "target",
        ]
    ].copy()

    result["prediction"] = np.asarray(
        predictions,
        dtype=np.float64,
    )

    return result


def load_frozen_prediction(
    model: str,
    fold: int,
    split: str,
    directory: Path,
) -> pd.DataFrame:
    path = (
        directory
        / (
            f"{model}_fold_{fold}_"
            f"{split}.parquet"
        )
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    return (
        pd.read_parquet(
            path,
            columns=[
                "timestamp",
                "symbol",
                "raw_target",
                "target",
                "prediction",
            ],
        )
        .sort_values(
            [
                "timestamp",
                "symbol",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def select_ridge(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_source: pd.DataFrame,
    alphas: list[float],
    model: str,
    variant: str,
    fold: int,
) -> tuple[
    object,
    np.ndarray,
    float,
    list[dict[str, Any]],
]:
    candidates = []

    for alpha in alphas:
        fitted = fit_ridge(
            features=train_features,
            targets=train_targets,
            alpha=alpha,
        )

        predictions = fitted.predict(
            validation_features
        )

        metrics = summarize_predictions(
            build_prediction_frame(
                source=validation_source,
                predictions=predictions,
            )
        )

        candidates.append(
            {
                "alpha": alpha,
                "fitted": fitted,
                "predictions": predictions,
                "validation_mean_rank_ic": (
                    float(
                        metrics[
                            "mean_rank_ic"
                        ]
                    )
                ),
            }
        )

    selected = max(
        candidates,
        key=lambda candidate: (
            candidate[
                "validation_mean_rank_ic"
            ]
        ),
    )

    rows = []

    for candidate in candidates:
        rows.append(
            {
                "model": model,
                "variant": variant,
                "fold": fold,
                "selection_type": (
                    "ridge_alpha"
                ),
                "candidate": (
                    f"alpha={candidate['alpha']}"
                ),
                "validation_mean_rank_ic": (
                    candidate[
                        "validation_mean_rank_ic"
                    ]
                ),
                "selected": (
                    candidate["alpha"]
                    == selected["alpha"]
                ),
                "source_frozen": False,
            }
        )

    return (
        selected["fitted"],
        selected["predictions"],
        float(selected["alpha"]),
        rows,
    )


def load_execution_returns(
    symbols: list[str],
    processed_directory: Path,
    holding_bars: int,
) -> pd.DataFrame:
    frames = []

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(symbols):02d}] "
            f"Loading execution bars for "
            f"{symbol}",
            flush=True,
        )

        path = (
            processed_directory
            / "bars"
            / f"symbol={symbol}"
            / "part-00000.parquet"
        )

        bars = (
            pd.read_parquet(
                path,
                columns=[
                    "timestamp",
                    "open",
                    "close",
                ],
            )
            .sort_values(
                "timestamp",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": (
                        bars["timestamp"]
                    ),
                    "symbol": symbol,
                    "execution_return": np.log(
                        bars["close"].shift(
                            -holding_bars
                        )
                        / bars["open"].shift(-1)
                    ),
                }
            )
        )

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "timestamp",
                "symbol",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def verify_full_prediction_summary(
    feature_summary: pd.DataFrame,
    source_oos: pd.DataFrame,
) -> None:
    full = feature_summary.loc[
        feature_summary[
            "variant"
        ]
        == "full"
    ]

    comparison = (
        source_oos.loc[
            source_oos["model"].isin(
                [
                    "ridge",
                    "lightgbm",
                ]
            )
        ]
        .merge(
            full,
            on="model",
            validate="one_to_one",
            suffixes=(
                "_source",
                "_reproduced",
            ),
        )
    )

    if len(comparison) != 2:
        raise ValueError(
            "Full prediction reproduction "
            "does not cover both models"
        )

    numeric_columns = [
        "mean_rank_ic",
        "median_rank_ic",
        "rank_ic_std",
        "rank_ic_ir",
        "positive_timestamp_fraction",
        "rmse",
        "mae",
    ]

    for column in numeric_columns:
        np.testing.assert_allclose(
            comparison[
                f"{column}_source"
            ].to_numpy(dtype=np.float64),
            comparison[
                f"{column}_reproduced"
            ].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-10,
        )

    for column in [
        "n_rows",
        "n_timestamps",
    ]:
        if not np.array_equal(
            comparison[
                f"{column}_source"
            ].to_numpy(),
            comparison[
                f"{column}_reproduced"
            ].to_numpy(),
        ):
            raise ValueError(
                f"Full prediction {column} "
                "does not reproduce"
            )


def verify_full_portfolios(
    cohort_metrics: pd.DataFrame,
    source_cohorts: pd.DataFrame,
) -> None:
    full = cohort_metrics.loc[
        cohort_metrics[
            "variant"
        ]
        == "full"
    ]

    source = source_cohorts.loc[
        source_cohorts[
            "model"
        ].isin(
            [
                "ridge",
                "lightgbm",
            ]
        )
    ]

    comparison = source.merge(
        full,
        on=[
            "model",
            "cost_bps",
            "cohort",
        ],
        validate="one_to_one",
        suffixes=(
            "_source",
            "_reproduced",
        ),
    )

    if len(comparison) != len(source):
        raise ValueError(
            "Full portfolio reproduction "
            "is incomplete"
        )

    if not np.array_equal(
        comparison[
            "period_count_source"
        ].to_numpy(),
        comparison[
            "period_count_reproduced"
        ].to_numpy(),
    ):
        raise ValueError(
            "Full portfolio period counts "
            "differ"
        )

    numeric_columns = [
        "mean_net_return",
        "annualized_net_return",
        "annualized_volatility",
        "annualized_sharpe",
        "positive_fraction",
        "mean_turnover",
        "cumulative_net_log_return",
    ]

    for column in numeric_columns:
        np.testing.assert_allclose(
            comparison[
                f"{column}_source"
            ].to_numpy(dtype=np.float64),
            comparison[
                f"{column}_reproduced"
            ].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        )


def build_model_family_summary(
    source_oos: pd.DataFrame,
    source_break_even: pd.DataFrame,
    bootstrap_prediction: pd.DataFrame,
    bootstrap_portfolio: pd.DataFrame,
) -> pd.DataFrame:
    family_records = [
        {
            "model": "momentum_12",
            "model_family": (
                "momentum_reference"
            ),
            "learned": False,
            "temporal_input": True,
        },
        {
            "model": "reversal_12",
            "model_family": (
                "reversal_reference"
            ),
            "learned": False,
            "temporal_input": True,
        },
        {
            "model": "ridge",
            "model_family": (
                "linear_pointwise"
            ),
            "learned": True,
            "temporal_input": False,
        },
        {
            "model": "lightgbm",
            "model_family": (
                "tree_pointwise"
            ),
            "learned": True,
            "temporal_input": False,
        },
        {
            "model": "mlp",
            "model_family": (
                "neural_pointwise"
            ),
            "learned": True,
            "temporal_input": False,
        },
        {
            "model": "tcn",
            "model_family": (
                "neural_temporal_convolution"
            ),
            "learned": True,
            "temporal_input": True,
        },
        {
            "model": "transformer",
            "model_family": (
                "neural_temporal_attention"
            ),
            "learned": True,
            "temporal_input": True,
        },
    ]

    families = pd.DataFrame(
        family_records
    )

    prediction_interval = (
        bootstrap_prediction.loc[
            bootstrap_prediction[
                "block_length_timestamps"
            ]
            == 12,
            [
                "model",
                "confidence_lower",
                "confidence_upper",
                "probability_positive",
                "valid_timestamp_count",
                "undefined_timestamp_count",
            ],
        ]
        .rename(
            columns={
                "confidence_lower": (
                    "rank_ic_confidence_lower"
                ),
                "confidence_upper": (
                    "rank_ic_confidence_upper"
                ),
                "probability_positive": (
                    "rank_ic_probability_positive"
                ),
            }
        )
    )

    portfolio_interval = (
        bootstrap_portfolio.loc[
            bootstrap_portfolio[
                "block_length_periods"
            ]
            == 24,
            [
                "model",
                "break_even_confidence_lower",
                "break_even_confidence_upper",
                "probability_break_even_above_focus_cost",
                "net_probability_positive",
            ],
        ]
    )

    result = (
        families.merge(
            source_oos,
            on="model",
            validate="one_to_one",
        )
        .merge(
            source_break_even[
                [
                    "model",
                    "pooled_break_even_cost_bps",
                ]
            ],
            on="model",
            validate="one_to_one",
        )
        .merge(
            prediction_interval,
            on="model",
            validate="one_to_one",
        )
        .merge(
            portfolio_interval,
            on="model",
            validate="one_to_one",
        )
    )

    reversal = result.loc[
        result["model"]
        == "reversal_12"
    ].iloc[0]

    mlp = result.loc[
        result["model"]
        == "mlp"
    ].iloc[0]

    result[
        "rank_ic_delta_vs_reversal"
    ] = (
        result["mean_rank_ic"]
        - float(
            reversal["mean_rank_ic"]
        )
    )

    result[
        "break_even_delta_vs_reversal_bps"
    ] = (
        result[
            "pooled_break_even_cost_bps"
        ]
        - float(
            reversal[
                "pooled_break_even_cost_bps"
            ]
        )
    )

    result[
        "rank_ic_delta_vs_mlp"
    ] = (
        result["mean_rank_ic"]
        - float(
            mlp["mean_rank_ic"]
        )
    )

    result[
        "break_even_delta_vs_mlp_bps"
    ] = (
        result[
            "pooled_break_even_cost_bps"
        ]
        - float(
            mlp[
                "pooled_break_even_cost_bps"
            ]
        )
    )

    return (
        result.sort_values(
            "mean_rank_ic",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def main() -> None:
    config_path = Path(
        "configs/ablations.yaml"
    )

    config = load_yaml(
        config_path
    )["ablations"]

    data_config_path = Path(
        config["data_config"]
    )

    research_config_path = Path(
        config["research_config"]
    )

    baseline_config_path = Path(
        config["baseline_config"]
    )

    evaluation_config_path = Path(
        config["evaluation_config"]
    )

    data_config = load_yaml(
        data_config_path
    )["data"]

    research_config = load_yaml(
        research_config_path
    )["research"]

    baseline_config = load_yaml(
        baseline_config_path
    )["baselines"]

    evaluation_config = load_yaml(
        evaluation_config_path
    )["evaluation"]

    source_oos_path = Path(
        config["source_oos_metrics"]
    )

    source_cohort_path = Path(
        config[
            "source_portfolio_cohort_metrics"
        ]
    )

    source_portfolio_manifest_path = Path(
        config[
            "source_portfolio_manifest"
        ]
    )

    source_break_even_path = Path(
        config[
            "source_break_even_summary"
        ]
    )

    source_baseline_manifest_path = Path(
        config[
            "source_baseline_manifest"
        ]
    )

    source_baseline_selection_path = Path(
        config[
            "source_baseline_selection"
        ]
    )

    bootstrap_prediction_path = Path(
        config[
            "source_bootstrap_prediction"
        ]
    )

    bootstrap_portfolio_path = Path(
        config[
            "source_bootstrap_portfolio"
        ]
    )

    source_oos = pd.read_csv(
        source_oos_path
    )

    source_cohorts = pd.read_csv(
        source_cohort_path
    )

    source_break_even = pd.read_csv(
        source_break_even_path
    )

    source_baseline_selection = pd.read_csv(
        source_baseline_selection_path
    )

    bootstrap_prediction = pd.read_csv(
        bootstrap_prediction_path
    )

    bootstrap_portfolio = pd.read_csv(
        bootstrap_portfolio_path
    )

    source_portfolio_manifest = (
        json.loads(
            source_portfolio_manifest_path
            .read_text(
                encoding="utf-8"
            )
        )
    )

    source_baseline_manifest = (
        json.loads(
            source_baseline_manifest_path
            .read_text(
                encoding="utf-8"
            )
        )
    )

    configured_models = [
        str(value)
        for value in config["models"]
    ]

    if configured_models != [
        "ridge",
        "lightgbm",
    ]:
        raise ValueError(
            "Ablation models must be "
            "Ridge and LightGBM"
        )

    feature_families = {
        str(family): [
            str(feature)
            for feature in columns
        ]
        for family, columns in (
            config[
                "feature_families"
            ].items()
        )
    }

    feature_sets = (
        build_leave_one_family_out_sets(
            feature_columns=FEATURE_COLUMNS,
            feature_families=(
                feature_families
            ),
        )
    )

    symbols = [
        str(symbol)
        for symbol in data_config[
            "symbols"
        ]
    ]

    expected_symbol_count = int(
        config[
            "expected_symbol_count"
        ]
    )

    expected_prediction_rows = int(
        config[
            "expected_prediction_rows"
        ]
    )

    expected_prediction_timestamps = int(
        config[
            "expected_prediction_timestamps"
        ]
    )

    if len(symbols) != expected_symbol_count:
        raise ValueError(
            "Symbol universe size differs"
        )

    holding_bars = int(
        evaluation_config[
            "holding_bars"
        ]
    )

    interval_minutes = int(
        evaluation_config[
            "interval_minutes"
        ]
    )

    top_count = int(
        evaluation_config[
            "top_count"
        ]
    )

    annualization_periods = int(
        evaluation_config[
            "annualization_periods"
        ]
    )

    costs = [
        float(value)
        for value in config[
            "cost_bps"
        ]
    ]

    focus_cost_bps = float(
        config[
            "focus_cost_bps"
        ]
    )

    if not any(
        np.isclose(
            value,
            focus_cost_bps,
            rtol=0.0,
            atol=1e-12,
        )
        for value in costs
    ):
        raise ValueError(
            "Focus cost is absent from "
            "configured costs"
        )

    manifest_expectations = {
        "symbol_count": (
            expected_symbol_count
        ),
        "holding_bars": holding_bars,
        "interval_minutes": (
            interval_minutes
        ),
        "top_count": top_count,
        "annualization_periods": (
            annualization_periods
        ),
        "prediction_rows_per_model": (
            expected_prediction_rows
        ),
        "prediction_timestamps_per_model": (
            expected_prediction_timestamps
        ),
    }

    for key, expected_value in (
        manifest_expectations.items()
    ):
        actual_value = (
            source_portfolio_manifest.get(
                key
            )
        )

        if actual_value != expected_value:
            raise ValueError(
                "Source portfolio manifest "
                f"{key} differs: "
                f"{actual_value} versus "
                f"{expected_value}"
            )

    source_manifest_path = Path(
        research_config[
            "manifest_path"
        ]
    )

    source_manifest = json.loads(
        source_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    panel_path = Path(
        source_manifest[
            "output_path"
        ]
    )

    print(
        "Loading research panel",
        flush=True,
    )

    panel = (
        pd.read_parquet(
            panel_path,
            columns=[
                "timestamp",
                "label_end_timestamp",
                "symbol",
                "raw_target",
                "target",
                *FEATURE_COLUMNS,
            ],
        )
        .sort_values(
            [
                "timestamp",
                "symbol",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    fold_data = {}

    for fold_configuration in (
        research_config["folds"]
    ):
        fold = parse_fold_definition(
            fold_configuration
        )

        masks = build_fold_masks(
            panel,
            fold,
        )

        fold_data[
            fold.fold_id
        ] = {
            "fold": fold,
            "train": panel.loc[
                masks.train
            ].copy(),
            "validation": panel.loc[
                masks.validation
            ].copy(),
            "test": panel.loc[
                masks.test
            ].copy(),
        }

    execution_returns = (
        load_execution_returns(
            symbols=symbols,
            processed_directory=Path(
                data_config[
                    "processed_dir"
                ]
            ),
            holding_bars=holding_bars,
        )
    )

    prediction_directory = Path(
        baseline_config[
            "prediction_dir"
        ]
    )

    ridge_alphas = [
        float(value)
        for value in baseline_config[
            "ridge"
        ]["alphas"]
    ]

    seed = int(
        baseline_config["seed"]
    )

    fold_metric_rows = []
    selection_rows = []
    feature_summary_rows = []
    cohort_frames = []

    source_ridge_selection = (
        source_baseline_selection.loc[
            source_baseline_selection[
                "model"
            ]
            == "ridge"
        ]
        .copy()
    )

    for row in (
        source_ridge_selection
        .itertuples(index=False)
    ):
        selection_rows.append(
            {
                "model": "ridge",
                "variant": "full",
                "fold": int(row.fold),
                "selection_type": (
                    "ridge_alpha"
                ),
                "candidate": str(
                    row.candidate
                ),
                "validation_mean_rank_ic": (
                    float(
                        row.validation_mean_rank_ic
                    )
                ),
                "selected": bool(
                    row.selected
                ),
                "source_frozen": True,
            }
        )

    baseline_fold_records = {
        int(record["fold"]): record
        for record in (
            source_baseline_manifest[
                "folds"
            ]
        )
    }

    for model in configured_models:
        for variant, features in (
            feature_sets.items()
        ):
            print()
            print(
                f"=== Ablation: {model} / "
                f"{variant} ===",
                flush=True,
            )

            test_frames = []
            fold_test_rank_ics = []

            for fold_id in [
                1,
                2,
                3,
            ]:
                data = fold_data[
                    fold_id
                ]

                train = data["train"]
                validation = data[
                    "validation"
                ]

                test = data["test"]

                if variant == "full":
                    validation_frame = (
                        load_frozen_prediction(
                            model=model,
                            fold=fold_id,
                            split="validation",
                            directory=(
                                prediction_directory
                            ),
                        )
                    )

                    test_frame = (
                        load_frozen_prediction(
                            model=model,
                            fold=fold_id,
                            split="test",
                            directory=(
                                prediction_directory
                            ),
                        )
                    )

                    if model == "lightgbm":
                        record = (
                            baseline_fold_records[
                                fold_id
                            ]
                        )

                        validation_metrics = (
                            summarize_predictions(
                                validation_frame
                            )
                        )

                        selection_rows.append(
                            {
                                "model": model,
                                "variant": variant,
                                "fold": fold_id,
                                "selection_type": (
                                    "lightgbm_iteration"
                                ),
                                "candidate": (
                                    "best_iteration="
                                    f"{record['lightgbm_best_iteration']}"
                                ),
                                "validation_mean_rank_ic": (
                                    float(
                                        validation_metrics[
                                            "mean_rank_ic"
                                        ]
                                    )
                                ),
                                "selected": True,
                                "source_frozen": True,
                            }
                        )
                elif model == "ridge":
                    scaler = (
                        RobustFeatureScaler.fit(
                            train,
                            features,
                        )
                    )

                    train_features = (
                        scaler.transform(train)
                    )

                    validation_features = (
                        scaler.transform(
                            validation
                        )
                    )

                    test_features = (
                        scaler.transform(test)
                    )

                    train_targets = train[
                        "target"
                    ].to_numpy(
                        dtype=np.float64
                    )

                    (
                        fitted,
                        validation_predictions,
                        selected_alpha,
                        candidate_rows,
                    ) = select_ridge(
                        train_features=(
                            train_features
                        ),
                        train_targets=(
                            train_targets
                        ),
                        validation_features=(
                            validation_features
                        ),
                        validation_source=(
                            validation
                        ),
                        alphas=ridge_alphas,
                        model=model,
                        variant=variant,
                        fold=fold_id,
                    )

                    selection_rows.extend(
                        candidate_rows
                    )

                    test_predictions = (
                        fitted.predict(
                            test_features
                        )
                    )

                    validation_frame = (
                        build_prediction_frame(
                            source=validation,
                            predictions=(
                                validation_predictions
                            ),
                        )
                    )

                    test_frame = (
                        build_prediction_frame(
                            source=test,
                            predictions=(
                                test_predictions
                            ),
                        )
                    )

                    print(
                        f"  fold={fold_id} "
                        f"selected_alpha="
                        f"{selected_alpha}",
                        flush=True,
                    )
                else:
                    train_features = train[
                        features
                    ].to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )

                    validation_features = (
                        validation[
                            features
                        ].to_numpy(
                            dtype=np.float32,
                            copy=False,
                        )
                    )

                    test_features = test[
                        features
                    ].to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )

                    train_targets = train[
                        "target"
                    ].to_numpy(
                        dtype=np.float64
                    )

                    validation_targets = (
                        validation[
                            "target"
                        ].to_numpy(
                            dtype=np.float64
                        )
                    )

                    fitted = fit_lightgbm(
                        train_features=(
                            train_features
                        ),
                        train_targets=(
                            train_targets
                        ),
                        validation_features=(
                            validation_features
                        ),
                        validation_targets=(
                            validation_targets
                        ),
                        configuration=(
                            baseline_config[
                                "lightgbm"
                            ]
                        ),
                        seed=(
                            seed + fold_id
                        ),
                    )

                    best_iteration = int(
                        fitted.best_iteration_
                    )

                    validation_predictions = (
                        fitted.predict(
                            validation_features,
                            num_iteration=(
                                best_iteration
                            ),
                        )
                    )

                    test_predictions = (
                        fitted.predict(
                            test_features,
                            num_iteration=(
                                best_iteration
                            ),
                        )
                    )

                    validation_frame = (
                        build_prediction_frame(
                            source=validation,
                            predictions=(
                                validation_predictions
                            ),
                        )
                    )

                    test_frame = (
                        build_prediction_frame(
                            source=test,
                            predictions=(
                                test_predictions
                            ),
                        )
                    )

                    validation_metrics = (
                        summarize_predictions(
                            validation_frame
                        )
                    )

                    selection_rows.append(
                        {
                            "model": model,
                            "variant": variant,
                            "fold": fold_id,
                            "selection_type": (
                                "lightgbm_iteration"
                            ),
                            "candidate": (
                                "best_iteration="
                                f"{best_iteration}"
                            ),
                            "validation_mean_rank_ic": (
                                float(
                                    validation_metrics[
                                        "mean_rank_ic"
                                    ]
                                )
                            ),
                            "selected": True,
                            "source_frozen": False,
                        }
                    )

                    print(
                        f"  fold={fold_id} "
                        f"best_iteration="
                        f"{best_iteration}",
                        flush=True,
                    )

                for split_name, frame in [
                    (
                        "validation",
                        validation_frame,
                    ),
                    (
                        "test",
                        test_frame,
                    ),
                ]:
                    metrics = (
                        summarize_predictions(
                            frame
                        )
                    )

                    fold_metric_rows.append(
                        {
                            "model": model,
                            "variant": variant,
                            "removed_family": (
                                ""
                                if variant == "full"
                                else variant.removeprefix(
                                    "without_"
                                )
                            ),
                            "feature_count": len(
                                features
                            ),
                            "fold": fold_id,
                            "split": split_name,
                            **metrics,
                        }
                    )

                    if split_name == "test":
                        fold_test_rank_ics.append(
                            float(
                                metrics[
                                    "mean_rank_ic"
                                ]
                            )
                        )

                test_frames.append(
                    test_frame
                )

            combined_test = (
                pd.concat(
                    test_frames,
                    ignore_index=True,
                )
                .sort_values(
                    [
                        "timestamp",
                        "symbol",
                    ],
                    kind="mergesort",
                )
                .reset_index(drop=True)
            )

            aggregate_metrics = (
                summarize_predictions(
                    combined_test
                )
            )

            if (
                int(
                    aggregate_metrics[
                        "n_rows"
                    ]
                )
                != expected_prediction_rows
            ):
                raise ValueError(
                    f"Unexpected rows for "
                    f"{model}/{variant}"
                )

            if (
                int(
                    aggregate_metrics[
                        "n_timestamps"
                    ]
                )
                != expected_prediction_timestamps
            ):
                raise ValueError(
                    "Unexpected timestamp count "
                    f"for {model}/{variant}"
                )

            feature_summary_rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "removed_family": (
                        ""
                        if variant == "full"
                        else variant.removeprefix(
                            "without_"
                        )
                    ),
                    "feature_count": len(
                        features
                    ),
                    "feature_columns": (
                        ",".join(features)
                    ),
                    "source_frozen": (
                        variant == "full"
                    ),
                    "mean_fold_rank_ic": (
                        float(
                            np.mean(
                                fold_test_rank_ics
                            )
                        )
                    ),
                    "worst_fold_rank_ic": (
                        float(
                            np.min(
                                fold_test_rank_ics
                            )
                        )
                    ),
                    "best_fold_rank_ic": (
                        float(
                            np.max(
                                fold_test_rank_ics
                            )
                        )
                    ),
                    "positive_fold_fraction": (
                        float(
                            (
                                np.asarray(
                                    fold_test_rank_ics
                                )
                                > 0
                            ).mean()
                        )
                    ),
                    **aggregate_metrics,
                }
            )

            portfolio_input = (
                combined_test.merge(
                    execution_returns,
                    on=[
                        "timestamp",
                        "symbol",
                    ],
                    how="left",
                    validate="one_to_one",
                )
            )

            if portfolio_input[
                "execution_return"
            ].isna().any():
                raise ValueError(
                    "Missing execution return "
                    f"for {model}/{variant}"
                )

            periods = build_portfolio_periods(
                frame=portfolio_input,
                top_count=top_count,
                interval_minutes=(
                    interval_minutes
                ),
                holding_bars=holding_bars,
            )

            cohort_metrics = (
                summarize_portfolio_cohorts(
                    periods=periods,
                    cost_levels_bps=costs,
                    annualization_periods=(
                        annualization_periods
                    ),
                )
            )

            cohort_metrics.insert(
                0,
                "feature_count",
                len(features),
            )

            cohort_metrics.insert(
                0,
                "removed_family",
                (
                    ""
                    if variant == "full"
                    else variant.removeprefix(
                        "without_"
                    )
                ),
            )

            cohort_metrics.insert(
                0,
                "variant",
                variant,
            )

            cohort_metrics.insert(
                0,
                "model",
                model,
            )

            cohort_frames.append(
                cohort_metrics
            )

    fold_metrics = (
        pd.DataFrame(
            fold_metric_rows
        )
        .sort_values(
            [
                "model",
                "variant",
                "fold",
                "split",
            ]
        )
        .reset_index(drop=True)
    )

    prediction_summary = (
        pd.DataFrame(
            feature_summary_rows
        )
        .sort_values(
            [
                "model",
                "variant",
            ]
        )
        .reset_index(drop=True)
    )

    cohort_metrics = (
        pd.concat(
            cohort_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "variant",
                "cost_bps",
                "cohort",
            ]
        )
        .reset_index(drop=True)
    )

    portfolio_summary = (
        build_ablation_portfolio_summary(
            cohort_metrics=cohort_metrics,
            focus_cost_bps=(
                focus_cost_bps
            ),
            annualization_periods=(
                annualization_periods
            ),
        )
    )

    prediction_summary = (
        add_full_variant_deltas(
            frame=prediction_summary,
            metric_columns=[
                "mean_rank_ic",
                "worst_fold_rank_ic",
            ],
        )
    )

    portfolio_summary = (
        add_full_variant_deltas(
            frame=portfolio_summary,
            metric_columns=[
                "pooled_break_even_cost_bps",
                (
                    "pooled_annualized_net_return_"
                    "at_focus_cost"
                ),
            ],
        )
    )

    feature_summary = (
        prediction_summary.merge(
            portfolio_summary,
            on=[
                "model",
                "variant",
            ],
            validate="one_to_one",
            suffixes=(
                "_prediction",
                "_portfolio",
            ),
        )
        .sort_values(
            [
                "model",
                "mean_rank_ic",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    selection = (
        pd.DataFrame(
            selection_rows
        )
        .sort_values(
            [
                "model",
                "variant",
                "fold",
                "candidate",
            ]
        )
        .reset_index(drop=True)
    )

    verify_full_prediction_summary(
        feature_summary=(
            prediction_summary
        ),
        source_oos=source_oos,
    )

    verify_full_portfolios(
        cohort_metrics=cohort_metrics,
        source_cohorts=source_cohorts,
    )

    model_family_summary = (
        build_model_family_summary(
            source_oos=source_oos,
            source_break_even=(
                source_break_even
            ),
            bootstrap_prediction=(
                bootstrap_prediction
            ),
            bootstrap_portfolio=(
                bootstrap_portfolio
            ),
        )
    )

    evidence_directory = Path(
        config["evidence_dir"]
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics_path = (
        evidence_directory
        / "ablation_feature_fold_metrics.csv"
    )

    selection_path = (
        evidence_directory
        / "ablation_model_selection.csv"
    )

    feature_summary_path = (
        evidence_directory
        / "ablation_feature_summary.csv"
    )

    cohort_metrics_path = (
        evidence_directory
        / "ablation_portfolio_cohort_metrics.csv"
    )

    model_family_path = (
        evidence_directory
        / "ablation_model_family_summary.csv"
    )

    write_csv(
        fold_metrics,
        fold_metrics_path,
    )

    write_csv(
        selection,
        selection_path,
    )

    write_csv(
        feature_summary,
        feature_summary_path,
    )

    write_csv(
        cohort_metrics,
        cohort_metrics_path,
    )

    write_csv(
        model_family_summary,
        model_family_path,
    )

    manifest_path = (
        evidence_directory
        / "ablation_manifest.json"
    )

    evidence_files = [
        fold_metrics_path,
        selection_path,
        feature_summary_path,
        cohort_metrics_path,
        model_family_path,
    ]

    source_files = [
        source_oos_path,
        source_cohort_path,
        source_portfolio_manifest_path,
        source_break_even_path,
        source_baseline_manifest_path,
        source_baseline_selection_path,
        bootstrap_prediction_path,
        bootstrap_portfolio_path,
        source_manifest_path,
        panel_path,
    ]

    manifest = {
        "schema_version": 1,
        "ablation_config": str(
            config_path
        ),
        "ablation_config_sha256": (
            sha256_file(config_path)
        ),
        "data_config": str(
            data_config_path
        ),
        "data_config_sha256": (
            sha256_file(
                data_config_path
            )
        ),
        "research_config": str(
            research_config_path
        ),
        "research_config_sha256": (
            sha256_file(
                research_config_path
            )
        ),
        "baseline_config": str(
            baseline_config_path
        ),
        "baseline_config_sha256": (
            sha256_file(
                baseline_config_path
            )
        ),
        "evaluation_config": str(
            evaluation_config_path
        ),
        "evaluation_config_sha256": (
            sha256_file(
                evaluation_config_path
            )
        ),
        "models": configured_models,
        "model_family_summary_models": (
            model_family_summary[
                "model"
            ].tolist()
        ),
        "feature_columns": (
            FEATURE_COLUMNS
        ),
        "feature_families": (
            feature_families
        ),
        "feature_sets": feature_sets,
        "variant_count": len(
            feature_sets
        ),
        "folds": [
            1,
            2,
            3,
        ],
        "cost_bps": costs,
        "focus_cost_bps": (
            focus_cost_bps
        ),
        "holding_bars": holding_bars,
        "holding_minutes": (
            holding_bars
            * interval_minutes
        ),
        "top_count": top_count,
        "prediction_rows_per_model_variant": (
            expected_prediction_rows
        ),
        "prediction_timestamps_per_model_variant": (
            expected_prediction_timestamps
        ),
        "full_variants_use_frozen_predictions": (
            True
        ),
        "ablated_variants_retrained": (
            True
        ),
        "ablation_predictions_committed": (
            False
        ),
        "full_prediction_metrics_reproduced": (
            True
        ),
        "full_portfolio_metrics_reproduced": (
            True
        ),
        "feature_ablation_scope": (
            "deterministic Ridge and "
            "LightGBM leave-one-family-out "
            "retraining"
        ),
        "model_ablation_scope": (
            "frozen seven-model family "
            "comparison with bootstrap "
            "uncertainty"
        ),
        "source_files": {
            str(path): sha256_file(path)
            for path in source_files
        },
        "evidence_files": {
            str(path): sha256_file(path)
            for path in evidence_files
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
    print(
        "Model and feature ablation "
        "analysis complete"
    )

    print()
    print(
        "Feature-family ablations"
    )

    print(
        feature_summary[
            [
                "model",
                "variant",
                "feature_count",
                "mean_rank_ic",
                "mean_rank_ic_delta_vs_full",
                "worst_fold_rank_ic",
                "pooled_break_even_cost_bps",
                "pooled_break_even_cost_bps_delta_vs_full",
                "pooled_annualized_net_return_at_focus_cost",
            ]
        ]
        .sort_values(
            [
                "model",
                "mean_rank_ic",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(index=False)
    )

    print()
    print(
        "Frozen model-family comparison"
    )

    print(
        model_family_summary[
            [
                "model",
                "model_family",
                "mean_rank_ic",
                "rank_ic_confidence_lower",
                "rank_ic_confidence_upper",
                "pooled_break_even_cost_bps",
                "break_even_confidence_lower",
                "break_even_confidence_upper",
                "rank_ic_delta_vs_mlp",
                "break_even_delta_vs_mlp_bps",
            ]
        ]
        .sort_values(
            "mean_rank_ic",
            ascending=False,
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
