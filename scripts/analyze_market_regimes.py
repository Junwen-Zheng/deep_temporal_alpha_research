from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.data.splits import (
    parse_fold_definition,
)
from deep_alpha.evaluation.metrics import (
    summarize_predictions,
)
from deep_alpha.evaluation.portfolio import (
    build_portfolio_periods,
    summarize_portfolio_cohorts,
)
from deep_alpha.evaluation.regimes import (
    REGIME_COLUMNS,
    build_fold_regime_assignments,
    build_market_state_panel,
    build_regime_model_summary,
    melt_regime_assignments,
)
from deep_alpha.evaluation.stability import (
    summarize_grouped_predictions,
    summarize_temporal_portfolio_groups,
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


def load_model_predictions(
    model_name: str,
    prediction_directory: Path,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    frames = []
    records = []

    for fold in [
        1,
        2,
        3,
    ]:
        path = (
            prediction_directory
            / (
                f"{model_name}_fold_"
                f"{fold}_test.parquet"
            )
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_parquet(
            path,
            columns=[
                "timestamp",
                "symbol",
                "raw_target",
                "target",
                "prediction",
            ],
        )

        frame["fold"] = fold
        frames.append(frame)

        records.append(
            {
                "model": model_name,
                "fold": fold,
                "path": str(path),
                "rows": len(frame),
                "timestamps": (
                    frame[
                        "timestamp"
                    ].nunique()
                ),
                "sha256": sha256_file(path),
            }
        )

    combined = (
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

    if combined.duplicated(
        [
            "timestamp",
            "symbol",
        ]
    ).any():
        raise ValueError(
            f"Duplicate prediction keys "
            f"for {model_name}"
        )

    return combined, records


def verify_prediction_metrics(
    model_name: str,
    predictions: pd.DataFrame,
    source_metrics: pd.DataFrame,
) -> None:
    actual = summarize_predictions(
        predictions
    )

    source = source_metrics.loc[
        source_metrics["model"]
        == model_name
    ]

    if len(source) != 1:
        raise ValueError(
            f"Expected one source metric row "
            f"for {model_name}"
        )

    source_row = source.iloc[0]

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
        if not np.isclose(
            float(actual[column]),
            float(source_row[column]),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                f"Aggregate predictive "
                f"{column} differs for "
                f"{model_name}"
            )

    for column in [
        "n_rows",
        "n_timestamps",
    ]:
        if (
            int(actual[column])
            != int(source_row[column])
        ):
            raise ValueError(
                f"Aggregate predictive "
                f"{column} differs for "
                f"{model_name}"
            )


def verify_portfolio_metrics(
    model_name: str,
    periods: pd.DataFrame,
    source_metrics: pd.DataFrame,
    annualization_periods: int,
) -> None:
    source = source_metrics.loc[
        source_metrics["model"]
        == model_name
    ].copy()

    if source.empty:
        raise ValueError(
            f"No source portfolio rows "
            f"for {model_name}"
        )

    source_costs = sorted(
        set(
            source[
                "cost_bps"
            ].astype(float)
        )
    )

    reproduced = (
        summarize_portfolio_cohorts(
            periods=periods,
            cost_levels_bps=source_costs,
            annualization_periods=(
                annualization_periods
            ),
        )
    )

    comparison = source.merge(
        reproduced,
        on=[
            "cost_bps",
            "cohort",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_source",
            "_reproduced",
        ),
    )

    if len(comparison) != len(source):
        raise ValueError(
            f"Portfolio reproduction is "
            f"incomplete for {model_name}"
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
            f"Portfolio period counts differ "
            f"for {model_name}"
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
        if not np.allclose(
            comparison[
                f"{column}_source"
            ].to_numpy(dtype=np.float64),
            comparison[
                f"{column}_reproduced"
            ].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"Portfolio {column} differs "
                f"for {model_name}"
            )


def build_long_prediction_regimes(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    for dimension, column in (
        REGIME_COLUMNS.items()
    ):
        working = predictions.copy()

        working[
            "regime_dimension"
        ] = dimension

        working["regime"] = working[
            column
        ]

        frames.append(working)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_long_portfolio_regimes(
    periods: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    for dimension, column in (
        REGIME_COLUMNS.items()
    ):
        working = periods.copy()

        working[
            "regime_dimension"
        ] = dimension

        working["regime"] = working[
            column
        ]

        frames.append(working)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def main() -> None:
    config_path = Path(
        "configs/regimes.yaml"
    )

    config = load_yaml(
        config_path
    )["regimes"]

    data_config_path = Path(
        config["data_config"]
    )

    research_config_path = Path(
        config["research_config"]
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

    source_manifest_path = Path(
        config[
            "source_portfolio_manifest"
        ]
    )

    source_oos = pd.read_csv(
        source_oos_path
    )

    source_cohorts = pd.read_csv(
        source_cohort_path
    )

    source_manifest = json.loads(
        source_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    models = [
        str(model)
        for model in evaluation_config[
            "models"
        ]
    ]

    symbols = [
        str(symbol)
        for symbol in data_config[
            "symbols"
        ]
    ]

    expected_symbol_count = int(
        config["expected_symbol_count"]
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

    state_window_bars = int(
        config["state_window_bars"]
    )

    lower_quantile = float(
        config["lower_quantile"]
    )

    upper_quantile = float(
        config["upper_quantile"]
    )

    costs = [
        float(value)
        for value in config["cost_bps"]
    ]

    focus_cost_bps = float(
        config["focus_cost_bps"]
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
            "Focus cost must be included "
            "in regime costs"
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
        actual_value = source_manifest.get(
            key
        )

        if actual_value != expected_value:
            raise ValueError(
                "Source portfolio manifest "
                f"{key} differs: "
                f"{actual_value} versus "
                f"{expected_value}"
            )

    if set(
        source_manifest.get(
            "models",
            [],
        )
    ) != set(models):
        raise ValueError(
            "Source portfolio model coverage "
            "differs"
        )

    processed_directory = Path(
        data_config["processed_dir"]
    )

    state_frames = []
    execution_frames = []

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(symbols):02d}] "
            f"Loading bars for {symbol}",
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

        state_frames.append(
            pd.DataFrame(
                {
                    "timestamp": (
                        bars["timestamp"]
                    ),
                    "symbol": symbol,
                    "close": bars["close"],
                }
            )
        )

        execution_frames.append(
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

    market_states = (
        build_market_state_panel(
            bars=pd.concat(
                state_frames,
                ignore_index=True,
            ),
            window_bars=state_window_bars,
            expected_symbol_count=(
                expected_symbol_count
            ),
        )
    )

    execution_returns = (
        pd.concat(
            execution_frames,
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

    fold_definitions = {
        int(fold.fold_id): fold
        for fold in (
            parse_fold_definition(
                configuration
            )
            for configuration
            in research_config["folds"]
        )
    }

    if set(
        fold_definitions
    ) != {
        1,
        2,
        3,
    }:
        raise ValueError(
            "Expected exactly three folds"
        )

    reference_model = models[0]

    reference_directory = Path(
        evaluation_config["models"][
            reference_model
        ]["directory"]
    )

    assignment_frames = []
    threshold_records = []

    for fold_id in [
        1,
        2,
        3,
    ]:
        path = (
            reference_directory
            / (
                f"{reference_model}_fold_"
                f"{fold_id}_test.parquet"
            )
        )

        reference_fold = pd.read_parquet(
            path,
            columns=[
                "timestamp",
            ],
        )

        assignments, threshold_record = (
            build_fold_regime_assignments(
                market_states=market_states,
                fold=fold_definitions[
                    fold_id
                ],
                test_timestamps=(
                    reference_fold[
                        "timestamp"
                    ]
                ),
                lower_quantile=(
                    lower_quantile
                ),
                upper_quantile=(
                    upper_quantile
                ),
            )
        )

        assignment_frames.append(
            assignments
        )

        threshold_records.append(
            threshold_record
        )

    assignments = (
        pd.concat(
            assignment_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "fold",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    thresholds = (
        pd.DataFrame(
            threshold_records
        )
        .sort_values("fold")
        .reset_index(drop=True)
    )

    if assignments.duplicated(
        [
            "fold",
            "timestamp",
        ]
    ).any():
        raise ValueError(
            "Duplicate regime assignment keys"
        )

    if (
        len(assignments)
        != expected_prediction_timestamps
    ):
        raise ValueError(
            "Unexpected regime assignment count"
        )

    long_assignments = (
        melt_regime_assignments(
            assignments
        )
    )

    prediction_metric_frames = []
    portfolio_metric_frames = []
    prediction_records = []

    for model_name, model_config in (
        evaluation_config["models"].items()
    ):
        model_name = str(model_name)

        print()
        print(
            f"=== Market regimes: "
            f"{model_name} ===",
            flush=True,
        )

        predictions, records = (
            load_model_predictions(
                model_name=model_name,
                prediction_directory=Path(
                    model_config["directory"]
                ),
            )
        )

        prediction_records.extend(records)

        if (
            len(predictions)
            != expected_prediction_rows
        ):
            raise ValueError(
                f"Unexpected prediction rows "
                f"for {model_name}"
            )

        if (
            predictions[
                "timestamp"
            ].nunique()
            != expected_prediction_timestamps
        ):
            raise ValueError(
                "Unexpected prediction "
                f"timestamps for {model_name}"
            )

        verify_prediction_metrics(
            model_name=model_name,
            predictions=predictions,
            source_metrics=source_oos,
        )

        predictions = predictions.merge(
            assignments[
                [
                    "fold",
                    "timestamp",
                    *REGIME_COLUMNS.values(),
                ]
            ],
            on=[
                "fold",
                "timestamp",
            ],
            how="left",
            validate="many_to_one",
        )

        if predictions[
            list(
                REGIME_COLUMNS.values()
            )
        ].isna().any().any():
            raise ValueError(
                f"Missing regime labels "
                f"for {model_name}"
            )

        long_predictions = (
            build_long_prediction_regimes(
                predictions
            )
        )

        prediction_metrics = (
            summarize_grouped_predictions(
                long_predictions,
                group_columns=[
                    "fold",
                    "regime_dimension",
                    "regime",
                ],
            )
        )

        prediction_metrics.insert(
            0,
            "model",
            model_name,
        )

        prediction_metric_frames.append(
            prediction_metrics
        )

        portfolio_input = (
            predictions.merge(
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
                f"Missing execution returns "
                f"for {model_name}"
            )

        periods = build_portfolio_periods(
            frame=portfolio_input,
            top_count=top_count,
            interval_minutes=(
                interval_minutes
            ),
            holding_bars=holding_bars,
        )

        verify_portfolio_metrics(
            model_name=model_name,
            periods=periods,
            source_metrics=source_cohorts,
            annualization_periods=(
                annualization_periods
            ),
        )

        timestamp_metadata = (
            assignments[
                [
                    "fold",
                    "timestamp",
                    *REGIME_COLUMNS.values(),
                ]
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        periods = periods.merge(
            timestamp_metadata,
            on="timestamp",
            how="left",
            validate="many_to_one",
        )

        if periods[
            [
                "fold",
                *REGIME_COLUMNS.values(),
            ]
        ].isna().any().any():
            raise ValueError(
                f"Portfolio periods lack "
                f"regime metadata for "
                f"{model_name}"
            )

        long_periods = (
            build_long_portfolio_regimes(
                periods
            )
        )

        portfolio_metrics = (
            summarize_temporal_portfolio_groups(
                periods=long_periods,
                group_columns=[
                    "fold",
                    "regime_dimension",
                    "regime",
                ],
                cost_levels_bps=costs,
                annualization_periods=(
                    annualization_periods
                ),
            )
        )

        portfolio_metrics.insert(
            0,
            "model",
            model_name,
        )

        portfolio_metric_frames.append(
            portfolio_metrics
        )

    prediction_metrics = (
        pd.concat(
            prediction_metric_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "fold",
                "regime_dimension",
                "regime",
            ]
        )
        .reset_index(drop=True)
    )

    portfolio_metrics = (
        pd.concat(
            portfolio_metric_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "fold",
                "regime_dimension",
                "regime",
                "cost_bps",
            ]
        )
        .reset_index(drop=True)
    )

    model_summary = (
        build_regime_model_summary(
            prediction_metrics=(
                prediction_metrics
            ),
            portfolio_metrics=(
                portfolio_metrics
            ),
            focus_cost_bps=(
                focus_cost_bps
            ),
        )
    )

    expected_regime_labels = {
        "direction": {
            "down",
            "flat",
            "up",
        },
        "volatility": {
            "low",
            "medium",
            "high",
        },
        "dispersion": {
            "low",
            "medium",
            "high",
        },
    }

    for dimension, labels in (
        expected_regime_labels.items()
    ):
        assignment_labels = set(
            long_assignments.loc[
                long_assignments[
                    "regime_dimension"
                ]
                == dimension,
                "regime",
            ]
        )

        if assignment_labels != labels:
            raise ValueError(
                f"Regime coverage differs "
                f"for {dimension}: "
                f"{assignment_labels}"
            )

        prediction_counts = (
            prediction_metrics.loc[
                prediction_metrics[
                    "regime_dimension"
                ]
                == dimension
            ]
            .groupby(
                [
                    "model",
                    "fold",
                ]
            )["regime"]
            .nunique()
        )

        if not prediction_counts.eq(3).all():
            raise ValueError(
                f"Prediction regime cells are "
                f"incomplete for {dimension}"
            )

    if not (
        portfolio_metrics[
            "cohort_count"
        ]
        .eq(holding_bars)
        .all()
    ):
        raise ValueError(
            "Regime portfolio cells do not "
            "contain all cohorts"
        )

    evidence_directory = Path(
        config["evidence_dir"]
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds_path = (
        evidence_directory
        / "regime_thresholds.csv"
    )

    assignments_path = (
        evidence_directory
        / "regime_assignments.csv"
    )

    prediction_path = (
        evidence_directory
        / "regime_prediction_metrics.csv"
    )

    portfolio_path = (
        evidence_directory
        / "regime_portfolio_metrics.csv"
    )

    summary_path = (
        evidence_directory
        / "regime_model_summary.csv"
    )

    write_csv(
        thresholds,
        thresholds_path,
    )

    write_csv(
        assignments,
        assignments_path,
    )

    write_csv(
        prediction_metrics,
        prediction_path,
    )

    write_csv(
        portfolio_metrics,
        portfolio_path,
    )

    write_csv(
        model_summary,
        summary_path,
    )

    manifest_path = (
        evidence_directory
        / "regime_manifest.json"
    )

    evidence_files = [
        thresholds_path,
        assignments_path,
        prediction_path,
        portfolio_path,
        summary_path,
    ]

    manifest = {
        "schema_version": 1,
        "regime_config": str(
            config_path
        ),
        "regime_config_sha256": (
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
        "evaluation_config": str(
            evaluation_config_path
        ),
        "evaluation_config_sha256": (
            sha256_file(
                evaluation_config_path
            )
        ),
        "source_oos_metrics": str(
            source_oos_path
        ),
        "source_oos_metrics_sha256": (
            sha256_file(
                source_oos_path
            )
        ),
        "source_portfolio_cohort_metrics": (
            str(source_cohort_path)
        ),
        "source_portfolio_cohort_metrics_sha256": (
            sha256_file(
                source_cohort_path
            )
        ),
        "source_portfolio_manifest": str(
            source_manifest_path
        ),
        "source_portfolio_manifest_sha256": (
            sha256_file(
                source_manifest_path
            )
        ),
        "models": models,
        "symbols": symbols,
        "folds": [
            1,
            2,
            3,
        ],
        "state_window_bars": (
            state_window_bars
        ),
        "state_window_minutes": (
            state_window_bars
            * interval_minutes
        ),
        "lower_quantile": (
            lower_quantile
        ),
        "upper_quantile": (
            upper_quantile
        ),
        "threshold_calibration_rule": (
            "all available market-state "
            "timestamps at or before each "
            "fold validation_end"
        ),
        "regime_dimensions": {
            "direction": [
                "down",
                "flat",
                "up",
            ],
            "volatility": [
                "low",
                "medium",
                "high",
            ],
            "dispersion": [
                "low",
                "medium",
                "high",
            ],
        },
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
        "prediction_rows_per_model": (
            expected_prediction_rows
        ),
        "prediction_timestamps_per_model": (
            expected_prediction_timestamps
        ),
        "aggregate_prediction_metrics_reproduced": (
            True
        ),
        "aggregate_portfolio_metrics_reproduced": (
            True
        ),
        "models_retrained": False,
        "regime_thresholds_use_test_data": (
            False
        ),
        "prediction_files": sorted(
            prediction_records,
            key=lambda record: (
                record["model"],
                record["fold"],
            ),
        ),
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
        "Market-regime analysis complete"
    )

    print()
    print(
        "Regime model summary"
    )

    print(
        model_summary[
            [
                "model",
                "regime_dimension",
                "worst_cell_rank_ic",
                "positive_rank_ic_cell_fraction",
                "worst_break_even_cost_bps",
                "positive_break_even_cell_fraction",
                "worst_break_even_regime",
                "positive_cell_fraction_at_focus_cost",
            ]
        ]
        .sort_values(
            [
                "regime_dimension",
                "worst_cell_rank_ic",
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
        "Worst predictive regime cell "
        "per model"
    )

    worst_prediction = (
        prediction_metrics.sort_values(
            [
                "model",
                "mean_rank_ic",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .groupby(
            "model",
            as_index=False,
            sort=True,
        )
        .first()
    )

    print(
        worst_prediction[
            [
                "model",
                "fold",
                "regime_dimension",
                "regime",
                "mean_rank_ic",
                "rank_ic_ir",
                "positive_timestamp_fraction",
                "n_timestamps",
            ]
        ]
        .sort_values(
            "mean_rank_ic"
        )
        .to_string(index=False)
    )

    print()
    print(
        "Worst one-basis-point regime cell "
        "per model"
    )

    focus_metrics = (
        portfolio_metrics.loc[
            np.isclose(
                portfolio_metrics[
                    "cost_bps"
                ].to_numpy(
                    dtype=np.float64
                ),
                focus_cost_bps,
                rtol=0.0,
                atol=1e-12,
            )
        ]
    )

    worst_economic = (
        focus_metrics.sort_values(
            [
                "model",
                "pooled_annualized_net_return",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .groupby(
            "model",
            as_index=False,
            sort=True,
        )
        .first()
    )

    print(
        worst_economic[
            [
                "model",
                "fold",
                "regime_dimension",
                "regime",
                "break_even_cost_bps",
                "pooled_annualized_net_return",
                "positive_cohort_fraction",
                "mean_turnover",
            ]
        ]
        .sort_values(
            "pooled_annualized_net_return"
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
