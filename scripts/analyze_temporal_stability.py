from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.evaluation.metrics import (
    summarize_predictions,
)
from deep_alpha.evaluation.portfolio import (
    build_portfolio_periods,
    summarize_portfolio_cohorts,
)
from deep_alpha.evaluation.stability import (
    add_calendar_month,
    build_temporal_stability_summary,
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
            f"Duplicate out-of-sample keys "
            f"for {model_name}"
        )

    return combined, records


def compare_aggregate_metrics(
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
                f"Aggregate {column} differs "
                f"for {model_name}"
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
                f"Aggregate {column} differs "
                f"for {model_name}"
            )


def compare_portfolio_metrics(
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


def main() -> None:
    stability_config_path = Path(
        "configs/stability.yaml"
    )

    stability_config = load_yaml(
        stability_config_path
    )["stability"]

    data_config_path = Path(
        stability_config[
            "data_config"
        ]
    )

    evaluation_config_path = Path(
        stability_config[
            "evaluation_config"
        ]
    )

    data_config = load_yaml(
        data_config_path
    )["data"]

    evaluation_config = load_yaml(
        evaluation_config_path
    )["evaluation"]

    source_oos_path = Path(
        stability_config[
            "source_oos_metrics"
        ]
    )

    source_cohort_path = Path(
        stability_config[
            "source_portfolio_cohort_metrics"
        ]
    )

    source_manifest_path = Path(
        stability_config[
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
        stability_config[
            "expected_symbol_count"
        ]
    )

    expected_prediction_rows = int(
        stability_config[
            "expected_prediction_rows"
        ]
    )

    expected_prediction_timestamps = int(
        stability_config[
            "expected_prediction_timestamps"
        ]
    )

    if len(symbols) != expected_symbol_count:
        raise ValueError(
            "Configured symbol count differs "
            "from the data universe"
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
        for value in stability_config[
            "cost_bps"
        ]
    ]

    focus_cost_bps = float(
        stability_config[
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
            "focus_cost_bps must be "
            "included in cost_bps"
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
            "Source manifest model coverage "
            "differs"
        )

    processed_directory = Path(
        data_config["processed_dir"]
    )

    execution_returns = (
        load_execution_returns(
            symbols=symbols,
            processed_directory=(
                processed_directory
            ),
            holding_bars=holding_bars,
        )
    )

    fold_prediction_frames = []
    month_prediction_frames = []
    fold_portfolio_frames = []
    month_portfolio_frames = []
    prediction_records = []

    reference_targets: (
        pd.DataFrame | None
    ) = None

    for model_name, configuration in (
        evaluation_config["models"].items()
    ):
        model_name = str(model_name)

        print()
        print(
            f"=== Temporal stability: "
            f"{model_name} ===",
            flush=True,
        )

        (
            predictions,
            records,
        ) = load_model_predictions(
            model_name=model_name,
            prediction_directory=Path(
                configuration["directory"]
            ),
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
                f"Unexpected timestamp count "
                f"for {model_name}"
            )

        target_columns = [
            "timestamp",
            "symbol",
            "raw_target",
            "target",
            "fold",
        ]

        candidate_targets = (
            predictions[
                target_columns
            ].copy()
        )

        if reference_targets is None:
            reference_targets = (
                candidate_targets
            )
        else:
            if not np.array_equal(
                pd.DatetimeIndex(
                    reference_targets[
                        "timestamp"
                    ]
                ).asi8,
                pd.DatetimeIndex(
                    candidate_targets[
                        "timestamp"
                    ]
                ).asi8,
            ):
                raise ValueError(
                    f"Timestamp alignment differs "
                    f"for {model_name}"
                )

            for column in [
                "symbol",
                "fold",
            ]:
                if not np.array_equal(
                    reference_targets[
                        column
                    ].to_numpy(),
                    candidate_targets[
                        column
                    ].to_numpy(),
                ):
                    raise ValueError(
                        f"{column} alignment differs "
                        f"for {model_name}"
                    )

            for column in [
                "raw_target",
                "target",
            ]:
                if not np.allclose(
                    reference_targets[
                        column
                    ].to_numpy(
                        dtype=np.float64
                    ),
                    candidate_targets[
                        column
                    ].to_numpy(
                        dtype=np.float64
                    ),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"{column} differs for "
                        f"{model_name}"
                    )

        compare_aggregate_metrics(
            model_name=model_name,
            predictions=predictions,
            source_metrics=source_oos,
        )

        predictions = add_calendar_month(
            predictions
        )

        fold_prediction = (
            summarize_grouped_predictions(
                predictions,
                group_columns=[
                    "fold",
                ],
            )
        )

        fold_prediction.insert(
            0,
            "model",
            model_name,
        )

        month_prediction = (
            summarize_grouped_predictions(
                predictions,
                group_columns=[
                    "fold",
                    "month",
                ],
            )
        )

        month_prediction.insert(
            0,
            "model",
            model_name,
        )

        fold_prediction_frames.append(
            fold_prediction
        )

        month_prediction_frames.append(
            month_prediction
        )

        portfolio_input = predictions.merge(
            execution_returns,
            on=[
                "timestamp",
                "symbol",
            ],
            how="left",
            validate="one_to_one",
        )

        if (
            portfolio_input[
                "execution_return"
            ].isna().any()
        ):
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

        compare_portfolio_metrics(
            model_name=model_name,
            periods=periods,
            source_metrics=source_cohorts,
            annualization_periods=(
                annualization_periods
            ),
        )

        timestamp_metadata = (
            predictions[
                [
                    "timestamp",
                    "fold",
                    "month",
                ]
            ]
            .drop_duplicates()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if (
            timestamp_metadata[
                "timestamp"
            ].duplicated().any()
        ):
            raise ValueError(
                "Timestamp metadata is not unique"
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
                "month",
            ]
        ].isna().any().any():
            raise ValueError(
                "Portfolio periods lack "
                "temporal metadata"
            )

        fold_portfolio = (
            summarize_temporal_portfolio_groups(
                periods=periods,
                group_columns=[
                    "fold",
                ],
                cost_levels_bps=costs,
                annualization_periods=(
                    annualization_periods
                ),
            )
        )

        fold_portfolio.insert(
            0,
            "model",
            model_name,
        )

        month_portfolio = (
            summarize_temporal_portfolio_groups(
                periods=periods,
                group_columns=[
                    "fold",
                    "month",
                ],
                cost_levels_bps=costs,
                annualization_periods=(
                    annualization_periods
                ),
            )
        )

        month_portfolio.insert(
            0,
            "model",
            model_name,
        )

        fold_portfolio_frames.append(
            fold_portfolio
        )

        month_portfolio_frames.append(
            month_portfolio
        )

    fold_prediction_metrics = (
        pd.concat(
            fold_prediction_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "fold",
            ]
        )
        .reset_index(drop=True)
    )

    month_prediction_metrics = (
        pd.concat(
            month_prediction_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "fold",
                "month",
            ]
        )
        .reset_index(drop=True)
    )

    fold_portfolio_metrics = (
        pd.concat(
            fold_portfolio_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "fold",
                "cost_bps",
            ]
        )
        .reset_index(drop=True)
    )

    month_portfolio_metrics = (
        pd.concat(
            month_portfolio_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "fold",
                "month",
                "cost_bps",
            ]
        )
        .reset_index(drop=True)
    )

    model_summary = (
        build_temporal_stability_summary(
            fold_prediction_metrics=(
                fold_prediction_metrics
            ),
            month_prediction_metrics=(
                month_prediction_metrics
            ),
            fold_portfolio_metrics=(
                fold_portfolio_metrics
            ),
            month_portfolio_metrics=(
                month_portfolio_metrics
            ),
            focus_cost_bps=(
                focus_cost_bps
            ),
        )
    )

    reference_model = models[0]

    reference_segments = (
        month_prediction_metrics.loc[
            month_prediction_metrics[
                "model"
            ]
            == reference_model,
            [
                "fold",
                "month",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            [
                "fold",
                "month",
            ]
        )
        .reset_index(drop=True)
    )

    for model in models[1:]:
        candidate_segments = (
            month_prediction_metrics.loc[
                month_prediction_metrics[
                    "model"
                ]
                == model,
                [
                    "fold",
                    "month",
                ],
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "fold",
                    "month",
                ]
            )
            .reset_index(drop=True)
        )

        if not reference_segments.equals(
            candidate_segments
        ):
            raise ValueError(
                f"Month segments differ "
                f"for {model}"
            )

    if not (
        fold_portfolio_metrics[
            "cohort_count"
        ]
        .eq(holding_bars)
        .all()
    ):
        raise ValueError(
            "Fold portfolio evidence does "
            "not contain all cohorts"
        )

    if not (
        month_portfolio_metrics[
            "cohort_count"
        ]
        .eq(holding_bars)
        .all()
    ):
        raise ValueError(
            "Monthly portfolio evidence does "
            "not contain all cohorts"
        )

    evidence_directory = Path(
        stability_config[
            "evidence_dir"
        ]
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_prediction_path = (
        evidence_directory
        / "stability_fold_prediction_metrics.csv"
    )

    month_prediction_path = (
        evidence_directory
        / "stability_month_prediction_metrics.csv"
    )

    fold_portfolio_path = (
        evidence_directory
        / "stability_fold_portfolio_metrics.csv"
    )

    month_portfolio_path = (
        evidence_directory
        / "stability_month_portfolio_metrics.csv"
    )

    model_summary_path = (
        evidence_directory
        / "stability_model_summary.csv"
    )

    write_csv(
        fold_prediction_metrics,
        fold_prediction_path,
    )

    write_csv(
        month_prediction_metrics,
        month_prediction_path,
    )

    write_csv(
        fold_portfolio_metrics,
        fold_portfolio_path,
    )

    write_csv(
        month_portfolio_metrics,
        month_portfolio_path,
    )

    write_csv(
        model_summary,
        model_summary_path,
    )

    manifest_path = (
        evidence_directory
        / "stability_manifest.json"
    )

    evidence_files = [
        fold_prediction_path,
        month_prediction_path,
        fold_portfolio_path,
        month_portfolio_path,
        model_summary_path,
    ]

    month_segments = [
        {
            "fold": int(row.fold),
            "month": str(row.month),
        }
        for row in (
            reference_segments.itertuples(
                index=False
            )
        )
    ]

    manifest = {
        "schema_version": 1,
        "stability_config": str(
            stability_config_path
        ),
        "stability_config_sha256": (
            sha256_file(
                stability_config_path
            )
        ),
        "data_config": str(
            data_config_path
        ),
        "data_config_sha256": (
            sha256_file(
                data_config_path
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
        "folds": [
            1,
            2,
            3,
        ],
        "month_segments": (
            month_segments
        ),
        "month_segment_count": len(
            month_segments
        ),
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
        "annualization_periods": (
            annualization_periods
        ),
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
        "monthly_sharpe_aggregation_performed": (
            False
        ),
        "month_assignment": (
            "prediction timestamp UTC month"
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
        "Temporal stability analysis complete"
    )

    print()
    print("Model stability summary")

    print(
        model_summary[
            [
                "model",
                "mean_fold_rank_ic",
                "worst_fold_rank_ic",
                "positive_fold_fraction",
                "mean_month_rank_ic",
                "worst_month_rank_ic",
                "positive_month_rank_ic_fraction",
                "mean_month_break_even_cost_bps",
                "worst_month_break_even_cost_bps",
                "positive_month_fraction_at_focus_cost",
            ]
        ]
        .sort_values(
            "mean_month_rank_ic",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()
    print(
        "Worst predictive month per model"
    )

    worst_months = (
        month_prediction_metrics.sort_values(
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
        worst_months[
            [
                "model",
                "fold",
                "month",
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
        "Worst one-basis-point month "
        "per model"
    )

    focus_months = (
        month_portfolio_metrics.loc[
            np.isclose(
                month_portfolio_metrics[
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

    worst_economic_months = (
        focus_months.sort_values(
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
        worst_economic_months[
            [
                "model",
                "fold",
                "month",
                "break_even_cost_bps",
                "pooled_annualized_net_return",
                "worst_cohort_annualized_net_return",
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
