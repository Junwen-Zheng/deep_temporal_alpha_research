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
from deep_alpha.evaluation.horizons import (
    annualization_periods_for_horizon,
    build_cross_sectional_horizon_target,
    build_horizon_break_even,
    build_horizon_portfolio_summary,
    build_symbol_horizon_reference,
    validate_horizon_bars,
)
from deep_alpha.evaluation.metrics import (
    summarize_predictions,
)
from deep_alpha.evaluation.portfolio import (
    build_portfolio_periods,
    summarize_portfolio_cohorts,
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


def load_prediction_fold(
    model_name: str,
    fold: int,
    directory: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    path = (
        directory
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
            "prediction",
        ],
    )

    frame = (
        frame.sort_values(
            [
                "timestamp",
                "symbol",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    record = {
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

    return frame, record


def main() -> None:
    sensitivity_config_path = Path(
        "configs/horizon_sensitivity.yaml"
    )

    sensitivity_config = load_yaml(
        sensitivity_config_path
    )["horizon_sensitivity"]

    data_config_path = Path(
        sensitivity_config[
            "data_config"
        ]
    )

    research_config_path = Path(
        sensitivity_config[
            "research_config"
        ]
    )

    evaluation_config_path = Path(
        sensitivity_config[
            "evaluation_config"
        ]
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
        sensitivity_config[
            "source_oos_metrics"
        ]
    )

    source_cohort_path = Path(
        sensitivity_config[
            "source_portfolio_cohort_metrics"
        ]
    )

    source_manifest_path = Path(
        sensitivity_config[
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

    interval_minutes = int(
        evaluation_config[
            "interval_minutes"
        ]
    )

    original_horizon_bars = int(
        evaluation_config[
            "holding_bars"
        ]
    )

    horizons = validate_horizon_bars(
        [
            int(value)
            for value in sensitivity_config[
                "horizon_bars"
            ]
        ]
    )

    if original_horizon_bars not in horizons:
        raise ValueError(
            "The original evaluation horizon "
            "must be included"
        )

    cost_levels = [
        float(value)
        for value in sensitivity_config[
            "cost_bps"
        ]
    ]

    top_count = int(
        sensitivity_config[
            "top_count"
        ]
    )

    entry_offset_bars = int(
        sensitivity_config[
            "entry_offset_bars"
        ]
    )

    expected_symbol_count = int(
        sensitivity_config[
            "expected_symbol_count"
        ]
    )

    symbols = [
        str(symbol)
        for symbol in data_config[
            "symbols"
        ]
    ]

    if len(symbols) != expected_symbol_count:
        raise ValueError(
            "Symbol universe size differs "
            "from the horizon configuration"
        )

    model_configurations = (
        evaluation_config["models"]
    )

    model_names = [
        str(model)
        for model in model_configurations
    ]

    manifest_expectations = {
        "symbol_count": (
            expected_symbol_count
        ),
        "holding_bars": (
            original_horizon_bars
        ),
        "interval_minutes": (
            interval_minutes
        ),
        "top_count": top_count,
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
    ) != set(model_names):
        raise ValueError(
            "Source portfolio models differ "
            "from the current configuration"
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

    processed_directory = Path(
        data_config["processed_dir"]
    )

    print(
        "Loading normalized bar cache",
        flush=True,
    )

    bars_by_symbol = {}

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(symbols):02d}] "
            f"Loading {symbol}",
            flush=True,
        )

        path = (
            processed_directory
            / "bars"
            / f"symbol={symbol}"
            / "part-00000.parquet"
        )

        bars_by_symbol[symbol] = (
            pd.read_parquet(
                path,
                columns=[
                    "timestamp",
                    "open",
                    "close",
                ],
            )
        )

    fold_metric_rows = []
    model_metric_rows = []
    portfolio_frames = []
    prediction_records = []

    row_counts_by_horizon = {}
    timestamp_counts_by_horizon = {}

    first_horizon = horizons[0]

    for horizon_bars in horizons:
        horizon_minutes = (
            horizon_bars
            * interval_minutes
        )

        annualization_periods = (
            annualization_periods_for_horizon(
                interval_minutes=(
                    interval_minutes
                ),
                horizon_bars=(
                    horizon_bars
                ),
            )
        )

        print()
        print(
            f"=== Horizon {horizon_minutes} "
            "minutes ===",
            flush=True,
        )

        reference_frames = []

        for symbol in symbols:
            reference = (
                build_symbol_horizon_reference(
                    bars=bars_by_symbol[
                        symbol
                    ],
                    horizon_bars=(
                        horizon_bars
                    ),
                    interval_minutes=(
                        interval_minutes
                    ),
                    entry_offset_bars=(
                        entry_offset_bars
                    ),
                )
            )

            reference.insert(
                1,
                "symbol",
                symbol,
            )

            reference_frames.append(
                reference
            )

        reference = (
            build_cross_sectional_horizon_target(
                frame=pd.concat(
                    reference_frames,
                    ignore_index=True,
                ),
                expected_symbol_count=(
                    expected_symbol_count
                ),
            )
        )

        reference = (
            reference.sort_values(
                [
                    "timestamp",
                    "symbol",
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        horizon_model_counts = []

        for model_name, configuration in (
            model_configurations.items()
        ):
            model_name = str(model_name)

            print(
                f"Evaluating {model_name}",
                flush=True,
            )

            aligned_folds = []

            for fold_id in [
                1,
                2,
                3,
            ]:
                (
                    predictions,
                    prediction_record,
                ) = load_prediction_fold(
                    model_name=model_name,
                    fold=fold_id,
                    directory=Path(
                        configuration[
                            "directory"
                        ]
                    ),
                )

                if (
                    horizon_bars
                    == first_horizon
                ):
                    prediction_records.append(
                        prediction_record
                    )

                fold_definition = (
                    fold_definitions[
                        fold_id
                    ]
                )

                valid_reference = (
                    reference.loc[
                        reference[
                            "label_end_timestamp"
                        ]
                        <= fold_definition.test_end
                    ]
                )

                aligned = predictions.merge(
                    valid_reference,
                    on=[
                        "timestamp",
                        "symbol",
                    ],
                    how="inner",
                    validate="one_to_one",
                )

                aligned = (
                    aligned.sort_values(
                        [
                            "timestamp",
                            "symbol",
                        ],
                        kind="mergesort",
                    )
                    .reset_index(drop=True)
                )

                if aligned.empty:
                    raise ValueError(
                        f"No aligned rows for "
                        f"{model_name}, fold "
                        f"{fold_id}, horizon "
                        f"{horizon_bars}"
                    )

                cross_section_sizes = (
                    aligned.groupby(
                        "timestamp",
                        sort=False,
                    ).size()
                )

                if not cross_section_sizes.eq(
                    expected_symbol_count
                ).all():
                    raise ValueError(
                        "Incomplete aligned "
                        "cross-sections"
                    )

                metrics = (
                    summarize_predictions(
                        aligned
                    )
                )

                fold_metric_rows.append(
                    {
                        "model": model_name,
                        "horizon_bars": (
                            horizon_bars
                        ),
                        "horizon_minutes": (
                            horizon_minutes
                        ),
                        "fold": fold_id,
                        **metrics,
                    }
                )

                aligned_folds.append(
                    aligned
                )

            combined = (
                pd.concat(
                    aligned_folds,
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

            combined_metrics = (
                summarize_predictions(
                    combined
                )
            )

            model_metric_rows.append(
                {
                    "model": model_name,
                    "horizon_bars": (
                        horizon_bars
                    ),
                    "horizon_minutes": (
                        horizon_minutes
                    ),
                    **combined_metrics,
                }
            )

            horizon_model_counts.append(
                (
                    model_name,
                    int(
                        combined_metrics[
                            "n_rows"
                        ]
                    ),
                    int(
                        combined_metrics[
                            "n_timestamps"
                        ]
                    ),
                )
            )

            periods = (
                build_portfolio_periods(
                    frame=combined,
                    top_count=top_count,
                    interval_minutes=(
                        interval_minutes
                    ),
                    holding_bars=(
                        horizon_bars
                    ),
                )
            )

            cohort_metrics = (
                summarize_portfolio_cohorts(
                    periods=periods,
                    cost_levels_bps=(
                        cost_levels
                    ),
                    annualization_periods=(
                        annualization_periods
                    ),
                )
            )

            cohort_metrics.insert(
                0,
                "horizon_minutes",
                horizon_minutes,
            )

            cohort_metrics.insert(
                0,
                "horizon_bars",
                horizon_bars,
            )

            cohort_metrics.insert(
                0,
                "model",
                model_name,
            )

            portfolio_frames.append(
                cohort_metrics
            )

        row_counts = {
            value[1]
            for value in horizon_model_counts
        }

        timestamp_counts = {
            value[2]
            for value in horizon_model_counts
        }

        if len(row_counts) != 1:
            raise ValueError(
                "Model row counts differ at "
                f"horizon {horizon_bars}"
            )

        if len(timestamp_counts) != 1:
            raise ValueError(
                "Model timestamp counts differ "
                f"at horizon {horizon_bars}"
            )

        row_counts_by_horizon[
            str(horizon_bars)
        ] = row_counts.pop()

        timestamp_counts_by_horizon[
            str(horizon_bars)
        ] = timestamp_counts.pop()

    fold_metrics = (
        pd.DataFrame(
            fold_metric_rows
        )
        .sort_values(
            [
                "horizon_bars",
                "model",
                "fold",
            ]
        )
        .reset_index(drop=True)
    )

    model_metrics = (
        pd.DataFrame(
            model_metric_rows
        )
        .sort_values(
            [
                "horizon_bars",
                "mean_rank_ic",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    portfolio_cohorts = (
        pd.concat(
            portfolio_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "horizon_bars",
                "model",
                "cost_bps",
                "cohort",
            ]
        )
        .reset_index(drop=True)
    )

    portfolio_summary = (
        build_horizon_portfolio_summary(
            portfolio_cohorts
        )
    )

    break_even = (
        build_horizon_break_even(
            portfolio_cohorts
        )
    )

    print()
    print(
        "Verifying 60-minute predictive metrics",
        flush=True,
    )

    original_model_metrics = (
        model_metrics.loc[
            model_metrics[
                "horizon_bars"
            ]
            == original_horizon_bars
        ]
    )

    metric_comparison = source_oos.merge(
        original_model_metrics,
        on="model",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_source",
            "_reproduced",
        ),
    )

    if len(metric_comparison) != len(
        source_oos
    ):
        raise ValueError(
            "The 60-minute model comparison "
            "does not cover all source models"
        )

    metric_columns = [
        "mean_rank_ic",
        "median_rank_ic",
        "rank_ic_std",
        "rank_ic_ir",
        "positive_timestamp_fraction",
        "rmse",
        "mae",
    ]

    for column in metric_columns:
        if not np.allclose(
            metric_comparison[
                f"{column}_source"
            ].to_numpy(dtype=np.float64),
            metric_comparison[
                f"{column}_reproduced"
            ].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "The 60-minute predictive "
                f"metric {column} differs"
            )

    integer_columns = [
        "n_rows",
        "n_timestamps",
    ]

    for column in integer_columns:
        if not np.array_equal(
            metric_comparison[
                f"{column}_source"
            ].to_numpy(),
            metric_comparison[
                f"{column}_reproduced"
            ].to_numpy(),
        ):
            raise ValueError(
                "The 60-minute predictive "
                f"count {column} differs"
            )

    print(
        "Verifying 60-minute portfolio metrics",
        flush=True,
    )

    source_costs = set(
        source_cohorts[
            "cost_bps"
        ].astype(float)
    )

    reproduced_cohorts = (
        portfolio_cohorts.loc[
            (
                portfolio_cohorts[
                    "horizon_bars"
                ]
                == original_horizon_bars
            )
            & portfolio_cohorts[
                "cost_bps"
            ].isin(source_costs)
        ]
    )

    portfolio_comparison = (
        source_cohorts.merge(
            reproduced_cohorts,
            on=[
                "model",
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
    )

    if len(portfolio_comparison) != len(
        source_cohorts
    ):
        raise ValueError(
            "The 60-minute portfolio "
            "reproduction is incomplete"
        )

    portfolio_columns = [
        "mean_net_return",
        "annualized_net_return",
        "annualized_volatility",
        "annualized_sharpe",
        "positive_fraction",
        "mean_turnover",
        "cumulative_net_log_return",
    ]

    for column in portfolio_columns:
        if not np.allclose(
            portfolio_comparison[
                f"{column}_source"
            ].to_numpy(dtype=np.float64),
            portfolio_comparison[
                f"{column}_reproduced"
            ].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        ):
            maximum_error = float(
                np.abs(
                    portfolio_comparison[
                        f"{column}_source"
                    ].to_numpy(
                        dtype=np.float64
                    )
                    - portfolio_comparison[
                        f"{column}_reproduced"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ).max()
            )

            raise ValueError(
                "The 60-minute portfolio "
                f"metric {column} differs: "
                f"{maximum_error}"
            )

    evidence_directory = Path(
        sensitivity_config[
            "evidence_dir"
        ]
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics_path = (
        evidence_directory
        / "horizon_fold_metrics.csv"
    )

    model_metrics_path = (
        evidence_directory
        / "horizon_model_metrics.csv"
    )

    portfolio_cohorts_path = (
        evidence_directory
        / "horizon_portfolio_cohort_metrics.csv"
    )

    portfolio_summary_path = (
        evidence_directory
        / "horizon_portfolio_summary.csv"
    )

    break_even_path = (
        evidence_directory
        / "horizon_break_even.csv"
    )

    write_csv(
        fold_metrics,
        fold_metrics_path,
    )

    write_csv(
        model_metrics,
        model_metrics_path,
    )

    write_csv(
        portfolio_cohorts,
        portfolio_cohorts_path,
    )

    write_csv(
        portfolio_summary,
        portfolio_summary_path,
    )

    write_csv(
        break_even,
        break_even_path,
    )

    manifest_path = (
        evidence_directory
        / "horizon_sensitivity_manifest.json"
    )

    evidence_files = [
        fold_metrics_path,
        model_metrics_path,
        portfolio_cohorts_path,
        portfolio_summary_path,
        break_even_path,
    ]

    manifest = {
        "schema_version": 1,
        "sensitivity_config": str(
            sensitivity_config_path
        ),
        "sensitivity_config_sha256": (
            sha256_file(
                sensitivity_config_path
            )
        ),
        "data_config": str(
            data_config_path
        ),
        "data_config_sha256": (
            sha256_file(data_config_path)
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
        "models": model_names,
        "horizon_bars": list(
            horizons
        ),
        "horizon_minutes": [
            value * interval_minutes
            for value in horizons
        ],
        "cost_bps": cost_levels,
        "top_count": top_count,
        "entry_offset_bars": (
            entry_offset_bars
        ),
        "interval_minutes": (
            interval_minutes
        ),
        "expected_symbol_count": (
            expected_symbol_count
        ),
        "row_counts_by_horizon": (
            row_counts_by_horizon
        ),
        "timestamp_counts_by_horizon": (
            timestamp_counts_by_horizon
        ),
        "original_horizon_bars": (
            original_horizon_bars
        ),
        "original_model_metrics_reproduced": (
            True
        ),
        "original_portfolio_metrics_reproduced": (
            True
        ),
        "models_retrained": False,
        "interpretation": (
            "Frozen 60-minute model forecasts "
            "evaluated against alternate forward "
            "return and execution horizons"
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
        "Horizon sensitivity analysis "
        "complete"
    )

    print()
    print(
        "Predictive Rank IC by horizon"
    )

    print(
        model_metrics[
            [
                "model",
                "horizon_minutes",
                "mean_rank_ic",
                "rank_ic_ir",
                "positive_timestamp_fraction",
                "n_timestamps",
            ]
        ]
        .sort_values(
            [
                "horizon_minutes",
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
        "Break-even cost by horizon"
    )

    print(
        break_even[
            [
                "model",
                "horizon_minutes",
                "pooled_break_even_cost_bps",
                "worst_individual_break_even_cost_bps",
                "mean_zero_cost_sharpe",
                "mean_turnover",
                "positive_cohort_fraction",
            ]
        ]
        .sort_values(
            [
                "horizon_minutes",
                "pooled_break_even_cost_bps",
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
        "One-basis-point horizon summary"
    )

    print(
        portfolio_summary.loc[
            np.isclose(
                portfolio_summary[
                    "cost_bps"
                ].to_numpy(
                    dtype=np.float64
                ),
                1.0,
                rtol=0.0,
                atol=1e-12,
            ),
            [
                "model",
                "horizon_minutes",
                "cohort_count",
                "mean_cohort_sharpe",
                "mean_annualized_net_return",
                "worst_annualized_net_return",
                "mean_turnover",
            ],
        ]
        .sort_values(
            [
                "horizon_minutes",
                "mean_cohort_sharpe",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
