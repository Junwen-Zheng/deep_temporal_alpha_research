from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.evaluation.bootstrap import (
    build_portfolio_bootstrap_distribution,
    build_timestamp_rank_ic_series,
    circular_block_bootstrap_nanmeans,
    summarize_bootstrap_distribution,
    validate_block_lengths,
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
) -> float:
    actual = summarize_predictions(
        predictions
    )

    source = source_metrics.loc[
        source_metrics["model"]
        == model_name
    ]

    if len(source) != 1:
        raise ValueError(
            f"Expected one source row "
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
                f"Predictive metric "
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
                f"Predictive count "
                f"{column} differs for "
                f"{model_name}"
            )

    return float(
        source_row["mean_rank_ic"]
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
                f"Portfolio metric "
                f"{column} differs for "
                f"{model_name}"
            )


def main() -> None:
    config_path = Path(
        "configs/bootstrap.yaml"
    )

    config = load_yaml(
        config_path
    )["bootstrap"]

    data_config_path = Path(
        config["data_config"]
    )

    evaluation_config_path = Path(
        config["evaluation_config"]
    )

    data_config = load_yaml(
        data_config_path
    )["data"]

    evaluation_config = load_yaml(
        evaluation_config_path
    )["evaluation"]

    source_oos_path = Path(
        config[
            "source_oos_metrics"
        ]
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

    prediction_blocks = (
        validate_block_lengths(
            config[
                "prediction_block_lengths"
            ]
        )
    )

    portfolio_blocks = (
        validate_block_lengths(
            config[
                "portfolio_block_lengths"
            ]
        )
    )

    primary_prediction_block = int(
        config[
            "primary_prediction_block_length"
        ]
    )

    primary_portfolio_block = int(
        config[
            "primary_portfolio_block_length"
        ]
    )

    if (
        primary_prediction_block
        not in prediction_blocks
    ):
        raise ValueError(
            "Primary prediction block is "
            "not configured"
        )

    if (
        primary_portfolio_block
        not in portfolio_blocks
    ):
        raise ValueError(
            "Primary portfolio block is "
            "not configured"
        )

    resamples = int(
        config["resamples"]
    )

    confidence_level = float(
        config[
            "confidence_level"
        ]
    )

    random_seed = int(
        config["random_seed"]
    )

    focus_cost_bps = float(
        config["focus_cost_bps"]
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

    execution_returns = (
        load_execution_returns(
            symbols=symbols,
            processed_directory=(
                processed_directory
            ),
            holding_bars=holding_bars,
        )
    )

    prediction_summary_rows = []
    prediction_distribution_frames = []
    portfolio_summary_rows = []
    portfolio_distribution_frames = []
    prediction_records = []

    for model_index, (
        model_name,
        model_config,
    ) in enumerate(
        evaluation_config[
            "models"
        ].items()
    ):
        model_name = str(model_name)

        print()
        print(
            f"=== Bootstrap uncertainty: "
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
                f"Unexpected prediction "
                f"timestamps for {model_name}"
            )

        source_mean_rank_ic = (
            verify_prediction_metrics(
                model_name=model_name,
                predictions=predictions,
                source_metrics=source_oos,
            )
        )

        rank_ic_series = (
            build_timestamp_rank_ic_series(
                frame=predictions,
                expected_symbol_count=(
                    expected_symbol_count
                ),
            )
        )

        if (
            len(rank_ic_series)
            != expected_prediction_timestamps
        ):
            raise ValueError(
                f"Unexpected Rank IC timeline "
                f"length for {model_name}"
            )

        valid_rank_ic_count = int(
            rank_ic_series[
                "rank_ic"
            ].notna().sum()
        )

        undefined_rank_ic_count = (
            len(rank_ic_series)
            - valid_rank_ic_count
        )

        if valid_rank_ic_count <= 0:
            raise ValueError(
                f"No valid Rank IC observations "
                f"for {model_name}"
            )

        print(
            f"  valid Rank IC timestamps: "
            f"{valid_rank_ic_count}; "
            f"undefined: "
            f"{undefined_rank_ic_count}",
            flush=True,
        )

        observed_rank_ic = float(
            rank_ic_series[
                "rank_ic"
            ].mean()
        )

        if not np.isclose(
            observed_rank_ic,
            source_mean_rank_ic,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                f"Timestamp Rank IC mean "
                f"does not reproduce source "
                f"for {model_name}"
            )

        for block_length in (
            prediction_blocks
        ):
            print(
                f"  predictive block "
                f"{block_length} timestamps",
                flush=True,
            )

            seed = (
                random_seed
                + model_index
                * 100_000
                + block_length
            )

            samples = (
                circular_block_bootstrap_nanmeans(
                    values=rank_ic_series[
                        "rank_ic"
                    ].to_numpy(
                        dtype=np.float64
                    ),
                    resamples=resamples,
                    block_length=(
                        block_length
                    ),
                    seed=seed,
                )
            )

            statistics = (
                summarize_bootstrap_distribution(
                    samples=samples,
                    observed=observed_rank_ic,
                    confidence_level=(
                        confidence_level
                    ),
                )
            )

            prediction_summary_rows.append(
                {
                    "model": model_name,
                    "block_length_timestamps": (
                        block_length
                    ),
                    "block_minutes": (
                        block_length
                        * interval_minutes
                    ),
                    "resamples": resamples,
                    "confidence_level": (
                        confidence_level
                    ),
                    "timeline_timestamp_count": (
                        len(rank_ic_series)
                    ),
                    "valid_timestamp_count": (
                        valid_rank_ic_count
                    ),
                    "undefined_timestamp_count": (
                        undefined_rank_ic_count
                    ),
                    **statistics,
                }
            )

            prediction_distribution_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "block_length_timestamps": (
                            block_length
                        ),
                        "block_minutes": (
                            block_length
                            * interval_minutes
                        ),
                        "resample": np.arange(
                            resamples,
                            dtype=np.int64,
                        ),
                        "mean_rank_ic": (
                            samples
                        ),
                    }
                )
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

        for block_length in (
            portfolio_blocks
        ):
            print(
                f"  portfolio block "
                f"{block_length} hours",
                flush=True,
            )

            seed = (
                random_seed
                + model_index
                * 100_000
                + 50_000
                + block_length
            )

            (
                distribution,
                observed,
            ) = (
                build_portfolio_bootstrap_distribution(
                    periods=periods,
                    resamples=resamples,
                    block_length_periods=(
                        block_length
                    ),
                    seed=seed,
                    cost_bps=(
                        focus_cost_bps
                    ),
                    annualization_periods=(
                        annualization_periods
                    ),
                )
            )

            gross_statistics = (
                summarize_bootstrap_distribution(
                    samples=distribution[
                        "gross_annualized_return"
                    ],
                    observed=observed[
                        "gross_annualized_return"
                    ],
                    confidence_level=(
                        confidence_level
                    ),
                )
            )

            net_statistics = (
                summarize_bootstrap_distribution(
                    samples=distribution[
                        "net_annualized_return"
                    ],
                    observed=observed[
                        "net_annualized_return"
                    ],
                    confidence_level=(
                        confidence_level
                    ),
                )
            )

            break_even_statistics = (
                summarize_bootstrap_distribution(
                    samples=distribution[
                        "break_even_cost_bps"
                    ],
                    observed=observed[
                        "break_even_cost_bps"
                    ],
                    confidence_level=(
                        confidence_level
                    ),
                )
            )

            turnover_statistics = (
                summarize_bootstrap_distribution(
                    samples=distribution[
                        "mean_turnover"
                    ],
                    observed=observed[
                        "mean_turnover"
                    ],
                    confidence_level=(
                        confidence_level
                    ),
                )
            )

            portfolio_summary_rows.append(
                {
                    "model": model_name,
                    "block_length_periods": (
                        block_length
                    ),
                    "block_hours": (
                        block_length
                    ),
                    "resamples": resamples,
                    "confidence_level": (
                        confidence_level
                    ),
                    "focus_cost_bps": (
                        focus_cost_bps
                    ),
                    "observed_gross_annualized_return": (
                        gross_statistics[
                            "observed"
                        ]
                    ),
                    "gross_bootstrap_mean": (
                        gross_statistics[
                            "bootstrap_mean"
                        ]
                    ),
                    "gross_bootstrap_std": (
                        gross_statistics[
                            "bootstrap_std"
                        ]
                    ),
                    "gross_confidence_lower": (
                        gross_statistics[
                            "confidence_lower"
                        ]
                    ),
                    "gross_confidence_upper": (
                        gross_statistics[
                            "confidence_upper"
                        ]
                    ),
                    "gross_probability_positive": (
                        gross_statistics[
                            "probability_positive"
                        ]
                    ),
                    "observed_net_annualized_return": (
                        net_statistics[
                            "observed"
                        ]
                    ),
                    "net_bootstrap_mean": (
                        net_statistics[
                            "bootstrap_mean"
                        ]
                    ),
                    "net_bootstrap_std": (
                        net_statistics[
                            "bootstrap_std"
                        ]
                    ),
                    "net_confidence_lower": (
                        net_statistics[
                            "confidence_lower"
                        ]
                    ),
                    "net_confidence_upper": (
                        net_statistics[
                            "confidence_upper"
                        ]
                    ),
                    "net_probability_positive": (
                        net_statistics[
                            "probability_positive"
                        ]
                    ),
                    "observed_break_even_cost_bps": (
                        break_even_statistics[
                            "observed"
                        ]
                    ),
                    "break_even_bootstrap_mean": (
                        break_even_statistics[
                            "bootstrap_mean"
                        ]
                    ),
                    "break_even_bootstrap_std": (
                        break_even_statistics[
                            "bootstrap_std"
                        ]
                    ),
                    "break_even_confidence_lower": (
                        break_even_statistics[
                            "confidence_lower"
                        ]
                    ),
                    "break_even_confidence_upper": (
                        break_even_statistics[
                            "confidence_upper"
                        ]
                    ),
                    "break_even_probability_positive": (
                        break_even_statistics[
                            "probability_positive"
                        ]
                    ),
                    "probability_break_even_above_focus_cost": (
                        float(
                            (
                                distribution[
                                    "break_even_cost_bps"
                                ]
                                > focus_cost_bps
                            ).mean()
                        )
                    ),
                    "observed_mean_turnover": (
                        turnover_statistics[
                            "observed"
                        ]
                    ),
                    "turnover_bootstrap_mean": (
                        turnover_statistics[
                            "bootstrap_mean"
                        ]
                    ),
                    "turnover_confidence_lower": (
                        turnover_statistics[
                            "confidence_lower"
                        ]
                    ),
                    "turnover_confidence_upper": (
                        turnover_statistics[
                            "confidence_upper"
                        ]
                    ),
                }
            )

            distribution.insert(
                0,
                "block_hours",
                block_length,
            )

            distribution.insert(
                0,
                "block_length_periods",
                block_length,
            )

            distribution.insert(
                0,
                "model",
                model_name,
            )

            portfolio_distribution_frames.append(
                distribution
            )

    prediction_summary = (
        pd.DataFrame(
            prediction_summary_rows
        )
        .sort_values(
            [
                "block_length_timestamps",
                "model",
            ]
        )
        .reset_index(drop=True)
    )

    prediction_distribution = (
        pd.concat(
            prediction_distribution_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "block_length_timestamps",
                "resample",
            ]
        )
        .reset_index(drop=True)
    )

    portfolio_summary = (
        pd.DataFrame(
            portfolio_summary_rows
        )
        .sort_values(
            [
                "block_length_periods",
                "model",
            ]
        )
        .reset_index(drop=True)
    )

    portfolio_distribution = (
        pd.concat(
            portfolio_distribution_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "block_length_periods",
                "resample",
            ]
        )
        .reset_index(drop=True)
    )

    evidence_directory = Path(
        config["evidence_dir"]
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_summary_path = (
        evidence_directory
        / "bootstrap_prediction_summary.csv"
    )

    prediction_distribution_path = (
        evidence_directory
        / "bootstrap_prediction_distribution.csv"
    )

    portfolio_summary_path = (
        evidence_directory
        / "bootstrap_portfolio_summary.csv"
    )

    portfolio_distribution_path = (
        evidence_directory
        / "bootstrap_portfolio_distribution.csv"
    )

    write_csv(
        prediction_summary,
        prediction_summary_path,
    )

    write_csv(
        prediction_distribution,
        prediction_distribution_path,
    )

    write_csv(
        portfolio_summary,
        portfolio_summary_path,
    )

    write_csv(
        portfolio_distribution,
        portfolio_distribution_path,
    )

    manifest_path = (
        evidence_directory
        / "bootstrap_manifest.json"
    )

    evidence_files = [
        prediction_summary_path,
        prediction_distribution_path,
        portfolio_summary_path,
        portfolio_distribution_path,
    ]

    manifest = {
        "schema_version": 1,
        "bootstrap_config": str(
            config_path
        ),
        "bootstrap_config_sha256": (
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
        "prediction_block_lengths": list(
            prediction_blocks
        ),
        "prediction_block_minutes": [
            value * interval_minutes
            for value in prediction_blocks
        ],
        "portfolio_block_lengths": list(
            portfolio_blocks
        ),
        "portfolio_block_hours": list(
            portfolio_blocks
        ),
        "primary_prediction_block_length": (
            primary_prediction_block
        ),
        "primary_portfolio_block_length": (
            primary_portfolio_block
        ),
        "resamples": resamples,
        "confidence_level": (
            confidence_level
        ),
        "random_seed": random_seed,
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
        "prediction_bootstrap_method": (
            "circular moving-block bootstrap "
            "of timestamp-level cross-sectional "
            "Rank IC"
        ),
        "portfolio_bootstrap_method": (
            "independent circular moving-block "
            "bootstrap within each non-overlapping "
            "hourly cohort, followed by equal "
            "weighting across cohorts"
        ),
        "confidence_interval_method": (
            "percentile"
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
        "Bootstrap uncertainty analysis "
        "complete"
    )

    print()
    print(
        "Primary Rank IC uncertainty"
    )

    print(
        prediction_summary.loc[
            prediction_summary[
                "block_length_timestamps"
            ]
            == primary_prediction_block,
            [
                "model",
                "observed",
                "confidence_lower",
                "confidence_upper",
                "probability_positive",
                "bootstrap_std",
            ],
        ]
        .sort_values(
            "observed",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()
    print(
        "Primary portfolio uncertainty"
    )

    print(
        portfolio_summary.loc[
            portfolio_summary[
                "block_length_periods"
            ]
            == primary_portfolio_block,
            [
                "model",
                "observed_break_even_cost_bps",
                "break_even_confidence_lower",
                "break_even_confidence_upper",
                "probability_break_even_above_focus_cost",
                "observed_net_annualized_return",
                "net_confidence_lower",
                "net_confidence_upper",
                "net_probability_positive",
            ],
        ]
        .sort_values(
            "observed_break_even_cost_bps",
            ascending=False,
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
