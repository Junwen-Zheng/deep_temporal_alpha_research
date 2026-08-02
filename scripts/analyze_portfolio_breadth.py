from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.evaluation.breadth import (
    build_breadth_break_even,
    build_breadth_periods,
    build_breadth_summary,
    build_prediction_matrices,
)
from deep_alpha.evaluation.portfolio import (
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

        execution_return = np.log(
            bars["close"].shift(
                -holding_bars
            )
            / bars["open"].shift(-1)
        )

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": (
                        bars["timestamp"]
                    ),
                    "symbol": symbol,
                    "execution_return": (
                        execution_return
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

    for fold in [1, 2, 3]:
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
                "prediction",
                "raw_target",
                "target",
            ],
        )

        frames.append(frame)

        records.append(
            {
                "model": model_name,
                "fold": fold,
                "path": str(path),
                "rows": len(frame),
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

    return combined, records


def main() -> None:
    sensitivity_config_path = Path(
        "configs/breadth_sensitivity.yaml"
    )

    sensitivity_config = load_yaml(
        sensitivity_config_path
    )["breadth_sensitivity"]

    data_config_path = Path(
        sensitivity_config[
            "data_config"
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

    evaluation_config = load_yaml(
        evaluation_config_path
    )["evaluation"]

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

    source_cohorts = pd.read_csv(
        source_cohort_path
    )

    source_manifest = json.loads(
        source_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    symbols = [
        str(symbol)
        for symbol in data_config[
            "symbols"
        ]
    ]

    expected_symbol_count = int(
        sensitivity_config[
            "expected_symbol_count"
        ]
    )

    expected_prediction_rows = int(
        sensitivity_config[
            "expected_prediction_rows"
        ]
    )

    expected_prediction_timestamps = int(
        sensitivity_config[
            "expected_prediction_timestamps"
        ]
    )

    if len(symbols) != expected_symbol_count:
        raise ValueError(
            "Configured symbol count differs "
            "from the data universe"
        )

    top_counts = [
        int(value)
        for value in sensitivity_config[
            "top_counts"
        ]
    ]

    if len(set(top_counts)) != len(
        top_counts
    ):
        raise ValueError(
            "Duplicate top-count values"
        )

    for top_count in top_counts:
        if top_count <= 0:
            raise ValueError(
                "top_count must be positive"
            )

        if (
            top_count * 2
            > expected_symbol_count
        ):
            raise ValueError(
                "top_count exceeds the universe"
            )

    cost_levels = [
        float(value)
        for value in sensitivity_config[
            "cost_bps"
        ]
    ]

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

    annualization_periods = int(
        evaluation_config[
            "annualization_periods"
        ]
    )

    configured_models = [
        str(model)
        for model in evaluation_config[
            "models"
        ]
    ]

    manifest_expectations = {
        "symbol_count": expected_symbol_count,
        "holding_bars": holding_bars,
        "interval_minutes": interval_minutes,
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
        actual_value = source_manifest.get(key)

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
    ) != set(configured_models):
        raise ValueError(
            "Source portfolio manifest models "
            "differ from the evaluation config"
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

    cohort_frames = []
    prediction_records = []

    for model_name, configuration in (
        evaluation_config["models"].items()
    ):
        model_name = str(model_name)

        print()
        print(
            f"=== Breadth analysis: "
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
                f"Unexpected row count for "
                f"{model_name}"
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

        merged = predictions.merge(
            execution_returns,
            on=[
                "timestamp",
                "symbol",
            ],
            how="left",
            validate="one_to_one",
        )

        if (
            merged[
                "execution_return"
            ].isna().any()
        ):
            raise ValueError(
                f"Missing execution returns "
                f"for {model_name}"
            )

        matrices = build_prediction_matrices(
            frame=merged,
            expected_symbol_count=(
                expected_symbol_count
            ),
        )

        if (
            matrices.timestamp_count
            != expected_prediction_timestamps
        ):
            raise ValueError(
                "Matrix timestamp count differs"
            )

        for top_count in top_counts:
            print(
                f"  top/bottom {top_count}",
                flush=True,
            )

            periods = build_breadth_periods(
                matrices=matrices,
                top_count=top_count,
                interval_minutes=(
                    interval_minutes
                ),
                holding_bars=holding_bars,
            )

            metrics = (
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

            metrics.insert(
                0,
                "top_count",
                top_count,
            )

            metrics.insert(
                0,
                "model",
                model_name,
            )

            cohort_frames.append(metrics)

    cohort_metrics = (
        pd.concat(
            cohort_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "top_count",
                "cost_bps",
                "cohort",
            ]
        )
        .reset_index(drop=True)
    )

    summary = build_breadth_summary(
        cohort_metrics
    )

    break_even = build_breadth_break_even(
        cohort_metrics
    )

    print()
    print(
        "Verifying top/bottom four against "
        "the committed portfolio evidence",
        flush=True,
    )

    source_costs = set(
        source_cohorts[
            "cost_bps"
        ].astype(float)
    )

    reproduced = cohort_metrics.loc[
        (
            cohort_metrics[
                "top_count"
            ]
            == 4
        )
        & cohort_metrics[
            "cost_bps"
        ].isin(source_costs)
    ].copy()

    comparison_columns = [
        "model",
        "cost_bps",
        "cohort",
    ]

    compared = source_cohorts.merge(
        reproduced,
        on=comparison_columns,
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_source",
            "_reproduced",
        ),
    )

    if len(compared) != len(
        source_cohorts
    ):
        raise ValueError(
            "Top-four reproduction does not "
            "cover all source rows"
        )

    exact_columns = [
        "period_count",
    ]

    for column in exact_columns:
        if not np.array_equal(
            compared[
                f"{column}_source"
            ].to_numpy(),
            compared[
                f"{column}_reproduced"
            ].to_numpy(),
        ):
            raise ValueError(
                f"Top-four {column} differs "
                "from source evidence"
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
            compared[
                f"{column}_source"
            ].to_numpy(dtype=np.float64),
            compared[
                f"{column}_reproduced"
            ].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        ):
            maximum_error = float(
                np.abs(
                    compared[
                        f"{column}_source"
                    ].to_numpy(
                        dtype=np.float64
                    )
                    - compared[
                        f"{column}_reproduced"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ).max()
            )

            raise ValueError(
                f"Top-four {column} "
                f"reproduction failed: "
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

    cohort_path = (
        evidence_directory
        / "breadth_cohort_metrics.csv"
    )

    summary_path = (
        evidence_directory
        / "breadth_summary.csv"
    )

    break_even_path = (
        evidence_directory
        / "breadth_break_even.csv"
    )

    write_csv(
        cohort_metrics,
        cohort_path,
    )

    write_csv(
        summary,
        summary_path,
    )

    write_csv(
        break_even,
        break_even_path,
    )

    manifest_path = (
        evidence_directory
        / "breadth_sensitivity_manifest.json"
    )

    evidence_files = [
        cohort_path,
        summary_path,
        break_even_path,
    ]

    normalized_manifest_path = Path(
        data_config[
            "normalized_manifest_path"
        ]
    )

    expected_models = [
        str(model)
        for model in evaluation_config[
            "models"
        ]
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
        "evaluation_config": str(
            evaluation_config_path
        ),
        "evaluation_config_sha256": (
            sha256_file(
                evaluation_config_path
            )
        ),
        "normalized_manifest": str(
            normalized_manifest_path
        ),
        "normalized_manifest_sha256": (
            sha256_file(
                normalized_manifest_path
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
        "models": expected_models,
        "top_counts": top_counts,
        "selected_symbol_counts": [
            value * 2
            for value in top_counts
        ],
        "cost_bps": cost_levels,
        "expected_symbol_count": (
            expected_symbol_count
        ),
        "holding_bars": holding_bars,
        "holding_minutes": (
            holding_bars
            * interval_minutes
        ),
        "cohort_count": holding_bars,
        "annualization_periods": (
            annualization_periods
        ),
        "prediction_rows_per_model": (
            expected_prediction_rows
        ),
        "prediction_timestamps_per_model": (
            expected_prediction_timestamps
        ),
        "top_count_four_reproduced": True,
        "prediction_files": sorted(
            prediction_records,
            key=lambda record: (
                record["model"],
                record["fold"],
            ),
        ),
        "method": (
            "equal-weight dollar-neutral "
            "top-and-bottom rank portfolios"
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
        "Portfolio-breadth sensitivity "
        "complete"
    )

    print()
    print(
        "Break-even cost by portfolio breadth"
    )

    print(
        break_even[
            [
                "model",
                "top_count",
                "selected_symbol_count",
                "pooled_break_even_cost_bps",
                "worst_individual_break_even_cost_bps",
                "mean_zero_cost_sharpe",
                "mean_turnover",
                "positive_cohort_fraction",
            ]
        ]
        .sort_values(
            [
                "top_count",
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
        "One-basis-point breadth summary"
    )

    print(
        summary.loc[
            np.isclose(
                summary[
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
                "top_count",
                "selected_symbol_count",
                "mean_cohort_sharpe",
                "mean_annualized_net_return",
                "worst_annualized_net_return",
                "mean_turnover",
            ],
        ]
        .sort_values(
            [
                "top_count",
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
