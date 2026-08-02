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
    validate_prediction_frame,
)
from deep_alpha.evaluation.portfolio import (
    build_portfolio_periods,
    summarize_portfolio_cohorts,
    summarize_portfolio_models,
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
            f"Building execution returns for {symbol}",
            flush=True,
        )

        path = (
            processed_directory
            / "bars"
            / f"symbol={symbol}"
            / "part-00000.parquet"
        )

        bars = pd.read_parquet(
            path,
            columns=[
                "timestamp",
                "open",
                "close",
            ],
        ).sort_values("timestamp")

        entry_open = bars["open"].shift(-1)

        exit_close = bars["close"].shift(
            -holding_bars
        )

        execution_return = np.log(
            exit_close / entry_open
        )

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": bars[
                        "timestamp"
                    ],
                    "symbol": symbol,
                    "execution_return": (
                        execution_return
                    ),
                }
            )
        )

    result = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            ["timestamp", "symbol"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return result


def load_model_predictions(
    model_name: str,
    directory: Path,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    frames = []
    source_records = []

    for fold in [1, 2, 3]:
        path = (
            directory
            / (
                f"{model_name}_fold_{fold}_"
                "test.parquet"
            )
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_parquet(path)
        frame["fold"] = fold

        frames.append(frame)

        source_records.append(
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
            ["timestamp", "symbol"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    validate_prediction_frame(combined)

    return combined, source_records


def assert_prediction_alignment(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    model_name: str,
) -> None:
    if len(reference) != len(candidate):
        raise ValueError(
            f"Prediction row mismatch for {model_name}"
        )

    if not np.array_equal(
        reference["timestamp"].array.asi8,
        candidate["timestamp"].array.asi8,
    ):
        raise ValueError(
            f"Prediction timestamps differ for {model_name}"
        )

    if not np.array_equal(
        reference["symbol"].to_numpy(),
        candidate["symbol"].to_numpy(),
    ):
        raise ValueError(
            f"Prediction symbols differ for {model_name}"
        )

    for column in [
        "raw_target",
        "target",
    ]:
        if not np.allclose(
            reference[column].to_numpy(
                dtype=np.float64
            ),
            candidate[column].to_numpy(
                dtype=np.float64
            ),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{column} differs for {model_name}"
            )


def build_model_comparison(
    prediction_metrics: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
) -> pd.DataFrame:
    base = prediction_metrics[
        [
            "model",
            "mean_rank_ic",
            "median_rank_ic",
            "rank_ic_ir",
            "positive_timestamp_fraction",
            "rmse",
            "mae",
        ]
    ].copy()

    selected = portfolio_summary.loc[
        portfolio_summary["cost_bps"].isin(
            [0.0, 5.0]
        ),
        [
            "model",
            "cost_bps",
            "mean_cohort_sharpe",
            "worst_cohort_sharpe",
            "mean_annualized_net_return",
            "worst_annualized_net_return",
            "mean_turnover",
        ],
    ].copy()

    wide_parts = []

    for cost_bps in [0.0, 5.0]:
        subset = selected.loc[
            selected["cost_bps"] == cost_bps
        ].drop(columns="cost_bps")

        suffix = (
            f"_{int(cost_bps)}bps"
        )

        subset = subset.rename(
            columns={
                column: column + suffix
                for column in subset.columns
                if column != "model"
            }
        )

        wide_parts.append(subset)

    comparison = base

    for part in wide_parts:
        comparison = comparison.merge(
            part,
            on="model",
            how="left",
            validate="one_to_one",
        )

    return comparison.sort_values(
        "mean_rank_ic",
        ascending=False,
    ).reset_index(drop=True)


def main() -> None:
    data_config_path = Path(
        "configs/data.yaml"
    )

    evaluation_config_path = Path(
        "configs/evaluation.yaml"
    )

    data_config = load_yaml(
        data_config_path
    )["data"]

    evaluation_config = load_yaml(
        evaluation_config_path
    )["evaluation"]

    symbols = [
        str(symbol)
        for symbol in data_config["symbols"]
    ]

    holding_bars = int(
        evaluation_config["holding_bars"]
    )

    interval_minutes = int(
        evaluation_config[
            "interval_minutes"
        ]
    )

    top_count = int(
        evaluation_config["top_count"]
    )

    annualization_periods = int(
        evaluation_config[
            "annualization_periods"
        ]
    )

    cost_levels = [
        float(value)
        for value in evaluation_config[
            "cost_bps"
        ]
    ]

    processed_directory = Path(
        data_config["processed_dir"]
    )

    evidence_directory = Path(
        evaluation_config["evidence_dir"]
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    execution_returns = load_execution_returns(
        symbols=symbols,
        processed_directory=processed_directory,
        holding_bars=holding_bars,
    )

    prediction_metric_rows = []
    cohort_frames = []
    prediction_source_records = []

    reference_predictions: (
        pd.DataFrame | None
    ) = None

    expected_rows = 1_041_840
    expected_timestamps = 52_092

    for model_name, model_configuration in (
        evaluation_config["models"].items()
    ):
        print()
        print(
            f"=== Evaluating {model_name} ===",
            flush=True,
        )

        predictions, source_records = (
            load_model_predictions(
                model_name=str(model_name),
                directory=Path(
                    model_configuration[
                        "directory"
                    ]
                ),
            )
        )

        prediction_source_records.extend(
            source_records
        )

        if len(predictions) != expected_rows:
            raise ValueError(
                f"Unexpected row count for {model_name}: "
                f"{len(predictions)}"
            )

        if (
            predictions["timestamp"].nunique()
            != expected_timestamps
        ):
            raise ValueError(
                f"Unexpected timestamp count "
                f"for {model_name}"
            )

        if reference_predictions is None:
            reference_predictions = predictions[
                [
                    "timestamp",
                    "symbol",
                    "raw_target",
                    "target",
                ]
            ].copy()
        else:
            assert_prediction_alignment(
                reference=reference_predictions,
                candidate=predictions,
                model_name=str(model_name),
            )

        metrics = summarize_predictions(
            predictions
        )

        metrics["model"] = str(model_name)

        prediction_metric_rows.append(
            metrics
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
            interval_minutes=interval_minutes,
            holding_bars=holding_bars,
        )

        cohort_metrics = (
            summarize_portfolio_cohorts(
                periods=periods,
                cost_levels_bps=cost_levels,
                annualization_periods=(
                    annualization_periods
                ),
            )
        )

        cohort_metrics.insert(
            0,
            "model",
            str(model_name),
        )

        cohort_frames.append(cohort_metrics)

    prediction_metrics = (
        pd.DataFrame(
            prediction_metric_rows
        )
        [
            [
                "model",
                "mean_rank_ic",
                "median_rank_ic",
                "rank_ic_std",
                "rank_ic_ir",
                "positive_timestamp_fraction",
                "rmse",
                "mae",
                "n_rows",
                "n_timestamps",
            ]
        ]
        .sort_values(
            "mean_rank_ic",
            ascending=False,
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
                "cost_bps",
                "cohort",
            ]
        )
        .reset_index(drop=True)
    )

    portfolio_summary = (
        summarize_portfolio_models(
            cohort_metrics
        )
    )

    comparison = build_model_comparison(
        prediction_metrics=prediction_metrics,
        portfolio_summary=portfolio_summary,
    )

    prediction_metrics_path = (
        evidence_directory
        / "oos_model_metrics.csv"
    )

    cohort_metrics_path = (
        evidence_directory
        / "portfolio_cohort_metrics.csv"
    )

    portfolio_summary_path = (
        evidence_directory
        / "portfolio_summary.csv"
    )

    comparison_path = (
        evidence_directory
        / "portfolio_model_comparison.csv"
    )

    write_csv(
        prediction_metrics,
        prediction_metrics_path,
    )

    write_csv(
        cohort_metrics,
        cohort_metrics_path,
    )

    write_csv(
        portfolio_summary,
        portfolio_summary_path,
    )

    write_csv(
        comparison,
        comparison_path,
    )

    normalized_manifest_path = Path(
        data_config[
            "normalized_manifest_path"
        ]
    )

    manifest_path = (
        evidence_directory
        / "portfolio_run_manifest.json"
    )

    evidence_files = [
        prediction_metrics_path,
        cohort_metrics_path,
        portfolio_summary_path,
        comparison_path,
    ]

    manifest = {
        "schema_version": 1,
        "data_config": str(
            data_config_path
        ),
        "data_config_sha256": sha256_file(
            data_config_path
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
        "models": list(
            evaluation_config["models"]
        ),
        "prediction_rows_per_model": (
            expected_rows
        ),
        "prediction_timestamps_per_model": (
            expected_timestamps
        ),
        "symbol_count": len(symbols),
        "holding_bars": holding_bars,
        "interval_minutes": interval_minutes,
        "holding_minutes": (
            holding_bars
            * interval_minutes
        ),
        "top_count": top_count,
        "cohort_count": holding_bars,
        "cost_bps": cost_levels,
        "annualization_periods": (
            annualization_periods
        ),
        "execution_convention": (
            "next_bar_open_to_horizon_close"
        ),
        "prediction_files": sorted(
            prediction_source_records,
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
    print("Economic evaluation complete")

    print()
    print("Out-of-sample prediction metrics")
    print(
        prediction_metrics[
            [
                "model",
                "mean_rank_ic",
                "rank_ic_ir",
            ]
        ].to_string(index=False)
    )

    print()
    print("Five-basis-point portfolio summary")

    print(
        portfolio_summary.loc[
            portfolio_summary[
                "cost_bps"
            ]
            == 5.0,
            [
                "model",
                "mean_cohort_sharpe",
                "worst_cohort_sharpe",
                "mean_annualized_net_return",
                "mean_turnover",
            ],
        ]
        .sort_values(
            "mean_cohort_sharpe",
            ascending=False,
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
