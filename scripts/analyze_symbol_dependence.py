from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.evaluation.breadth import (
    build_breadth_periods,
    build_prediction_matrices,
)
from deep_alpha.evaluation.metrics import (
    summarize_predictions,
)
from deep_alpha.evaluation.portfolio import (
    summarize_portfolio_cohorts,
)
from deep_alpha.evaluation.symbols import (
    build_symbol_contribution_metrics,
    build_symbol_dependence_model_summary,
    exclude_symbol_from_matrices,
    summarize_symbol_exclusion_portfolios,
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


def verify_full_prediction_metrics(
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
                f"Full-universe {column} "
                f"differs for {model_name}"
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
                f"Full-universe {column} "
                f"differs for {model_name}"
            )

    return float(
        source_row["mean_rank_ic"]
    )


def verify_full_portfolio_metrics(
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
            "Full-universe portfolio "
            f"reproduction failed for "
            f"{model_name}"
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
            f"Period counts differ for "
            f"{model_name}"
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
    config_path = Path(
        "configs/symbol_dependence.yaml"
    )

    config = load_yaml(
        config_path
    )["symbol_dependence"]

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

    source_break_even_path = Path(
        config[
            "source_break_even_summary"
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
            "Focus cost must be present "
            "in cost levels"
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

    if set(
        source_break_even["model"]
    ) != set(models):
        raise ValueError(
            "Source break-even model coverage "
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

    exclusion_prediction_rows = []
    exclusion_cohort_frames = []
    contribution_frames = []
    prediction_records = []

    expected_exclusion_rows = (
        expected_prediction_rows
        * (
            expected_symbol_count - 1
        )
        // expected_symbol_count
    )

    for model_name, model_config in (
        evaluation_config["models"].items()
    ):
        model_name = str(model_name)

        print()
        print(
            f"=== Symbol dependence: "
            f"{model_name} ===",
            flush=True,
        )

        (
            predictions,
            records,
        ) = load_model_predictions(
            model_name=model_name,
            prediction_directory=Path(
                model_config["directory"]
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
                "Unexpected prediction "
                f"timestamps for {model_name}"
            )

        full_mean_rank_ic = (
            verify_full_prediction_metrics(
                model_name=model_name,
                predictions=predictions,
                source_metrics=source_oos,
            )
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
                "Missing execution returns "
                f"for {model_name}"
            )

        full_matrices = (
            build_prediction_matrices(
                frame=merged,
                expected_symbol_count=(
                    expected_symbol_count
                ),
            )
        )

        contribution_frames.append(
            build_symbol_contribution_metrics(
                model=model_name,
                matrices=full_matrices,
                top_count=top_count,
            )
        )

        full_periods = (
            build_breadth_periods(
                matrices=full_matrices,
                top_count=top_count,
                interval_minutes=(
                    interval_minutes
                ),
                holding_bars=holding_bars,
            )
        )

        verify_full_portfolio_metrics(
            model_name=model_name,
            periods=full_periods,
            source_metrics=source_cohorts,
            annualization_periods=(
                annualization_periods
            ),
        )

        for index, symbol in enumerate(
            symbols,
            start=1,
        ):
            print(
                f"  [{index:02d}/"
                f"{len(symbols):02d}] "
                f"excluding {symbol}",
                flush=True,
            )

            excluded_predictions = (
                predictions.loc[
                    predictions["symbol"]
                    != symbol
                ]
                .copy()
            )

            if (
                len(excluded_predictions)
                != expected_exclusion_rows
            ):
                raise ValueError(
                    "Unexpected exclusion row "
                    f"count for {model_name}/"
                    f"{symbol}"
                )

            exclusion_metrics = (
                summarize_predictions(
                    excluded_predictions
                )
            )

            if (
                int(
                    exclusion_metrics[
                        "n_timestamps"
                    ]
                )
                != expected_prediction_timestamps
            ):
                raise ValueError(
                    "Exclusion timestamp count "
                    f"differs for {model_name}/"
                    f"{symbol}"
                )

            exclusion_prediction_rows.append(
                {
                    "model": model_name,
                    "excluded_symbol": symbol,
                    "remaining_symbol_count": (
                        expected_symbol_count
                        - 1
                    ),
                    "full_mean_rank_ic": (
                        full_mean_rank_ic
                    ),
                    **exclusion_metrics,
                    "rank_ic_delta": (
                        float(
                            exclusion_metrics[
                                "mean_rank_ic"
                            ]
                        )
                        - full_mean_rank_ic
                    ),
                }
            )

            excluded_matrices = (
                exclude_symbol_from_matrices(
                    matrices=full_matrices,
                    symbol=symbol,
                )
            )

            if (
                excluded_matrices.symbol_count
                != expected_symbol_count - 1
            ):
                raise ValueError(
                    "Excluded matrix symbol "
                    "count differs"
                )

            periods = build_breadth_periods(
                matrices=excluded_matrices,
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
                "excluded_symbol",
                symbol,
            )

            cohort_metrics.insert(
                0,
                "model",
                model_name,
            )

            exclusion_cohort_frames.append(
                cohort_metrics
            )

    exclusion_prediction_metrics = (
        pd.DataFrame(
            exclusion_prediction_rows
        )
        .sort_values(
            [
                "model",
                "excluded_symbol",
            ]
        )
        .reset_index(drop=True)
    )

    exclusion_cohort_metrics = (
        pd.concat(
            exclusion_cohort_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "excluded_symbol",
                "cost_bps",
                "cohort",
            ]
        )
        .reset_index(drop=True)
    )

    contribution_metrics = (
        pd.concat(
            contribution_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "model",
                "symbol",
            ]
        )
        .reset_index(drop=True)
    )

    exclusion_portfolio_summary = (
        summarize_symbol_exclusion_portfolios(
            cohort_metrics=(
                exclusion_cohort_metrics
            ),
            focus_cost_bps=(
                focus_cost_bps
            ),
            annualization_periods=(
                annualization_periods
            ),
        )
    )

    exclusion_summary = (
        exclusion_prediction_metrics.merge(
            exclusion_portfolio_summary,
            on=[
                "model",
                "excluded_symbol",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    full_break_even = (
        source_break_even[
            [
                "model",
                "pooled_break_even_cost_bps",
            ]
        ]
        .rename(
            columns={
                "pooled_break_even_cost_bps": (
                    "full_break_even_cost_bps"
                )
            }
        )
    )

    exclusion_summary = (
        exclusion_summary.merge(
            full_break_even,
            on="model",
            how="left",
            validate="many_to_one",
        )
    )

    if exclusion_summary[
        "full_break_even_cost_bps"
    ].isna().any():
        raise ValueError(
            "Missing full break-even values"
        )

    exclusion_summary[
        "break_even_delta_bps"
    ] = (
        exclusion_summary[
            "pooled_break_even_cost_bps"
        ]
        - exclusion_summary[
            "full_break_even_cost_bps"
        ]
    )

    model_summary = (
        build_symbol_dependence_model_summary(
            exclusion_summary=(
                exclusion_summary
            ),
            contribution_metrics=(
                contribution_metrics
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

    prediction_path = (
        evidence_directory
        / "symbol_exclusion_prediction_metrics.csv"
    )

    cohort_path = (
        evidence_directory
        / "symbol_exclusion_portfolio_cohort_metrics.csv"
    )

    exclusion_summary_path = (
        evidence_directory
        / "symbol_exclusion_summary.csv"
    )

    contribution_path = (
        evidence_directory
        / "symbol_contribution_metrics.csv"
    )

    model_summary_path = (
        evidence_directory
        / "symbol_dependence_model_summary.csv"
    )

    write_csv(
        exclusion_prediction_metrics,
        prediction_path,
    )

    write_csv(
        exclusion_cohort_metrics,
        cohort_path,
    )

    write_csv(
        exclusion_summary,
        exclusion_summary_path,
    )

    write_csv(
        contribution_metrics,
        contribution_path,
    )

    write_csv(
        model_summary,
        model_summary_path,
    )

    manifest_path = (
        evidence_directory
        / "symbol_dependence_manifest.json"
    )

    evidence_files = [
        prediction_path,
        cohort_path,
        exclusion_summary_path,
        contribution_path,
        model_summary_path,
    ]

    manifest = {
        "schema_version": 1,
        "symbol_dependence_config": str(
            config_path
        ),
        "symbol_dependence_config_sha256": (
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
        "source_break_even_summary": str(
            source_break_even_path
        ),
        "source_break_even_summary_sha256": (
            sha256_file(
                source_break_even_path
            )
        ),
        "models": models,
        "symbols": symbols,
        "excluded_symbol_count": (
            expected_symbol_count
        ),
        "remaining_symbol_count": (
            expected_symbol_count - 1
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
        "prediction_rows_per_model": (
            expected_prediction_rows
        ),
        "prediction_rows_per_exclusion": (
            expected_exclusion_rows
        ),
        "prediction_timestamps_per_model": (
            expected_prediction_timestamps
        ),
        "full_prediction_metrics_reproduced": (
            True
        ),
        "full_portfolio_metrics_reproduced": (
            True
        ),
        "method": (
            "leave-one-symbol-out predictive "
            "and portfolio evaluation with "
            "full-universe symbol contribution "
            "decomposition"
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
        "Symbol-dependence analysis complete"
    )

    print()
    print("Model dependence summary")

    print(
        model_summary[
            [
                "model",
                "worst_exclusion_rank_ic",
                "maximum_absolute_rank_ic_delta",
                "most_influential_prediction_symbol",
                "worst_exclusion_break_even_cost_bps",
                "maximum_absolute_break_even_delta_bps",
                "most_influential_economic_symbol",
                "positive_exclusion_fraction_at_focus_cost",
                "largest_absolute_contribution_symbol",
                "largest_absolute_contribution_share",
                "top_three_absolute_contribution_share",
            ]
        ]
        .sort_values(
            "worst_exclusion_rank_ic",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()
    print(
        "Worst excluded symbol per model "
        "by break-even cost"
    )

    worst_economic = (
        exclusion_summary.sort_values(
            [
                "model",
                "pooled_break_even_cost_bps",
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
                "excluded_symbol",
                "mean_rank_ic",
                "rank_ic_delta",
                "pooled_break_even_cost_bps",
                "break_even_delta_bps",
                "pooled_annualized_net_return_at_focus_cost",
                "positive_cohort_fraction_at_focus_cost",
            ]
        ]
        .sort_values(
            "pooled_break_even_cost_bps"
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
