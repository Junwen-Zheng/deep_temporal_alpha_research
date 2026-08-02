from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_alpha.config import load_yaml
from deep_alpha.data.download import sha256_file
from deep_alpha.data.splits import (
    build_fold_masks,
    parse_fold_definition,
)
from deep_alpha.evaluation.alignment import (
    assert_key_alignment,
    build_bar_alignment_reference,
    cross_sectionally_demean,
    maximum_absolute_error,
    maximum_timestamp_error_nanoseconds,
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


def ordered_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return (
        frame.sort_values(
            ["timestamp", "symbol"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def main() -> None:
    audit_config_path = Path(
        "configs/alignment_audit.yaml"
    )

    audit_config = load_yaml(
        audit_config_path
    )["alignment_audit"]

    data_config_path = Path(
        audit_config["data_config"]
    )

    research_config_path = Path(
        audit_config["research_config"]
    )

    evaluation_config_path = Path(
        audit_config["evaluation_config"]
    )

    portfolio_manifest_path = Path(
        audit_config["portfolio_manifest"]
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

    portfolio_manifest = json.loads(
        portfolio_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    horizon_bars = int(
        audit_config["horizon_bars"]
    )

    entry_offset_bars = int(
        audit_config[
            "entry_offset_bars"
        ]
    )

    exit_offset_bars = int(
        audit_config[
            "exit_offset_bars"
        ]
    )

    interval_minutes = int(
        audit_config["interval_minutes"]
    )

    absolute_tolerance = float(
        audit_config[
            "absolute_tolerance"
        ]
    )

    if (
        portfolio_manifest[
            "execution_convention"
        ]
        != "next_bar_open_to_horizon_close"
    ):
        raise ValueError(
            "Unexpected portfolio execution convention"
        )

    if (
        int(
            portfolio_manifest[
                "holding_bars"
            ]
        )
        != horizon_bars
    ):
        raise ValueError(
            "Portfolio holding horizon differs "
            "from the label horizon"
        )

    if (
        int(
            portfolio_manifest[
                "interval_minutes"
            ]
        )
        != interval_minutes
    ):
        raise ValueError(
            "Portfolio bar interval differs "
            "from the alignment audit"
        )

    research_manifest_path = Path(
        research_config["manifest_path"]
    )

    research_manifest = json.loads(
        research_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    panel_path = Path(
        research_manifest["output_path"]
    )

    print("Loading research panel", flush=True)

    panel = pd.read_parquet(
        panel_path,
        columns=[
            "timestamp",
            "label_end_timestamp",
            "symbol",
            "raw_target",
            "target",
        ],
    )

    panel = ordered_frame(panel)

    symbols = [
        str(symbol)
        for symbol in data_config["symbols"]
    ]

    processed_directory = Path(
        data_config["processed_dir"]
    )

    symbol_rows: list[dict[str, Any]] = []
    execution_frames = []

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(symbols):02d}] "
            f"Auditing {symbol}",
            flush=True,
        )

        bars_path = (
            processed_directory
            / "bars"
            / f"symbol={symbol}"
            / "part-00000.parquet"
        )

        bars = pd.read_parquet(
            bars_path,
            columns=[
                "timestamp",
                "open",
                "close",
            ],
        )

        reference = (
            build_bar_alignment_reference(
                bars=bars,
                horizon_bars=horizon_bars,
                entry_offset_bars=(
                    entry_offset_bars
                ),
                exit_offset_bars=(
                    exit_offset_bars
                ),
                interval_minutes=(
                    interval_minutes
                ),
            )
        )

        symbol_panel = panel.loc[
            panel["symbol"] == symbol
        ].copy()

        aligned = symbol_panel.merge(
            reference,
            on="timestamp",
            how="left",
            validate="one_to_one",
        )

        required_reference_columns = [
            "expected_label_end_timestamp",
            "expected_raw_target",
            "expected_execution_return",
        ]

        if aligned[
            required_reference_columns
        ].isna().any().any():
            raise ValueError(
                f"Missing reconstructed values "
                f"for {symbol}"
            )

        timestamp_error = (
            maximum_timestamp_error_nanoseconds(
                actual=aligned[
                    "label_end_timestamp"
                ],
                expected=aligned[
                    "expected_label_end_timestamp"
                ],
            )
        )

        raw_target_error = (
            maximum_absolute_error(
                actual=aligned[
                    "raw_target"
                ].to_numpy(),
                expected=aligned[
                    "expected_raw_target"
                ].to_numpy(),
            )
        )

        if timestamp_error != 0:
            raise ValueError(
                f"Label-end timestamp mismatch "
                f"for {symbol}: "
                f"{timestamp_error} ns"
            )

        if raw_target_error > absolute_tolerance:
            raise ValueError(
                f"Raw-target reconstruction "
                f"failed for {symbol}: "
                f"{raw_target_error}"
            )

        execution_values = aligned[
            "expected_execution_return"
        ].to_numpy(dtype=np.float64)

        raw_values = aligned[
            "raw_target"
        ].to_numpy(dtype=np.float64)

        execution_minus_label = (
            execution_values
            - raw_values
        )

        symbol_rows.append(
            {
                "symbol": symbol,
                "bar_rows": len(bars),
                "panel_rows": len(
                    symbol_panel
                ),
                "first_panel_timestamp": (
                    symbol_panel[
                        "timestamp"
                    ].iloc[0]
                ),
                "last_panel_timestamp": (
                    symbol_panel[
                        "timestamp"
                    ].iloc[-1]
                ),
                "max_label_end_error_ns": (
                    timestamp_error
                ),
                "max_raw_target_abs_error": (
                    raw_target_error
                ),
                "mean_raw_target": float(
                    raw_values.mean()
                ),
                "mean_execution_return": (
                    float(
                        execution_values.mean()
                    )
                ),
                "mean_execution_minus_label": (
                    float(
                        execution_minus_label.mean()
                    )
                ),
                "mean_abs_execution_minus_label": (
                    float(
                        np.abs(
                            execution_minus_label
                        ).mean()
                    )
                ),
            }
        )

        execution_frames.append(
            aligned[
                [
                    "timestamp",
                    "symbol",
                    "expected_execution_return",
                ]
            ].rename(
                columns={
                    "expected_execution_return": (
                        "execution_return"
                    )
                }
            )
        )

    symbol_metrics = (
        pd.DataFrame(symbol_rows)
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    execution_reference = ordered_frame(
        pd.concat(
            execution_frames,
            ignore_index=True,
        )
    )

    expected_target = (
        cross_sectionally_demean(
            frame=panel,
            value_column="raw_target",
        )
    )

    target_error = maximum_absolute_error(
        actual=panel["target"].to_numpy(),
        expected=expected_target,
    )

    if target_error > absolute_tolerance:
        raise ValueError(
            "Cross-sectional target "
            f"reconstruction failed: {target_error}"
        )

    target_mean_by_timestamp = (
        panel.groupby(
            "timestamp",
            sort=False,
        )["target"].mean()
    )

    maximum_target_mean = float(
        target_mean_by_timestamp.abs().max()
    )

    if maximum_target_mean > absolute_tolerance:
        raise ValueError(
            "Cross-sectionally demeaned targets "
            "do not have zero mean"
        )

    expected_horizon_nanoseconds = (
        pd.Timedelta(
            minutes=(
                interval_minutes
                * horizon_bars
            )
        ).value
    )

    actual_horizons = (
        pd.DatetimeIndex(
            panel["label_end_timestamp"]
        ).asi8
        - pd.DatetimeIndex(
            panel["timestamp"]
        ).asi8
    )

    if not np.all(
        actual_horizons
        == expected_horizon_nanoseconds
    ):
        raise ValueError(
            "Label horizons are not exactly "
            f"{horizon_bars} bars"
        )

    split_rows = []
    prediction_rows = []

    model_configurations = (
        evaluation_config["models"]
    )

    for fold_configuration in research_config[
        "folds"
    ]:
        fold = parse_fold_definition(
            fold_configuration
        )

        masks = build_fold_masks(
            panel,
            fold,
        )

        split_definitions = [
            (
                "train",
                masks.train,
                None,
                fold.train_end,
            ),
            (
                "validation",
                masks.validation,
                fold.train_end,
                fold.validation_end,
            ),
            (
                "test",
                masks.test,
                fold.validation_end,
                fold.test_end,
            ),
        ]

        for (
            split_name,
            mask,
            lower_boundary,
            upper_boundary,
        ) in split_definitions:
            subset = panel.loc[mask]

            if subset.empty:
                raise ValueError(
                    f"Fold {fold.fold_id} "
                    f"{split_name} is empty"
                )

            if (
                lower_boundary is not None
                and not (
                    subset["timestamp"]
                    > lower_boundary
                ).all()
            ):
                raise ValueError(
                    f"Fold {fold.fold_id} "
                    f"{split_name} crosses "
                    "its lower boundary"
                )

            if not (
                subset["label_end_timestamp"]
                <= upper_boundary
            ).all():
                raise ValueError(
                    f"Fold {fold.fold_id} "
                    f"{split_name} labels cross "
                    "their upper boundary"
                )

            split_rows.append(
                {
                    "fold": fold.fold_id,
                    "split": split_name,
                    "rows": len(subset),
                    "timestamps": (
                        subset[
                            "timestamp"
                        ].nunique()
                    ),
                    "first_timestamp": (
                        subset[
                            "timestamp"
                        ].min()
                    ),
                    "last_timestamp": (
                        subset[
                            "timestamp"
                        ].max()
                    ),
                    "last_label_end_timestamp": (
                        subset[
                            "label_end_timestamp"
                        ].max()
                    ),
                    "lower_boundary": (
                        lower_boundary
                    ),
                    "upper_boundary": (
                        upper_boundary
                    ),
                    "maximum_label_horizon_bars": (
                        int(
                            (
                                pd.DatetimeIndex(
                                    subset[
                                        "label_end_timestamp"
                                    ]
                                ).asi8
                                - pd.DatetimeIndex(
                                    subset[
                                        "timestamp"
                                    ]
                                ).asi8
                            ).max()
                            // (
                                pd.Timedelta(
                                    minutes=(
                                        interval_minutes
                                    )
                                ).value
                            )
                        )
                    ),
                }
            )

        expected_test = ordered_frame(
            panel.loc[
                masks.test,
                [
                    "timestamp",
                    "symbol",
                    "raw_target",
                    "target",
                ],
            ]
        )

        expected_execution = ordered_frame(
            execution_reference.merge(
                expected_test[
                    [
                        "timestamp",
                        "symbol",
                    ]
                ],
                on=[
                    "timestamp",
                    "symbol",
                ],
                how="inner",
                validate="one_to_one",
            )
        )

        assert_key_alignment(
            expected=expected_test,
            actual=expected_execution,
        )

        for (
            model_name,
            model_configuration,
        ) in model_configurations.items():
            prediction_path = (
                Path(
                    model_configuration[
                        "directory"
                    ]
                )
                / (
                    f"{model_name}_fold_"
                    f"{fold.fold_id}_test.parquet"
                )
            )

            predictions = pd.read_parquet(
                prediction_path,
                columns=[
                    "timestamp",
                    "symbol",
                    "raw_target",
                    "target",
                    "prediction",
                ],
            )

            predictions = ordered_frame(
                predictions
            )

            assert_key_alignment(
                expected=expected_test,
                actual=predictions,
            )

            raw_error = maximum_absolute_error(
                actual=predictions[
                    "raw_target"
                ].to_numpy(),
                expected=expected_test[
                    "raw_target"
                ].to_numpy(),
            )

            demeaned_error = (
                maximum_absolute_error(
                    actual=predictions[
                        "target"
                    ].to_numpy(),
                    expected=expected_test[
                        "target"
                    ].to_numpy(),
                )
            )

            if raw_error > absolute_tolerance:
                raise ValueError(
                    f"{model_name} fold "
                    f"{fold.fold_id} raw-target "
                    "alignment failed"
                )

            if demeaned_error > absolute_tolerance:
                raise ValueError(
                    f"{model_name} fold "
                    f"{fold.fold_id} target "
                    "alignment failed"
                )

            if not np.isfinite(
                predictions[
                    "prediction"
                ].to_numpy(
                    dtype=np.float64
                )
            ).all():
                raise ValueError(
                    f"{model_name} fold "
                    f"{fold.fold_id} contains "
                    "non-finite predictions"
                )

            prediction_rows.append(
                {
                    "model": str(
                        model_name
                    ),
                    "fold": fold.fold_id,
                    "rows": len(
                        predictions
                    ),
                    "timestamps": (
                        predictions[
                            "timestamp"
                        ].nunique()
                    ),
                    "max_raw_target_abs_error": (
                        raw_error
                    ),
                    "max_target_abs_error": (
                        demeaned_error
                    ),
                    "execution_rows": len(
                        expected_execution
                    ),
                    "prediction_file": str(
                        prediction_path
                    ),
                    "prediction_file_sha256": (
                        sha256_file(
                            prediction_path
                        )
                    ),
                }
            )

    split_metrics = (
        pd.DataFrame(split_rows)
        .sort_values(
            ["fold", "split"]
        )
        .reset_index(drop=True)
    )

    prediction_metrics = (
        pd.DataFrame(prediction_rows)
        .sort_values(
            ["model", "fold"]
        )
        .reset_index(drop=True)
    )

    summary = {
        "schema_version": 1,
        "panel_rows": len(panel),
        "panel_timestamps": (
            panel["timestamp"].nunique()
        ),
        "symbol_count": len(symbols),
        "rows_per_symbol": int(
            len(panel) / len(symbols)
        ),
        "interval_minutes": interval_minutes,
        "horizon_bars": horizon_bars,
        "horizon_minutes": (
            interval_minutes
            * horizon_bars
        ),
        "entry_offset_bars": (
            entry_offset_bars
        ),
        "exit_offset_bars": (
            exit_offset_bars
        ),
        "execution_convention": (
            "next_bar_open_to_horizon_close"
        ),
        "maximum_raw_target_abs_error": (
            float(
                symbol_metrics[
                    "max_raw_target_abs_error"
                ].max()
            )
        ),
        "maximum_label_end_error_ns": (
            int(
                symbol_metrics[
                    "max_label_end_error_ns"
                ].max()
            )
        ),
        "maximum_demeaned_target_abs_error": (
            target_error
        ),
        "maximum_cross_sectional_target_mean": (
            maximum_target_mean
        ),
        "all_label_horizons_exact": True,
        "all_split_labels_contained": True,
        "all_prediction_keys_aligned": True,
        "all_prediction_targets_aligned": True,
        "portfolio_manifest_execution_verified": (
            True
        ),
    }

    evidence_directory = Path(
        audit_config["evidence_dir"]
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    symbol_metrics_path = (
        evidence_directory
        / "alignment_symbol_metrics.csv"
    )

    split_metrics_path = (
        evidence_directory
        / "alignment_split_metrics.csv"
    )

    prediction_metrics_path = (
        evidence_directory
        / "alignment_prediction_metrics.csv"
    )

    summary_path = (
        evidence_directory
        / "alignment_summary.json"
    )

    write_csv(
        symbol_metrics,
        symbol_metrics_path,
    )

    write_csv(
        split_metrics,
        split_metrics_path,
    )

    write_csv(
        prediction_metrics,
        prediction_metrics_path,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = (
        evidence_directory
        / "alignment_audit_manifest.json"
    )

    evidence_files = [
        symbol_metrics_path,
        split_metrics_path,
        prediction_metrics_path,
        summary_path,
    ]

    manifest = {
        "schema_version": 1,
        "alignment_config": str(
            audit_config_path
        ),
        "alignment_config_sha256": (
            sha256_file(
                audit_config_path
            )
        ),
        "data_config": str(
            data_config_path
        ),
        "data_config_sha256": sha256_file(
            data_config_path
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
        "research_manifest": str(
            research_manifest_path
        ),
        "research_manifest_sha256": (
            sha256_file(
                research_manifest_path
            )
        ),
        "research_panel": str(
            panel_path
        ),
        "research_panel_sha256": (
            sha256_file(panel_path)
        ),
        "portfolio_manifest": str(
            portfolio_manifest_path
        ),
        "portfolio_manifest_sha256": (
            sha256_file(
                portfolio_manifest_path
            )
        ),
        "models": [
            str(model)
            for model in model_configurations
        ],
        "prediction_files": (
            prediction_metrics[
                [
                    "model",
                    "fold",
                    "prediction_file",
                    "prediction_file_sha256",
                ]
            ].to_dict(
                orient="records"
            )
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
        "Label and execution alignment "
        "audit complete"
    )

    print()
    print("Alignment summary")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print()
    print("Prediction alignment")

    print(
        prediction_metrics[
            [
                "model",
                "fold",
                "rows",
                "max_raw_target_abs_error",
                "max_target_abs_error",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
