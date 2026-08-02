from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def validate_feature_families(
    feature_columns: Sequence[str],
    feature_families: Mapping[
        str,
        Sequence[str],
    ],
) -> None:
    features = tuple(
        str(value)
        for value in feature_columns
    )

    if not features:
        raise ValueError(
            "Feature columns are empty"
        )

    if len(set(features)) != len(features):
        raise ValueError(
            "Feature columns contain duplicates"
        )

    if not feature_families:
        raise ValueError(
            "Feature families are empty"
        )

    assigned = []

    for family, columns in (
        feature_families.items()
    ):
        family_name = str(family)

        if not family_name:
            raise ValueError(
                "Feature-family name is empty"
            )

        family_columns = [
            str(value)
            for value in columns
        ]

        if not family_columns:
            raise ValueError(
                f"Feature family {family_name} "
                "is empty"
            )

        if (
            len(set(family_columns))
            != len(family_columns)
        ):
            raise ValueError(
                f"Feature family {family_name} "
                "contains duplicates"
            )

        assigned.extend(family_columns)

    unknown = set(assigned) - set(features)

    if unknown:
        raise ValueError(
            "Feature families contain unknown "
            f"features: {sorted(unknown)}"
        )

    counts = pd.Series(
        assigned,
        dtype="object",
    ).value_counts()

    duplicated = sorted(
        str(value)
        for value in counts[
            counts > 1
        ].index
    )

    if duplicated:
        raise ValueError(
            "Features belong to multiple "
            f"families: {duplicated}"
        )

    missing = set(features) - set(assigned)

    if missing:
        raise ValueError(
            "Features are not assigned to "
            f"a family: {sorted(missing)}"
        )


def build_leave_one_family_out_sets(
    feature_columns: Sequence[str],
    feature_families: Mapping[
        str,
        Sequence[str],
    ],
) -> dict[str, list[str]]:
    validate_feature_families(
        feature_columns=feature_columns,
        feature_families=feature_families,
    )

    features = [
        str(value)
        for value in feature_columns
    ]

    result = {
        "full": features,
    }

    for family, removed_columns in (
        feature_families.items()
    ):
        removed = {
            str(value)
            for value in removed_columns
        }

        retained = [
            feature
            for feature in features
            if feature not in removed
        ]

        if not retained:
            raise ValueError(
                f"Removing {family} leaves "
                "no features"
            )

        result[
            f"without_{family}"
        ] = retained

    return result


def build_ablation_portfolio_summary(
    cohort_metrics: pd.DataFrame,
    focus_cost_bps: float,
    annualization_periods: int,
) -> pd.DataFrame:
    required_columns = {
        "model",
        "variant",
        "cost_bps",
        "cohort",
        "period_count",
        "mean_net_return",
        "mean_turnover",
        "annualized_sharpe",
        "cumulative_net_log_return",
    }

    missing_columns = required_columns - set(
        cohort_metrics.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing ablation portfolio "
            f"columns: {sorted(missing_columns)}"
        )

    if annualization_periods <= 0:
        raise ValueError(
            "annualization_periods must "
            "be positive"
        )

    zero_cost = cohort_metrics.loc[
        np.isclose(
            cohort_metrics[
                "cost_bps"
            ].to_numpy(
                dtype=np.float64
            ),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    focus_cost = cohort_metrics.loc[
        np.isclose(
            cohort_metrics[
                "cost_bps"
            ].to_numpy(
                dtype=np.float64
            ),
            focus_cost_bps,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    if zero_cost.empty:
        raise ValueError(
            "No zero-cost ablation rows"
        )

    if focus_cost.empty:
        raise ValueError(
            "No focus-cost ablation rows"
        )

    rows = []

    for keys, zero_group in (
        zero_cost.groupby(
            [
                "model",
                "variant",
            ],
            sort=True,
        )
    ):
        model, variant = keys

        focus_group = focus_cost.loc[
            (
                focus_cost["model"]
                == model
            )
            & (
                focus_cost["variant"]
                == variant
            )
        ]

        if focus_group.empty:
            raise ValueError(
                "Missing focus-cost rows for "
                f"{model}/{variant}"
            )

        if set(
            zero_group["cohort"]
        ) != set(
            focus_group["cohort"]
        ):
            raise ValueError(
                "Zero- and focus-cost cohort "
                "coverage differs"
            )

        total_turnover = float(
            (
                zero_group[
                    "mean_turnover"
                ]
                * zero_group[
                    "period_count"
                ]
            ).sum()
        )

        total_gross_return = float(
            zero_group[
                "cumulative_net_log_return"
            ].sum()
        )

        if total_turnover <= 0:
            raise ValueError(
                "Ablation turnover must "
                "be positive"
            )

        individual_break_even = (
            zero_group[
                "mean_net_return"
            ]
            / zero_group[
                "mean_turnover"
            ]
            * 10_000.0
        )

        focus_periods = int(
            focus_group[
                "period_count"
            ].sum()
        )

        focus_total_return = float(
            focus_group[
                "cumulative_net_log_return"
            ].sum()
        )

        rows.append(
            {
                "model": str(model),
                "variant": str(variant),
                "cohort_count": int(
                    zero_group[
                        "cohort"
                    ].nunique()
                ),
                "total_periods": int(
                    zero_group[
                        "period_count"
                    ].sum()
                ),
                "pooled_break_even_cost_bps": (
                    total_gross_return
                    / total_turnover
                    * 10_000.0
                ),
                "equal_weighted_break_even_cost_bps": (
                    float(
                        zero_group[
                            "mean_net_return"
                        ].mean()
                        / zero_group[
                            "mean_turnover"
                        ].mean()
                        * 10_000.0
                    )
                ),
                "worst_individual_break_even_cost_bps": (
                    float(
                        individual_break_even.min()
                    )
                ),
                "best_individual_break_even_cost_bps": (
                    float(
                        individual_break_even.max()
                    )
                ),
                "mean_zero_cost_sharpe": float(
                    zero_group[
                        "annualized_sharpe"
                    ].mean()
                ),
                "mean_turnover": float(
                    zero_group[
                        "mean_turnover"
                    ].mean()
                ),
                "focus_cost_bps": (
                    focus_cost_bps
                ),
                "pooled_annualized_net_return_at_focus_cost": (
                    focus_total_return
                    / focus_periods
                    * annualization_periods
                ),
                "positive_cohort_fraction_at_focus_cost": (
                    float(
                        (
                            focus_group[
                                "mean_net_return"
                            ]
                            > 0
                        ).mean()
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "model",
                "variant",
            ]
        )
        .reset_index(drop=True)
    )


def add_full_variant_deltas(
    frame: pd.DataFrame,
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    required_columns = {
        "model",
        "variant",
        *metric_columns,
    }

    missing_columns = required_columns - set(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing ablation delta columns: "
            f"{sorted(missing_columns)}"
        )

    result = frame.copy()

    for metric in metric_columns:
        full_values = (
            result.loc[
                result["variant"] == "full",
                [
                    "model",
                    metric,
                ],
            ]
            .rename(
                columns={
                    metric: (
                        f"full_{metric}"
                    )
                }
            )
        )

        if full_values[
            "model"
        ].duplicated().any():
            raise ValueError(
                "Multiple full rows exist "
                f"for {metric}"
            )

        result = result.merge(
            full_values,
            on="model",
            how="left",
            validate="many_to_one",
        )

        full_column = (
            f"full_{metric}"
        )

        if result[
            full_column
        ].isna().any():
            raise ValueError(
                f"Missing full value for "
                f"{metric}"
            )

        result[
            f"{metric}_delta_vs_full"
        ] = (
            result[metric]
            - result[full_column]
        )

    return result
