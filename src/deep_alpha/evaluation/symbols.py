from __future__ import annotations

import numpy as np
import pandas as pd

from deep_alpha.evaluation.breadth import (
    PredictionMatrices,
    build_rank_weight_matrix,
)


def exclude_symbol_from_matrices(
    matrices: PredictionMatrices,
    symbol: str,
) -> PredictionMatrices:
    if symbol not in matrices.symbols:
        raise ValueError(
            f"Unknown symbol: {symbol}"
        )

    index = matrices.symbols.index(symbol)

    remaining_symbols = tuple(
        candidate
        for candidate in matrices.symbols
        if candidate != symbol
    )

    return PredictionMatrices(
        timestamps=matrices.timestamps,
        symbols=remaining_symbols,
        predictions=np.delete(
            matrices.predictions,
            index,
            axis=1,
        ),
        execution_returns=np.delete(
            matrices.execution_returns,
            index,
            axis=1,
        ),
    )


def build_symbol_contribution_metrics(
    model: str,
    matrices: PredictionMatrices,
    top_count: int,
) -> pd.DataFrame:
    weights = build_rank_weight_matrix(
        predictions=matrices.predictions,
        top_count=top_count,
    )

    contributions = (
        weights
        * matrices.execution_returns
    )

    cumulative_contribution = (
        contributions.sum(axis=0)
    )

    absolute_contribution = np.abs(
        cumulative_contribution
    )

    absolute_total = float(
        absolute_contribution.sum()
    )

    if absolute_total <= 0:
        raise ValueError(
            "Absolute symbol contribution "
            "is non-positive"
        )

    contribution_share = (
        absolute_contribution
        / absolute_total
    )

    result = pd.DataFrame(
        {
            "model": model,
            "symbol": matrices.symbols,
            "timestamp_count": (
                matrices.timestamp_count
            ),
            "long_selection_count": (
                (weights > 0).sum(axis=0)
            ),
            "short_selection_count": (
                (weights < 0).sum(axis=0)
            ),
            "active_selection_count": (
                (weights != 0).sum(axis=0)
            ),
            "long_selection_fraction": (
                (weights > 0).mean(axis=0)
            ),
            "short_selection_fraction": (
                (weights < 0).mean(axis=0)
            ),
            "active_selection_fraction": (
                (weights != 0).mean(axis=0)
            ),
            "mean_absolute_weight": (
                np.abs(weights).mean(axis=0)
            ),
            "mean_gross_return_contribution": (
                contributions.mean(axis=0)
            ),
            "cumulative_gross_log_return_contribution": (
                cumulative_contribution
            ),
            "absolute_cumulative_contribution": (
                absolute_contribution
            ),
            "absolute_contribution_share": (
                contribution_share
            ),
            "portfolio_gross_cumulative_log_return": (
                float(
                    cumulative_contribution.sum()
                )
            ),
        }
    )

    result[
        "absolute_contribution_rank"
    ] = (
        result[
            "absolute_cumulative_contribution"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return (
        result.sort_values("symbol")
        .reset_index(drop=True)
    )


def summarize_symbol_exclusion_portfolios(
    cohort_metrics: pd.DataFrame,
    focus_cost_bps: float,
    annualization_periods: int,
) -> pd.DataFrame:
    required_columns = {
        "model",
        "excluded_symbol",
        "cost_bps",
        "cohort",
        "period_count",
        "mean_net_return",
        "mean_turnover",
        "cumulative_net_log_return",
    }

    missing_columns = required_columns - set(
        cohort_metrics.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing exclusion portfolio columns: "
            f"{sorted(missing_columns)}"
        )

    if annualization_periods <= 0:
        raise ValueError(
            "annualization_periods must be positive"
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
            "No zero-cost exclusion metrics"
        )

    if focus_cost.empty:
        raise ValueError(
            "No focus-cost exclusion metrics"
        )

    rows = []

    group_columns = [
        "model",
        "excluded_symbol",
    ]

    for keys, zero_group in zero_cost.groupby(
        group_columns,
        sort=True,
    ):
        model, excluded_symbol = keys

        focus_group = focus_cost.loc[
            (
                focus_cost["model"]
                == model
            )
            & (
                focus_cost[
                    "excluded_symbol"
                ]
                == excluded_symbol
            )
        ]

        if focus_group.empty:
            raise ValueError(
                "Missing focus-cost exclusion "
                f"for {model}/{excluded_symbol}"
            )

        zero_cohorts = set(
            zero_group["cohort"]
        )

        focus_cohorts = set(
            focus_group["cohort"]
        )

        if zero_cohorts != focus_cohorts:
            raise ValueError(
                "Zero- and focus-cost cohorts "
                "differ"
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
                "Exclusion turnover must "
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

        focus_period_count = int(
            focus_group[
                "period_count"
            ].sum()
        )

        focus_total_return = float(
            focus_group[
                "cumulative_net_log_return"
            ].sum()
        )

        focus_pooled_mean = (
            focus_total_return
            / focus_period_count
        )

        rows.append(
            {
                "model": str(model),
                "excluded_symbol": str(
                    excluded_symbol
                ),
                "cohort_count": len(
                    zero_group
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
                "mean_individual_break_even_cost_bps": (
                    float(
                        individual_break_even.mean()
                    )
                ),
                "median_individual_break_even_cost_bps": (
                    float(
                        individual_break_even.median()
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
                "mean_turnover": float(
                    zero_group[
                        "mean_turnover"
                    ].mean()
                ),
                "focus_cost_bps": (
                    focus_cost_bps
                ),
                "pooled_annualized_net_return_at_focus_cost": (
                    focus_pooled_mean
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
                "excluded_symbol",
            ]
        )
        .reset_index(drop=True)
    )


def build_symbol_dependence_model_summary(
    exclusion_summary: pd.DataFrame,
    contribution_metrics: pd.DataFrame,
) -> pd.DataFrame:
    required_exclusion_columns = {
        "model",
        "excluded_symbol",
        "full_mean_rank_ic",
        "mean_rank_ic",
        "rank_ic_delta",
        "full_break_even_cost_bps",
        "pooled_break_even_cost_bps",
        "break_even_delta_bps",
        "pooled_annualized_net_return_at_focus_cost",
    }

    missing_exclusion_columns = (
        required_exclusion_columns
        - set(exclusion_summary.columns)
    )

    if missing_exclusion_columns:
        raise ValueError(
            "Missing exclusion summary columns: "
            f"{sorted(missing_exclusion_columns)}"
        )

    required_contribution_columns = {
        "model",
        "symbol",
        "absolute_contribution_share",
    }

    missing_contribution_columns = (
        required_contribution_columns
        - set(contribution_metrics.columns)
    )

    if missing_contribution_columns:
        raise ValueError(
            "Missing contribution columns: "
            f"{sorted(missing_contribution_columns)}"
        )

    rows = []

    for model, exclusions in (
        exclusion_summary.groupby(
            "model",
            sort=True,
        )
    ):
        contributions = (
            contribution_metrics.loc[
                contribution_metrics[
                    "model"
                ]
                == model
            ]
        )

        if contributions.empty:
            raise ValueError(
                f"No contribution metrics "
                f"for {model}"
            )

        prediction_influence_index = (
            exclusions[
                "rank_ic_delta"
            ]
            .abs()
            .idxmax()
        )

        economic_influence_index = (
            exclusions[
                "break_even_delta_bps"
            ]
            .abs()
            .idxmax()
        )

        contribution_index = (
            contributions[
                "absolute_contribution_share"
            ]
            .idxmax()
        )

        contribution_shares = (
            contributions[
                "absolute_contribution_share"
            ].to_numpy(dtype=np.float64)
        )

        top_three_share = float(
            contributions.nlargest(
                3,
                "absolute_contribution_share",
            )[
                "absolute_contribution_share"
            ].sum()
        )

        rank_ic_values = exclusions[
            "mean_rank_ic"
        ].to_numpy(dtype=np.float64)

        break_even_values = exclusions[
            "pooled_break_even_cost_bps"
        ].to_numpy(dtype=np.float64)

        focus_returns = exclusions[
            "pooled_annualized_net_return_at_focus_cost"
        ].to_numpy(dtype=np.float64)

        rows.append(
            {
                "model": str(model),
                "excluded_symbol_count": len(
                    exclusions
                ),
                "full_mean_rank_ic": float(
                    exclusions[
                        "full_mean_rank_ic"
                    ].iloc[0]
                ),
                "mean_exclusion_rank_ic": float(
                    rank_ic_values.mean()
                ),
                "worst_exclusion_rank_ic": float(
                    rank_ic_values.min()
                ),
                "best_exclusion_rank_ic": float(
                    rank_ic_values.max()
                ),
                "positive_exclusion_rank_ic_fraction": (
                    float(
                        (
                            rank_ic_values > 0
                        ).mean()
                    )
                ),
                "maximum_absolute_rank_ic_delta": (
                    float(
                        exclusions[
                            "rank_ic_delta"
                        ].abs().max()
                    )
                ),
                "most_influential_prediction_symbol": (
                    str(
                        exclusions.loc[
                            prediction_influence_index,
                            "excluded_symbol",
                        ]
                    )
                ),
                "full_break_even_cost_bps": float(
                    exclusions[
                        "full_break_even_cost_bps"
                    ].iloc[0]
                ),
                "mean_exclusion_break_even_cost_bps": (
                    float(
                        break_even_values.mean()
                    )
                ),
                "worst_exclusion_break_even_cost_bps": (
                    float(
                        break_even_values.min()
                    )
                ),
                "best_exclusion_break_even_cost_bps": (
                    float(
                        break_even_values.max()
                    )
                ),
                "positive_exclusion_break_even_fraction": (
                    float(
                        (
                            break_even_values > 0
                        ).mean()
                    )
                ),
                "maximum_absolute_break_even_delta_bps": (
                    float(
                        exclusions[
                            "break_even_delta_bps"
                        ].abs().max()
                    )
                ),
                "most_influential_economic_symbol": (
                    str(
                        exclusions.loc[
                            economic_influence_index,
                            "excluded_symbol",
                        ]
                    )
                ),
                "positive_exclusion_fraction_at_focus_cost": (
                    float(
                        (
                            focus_returns > 0
                        ).mean()
                    )
                ),
                "largest_absolute_contribution_symbol": (
                    str(
                        contributions.loc[
                            contribution_index,
                            "symbol",
                        ]
                    )
                ),
                "largest_absolute_contribution_share": (
                    float(
                        contributions.loc[
                            contribution_index,
                            "absolute_contribution_share",
                        ]
                    )
                ),
                "top_three_absolute_contribution_share": (
                    top_three_share
                ),
                "absolute_contribution_hhi": float(
                    np.square(
                        contribution_shares
                    ).sum()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "worst_exclusion_rank_ic",
            ascending=False,
        )
        .reset_index(drop=True)
    )
