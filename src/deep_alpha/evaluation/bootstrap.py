from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def validate_block_lengths(
    block_lengths: Sequence[int],
) -> tuple[int, ...]:
    values = tuple(
        sorted(
            {
                int(value)
                for value in block_lengths
            }
        )
    )

    if not values:
        raise ValueError(
            "At least one block length "
            "is required"
        )

    if any(value <= 0 for value in values):
        raise ValueError(
            "Block lengths must be positive"
        )

    return values


def _circular_window_sums(
    values: np.ndarray,
    window_length: int,
) -> np.ndarray:
    observation_count = values.shape[0]

    if window_length <= 0:
        raise ValueError(
            "window_length must be positive"
        )

    if window_length > observation_count:
        raise ValueError(
            "window_length cannot exceed "
            "the observation count"
        )

    if window_length == 1:
        return values.copy()

    extended = np.concatenate(
        [
            values,
            values[
                : window_length - 1
            ],
        ],
        axis=0,
    )

    cumulative = np.concatenate(
        [
            np.zeros(
                (
                    1,
                    values.shape[1],
                ),
                dtype=np.float64,
            ),
            np.cumsum(
                extended,
                axis=0,
                dtype=np.float64,
            ),
        ],
        axis=0,
    )

    return (
        cumulative[
            window_length:
        ]
        - cumulative[
            :-window_length
        ]
    )


def circular_block_bootstrap_means(
    values: np.ndarray | pd.Series,
    resamples: int,
    block_length: int,
    seed: int,
    chunk_size: int = 128,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    was_one_dimensional = (
        array.ndim == 1
    )

    if was_one_dimensional:
        array = array[:, None]

    if array.ndim != 2:
        raise ValueError(
            "Bootstrap values must be "
            "one- or two-dimensional"
        )

    observation_count = array.shape[0]

    if observation_count == 0:
        raise ValueError(
            "Bootstrap values are empty"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Bootstrap values contain "
            "non-finite values"
        )

    if resamples <= 0:
        raise ValueError(
            "resamples must be positive"
        )

    if block_length <= 0:
        raise ValueError(
            "block_length must be positive"
        )

    if block_length > observation_count:
        raise ValueError(
            "block_length cannot exceed "
            "the observation count"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    full_block_count = (
        observation_count
        // block_length
    )

    remainder = (
        observation_count
        % block_length
    )

    block_sums = (
        _circular_window_sums(
            values=array,
            window_length=block_length,
        )
    )

    partial_sums = None

    if remainder > 0:
        partial_sums = (
            _circular_window_sums(
                values=array,
                window_length=remainder,
            )
        )

    generator = np.random.default_rng(
        seed
    )

    result = np.empty(
        (
            resamples,
            array.shape[1],
        ),
        dtype=np.float64,
    )

    for start in range(
        0,
        resamples,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            resamples,
        )

        size = stop - start

        block_starts = generator.integers(
            low=0,
            high=observation_count,
            size=(
                size,
                full_block_count,
            ),
        )

        sampled_sums = block_sums[
            block_starts
        ].sum(axis=1)

        if remainder > 0:
            if partial_sums is None:
                raise RuntimeError(
                    "Partial block sums "
                    "were not constructed"
                )

            partial_starts = (
                generator.integers(
                    low=0,
                    high=observation_count,
                    size=size,
                )
            )

            sampled_sums += partial_sums[
                partial_starts
            ]

        result[start:stop] = (
            sampled_sums
            / observation_count
        )

    if was_one_dimensional:
        return result[:, 0]

    return result



def circular_block_bootstrap_nanmeans(
    values: np.ndarray | pd.Series,
    resamples: int,
    block_length: int,
    seed: int,
    chunk_size: int = 128,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            "NaN-aware bootstrap values must "
            "be one-dimensional"
        )

    if len(array) == 0:
        raise ValueError(
            "NaN-aware bootstrap values "
            "are empty"
        )

    valid = np.isfinite(array)

    if not valid.any():
        raise ValueError(
            "NaN-aware bootstrap has no "
            "finite observations"
        )

    paired_values = np.column_stack(
        [
            np.where(
                valid,
                array,
                0.0,
            ),
            valid.astype(np.float64),
        ]
    )

    sampled_means = (
        circular_block_bootstrap_means(
            values=paired_values,
            resamples=resamples,
            block_length=block_length,
            seed=seed,
            chunk_size=chunk_size,
        )
    )

    valid_fraction = sampled_means[:, 1]

    if np.any(valid_fraction <= 0):
        raise ValueError(
            "A bootstrap resample contains "
            "no valid observations"
        )

    return (
        sampled_means[:, 0]
        / valid_fraction
    )


def build_timestamp_rank_ic_series(
    frame: pd.DataFrame,
    expected_symbol_count: int,
) -> pd.DataFrame:
    required_columns = {
        "timestamp",
        "symbol",
        "prediction",
        "target",
    }

    missing_columns = required_columns - set(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing Rank IC columns: "
            f"{sorted(missing_columns)}"
        )

    if expected_symbol_count <= 1:
        raise ValueError(
            "expected_symbol_count must "
            "exceed one"
        )

    ordered = (
        frame.sort_values(
            [
                "timestamp",
                "symbol",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if ordered.empty:
        raise ValueError(
            "Prediction frame is empty"
        )

    if ordered.duplicated(
        [
            "timestamp",
            "symbol",
        ]
    ).any():
        raise ValueError(
            "Duplicate prediction keys"
        )

    group_sizes = ordered.groupby(
        "timestamp",
        sort=False,
    ).size()

    if not group_sizes.eq(
        expected_symbol_count
    ).all():
        raise ValueError(
            "Prediction cross-sections "
            "are incomplete"
        )

    prediction_matrix = (
        ordered.pivot(
            index="timestamp",
            columns="symbol",
            values="prediction",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    target_matrix = (
        ordered.pivot(
            index="timestamp",
            columns="symbol",
            values="target",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    if (
        prediction_matrix.shape
        != target_matrix.shape
    ):
        raise ValueError(
            "Prediction and target matrix "
            "shapes differ"
        )

    if (
        prediction_matrix.shape[1]
        != expected_symbol_count
    ):
        raise ValueError(
            "Unexpected Rank IC matrix width"
        )

    prediction_values = (
        prediction_matrix.to_numpy(
            dtype=np.float64
        )
    )

    target_values = (
        target_matrix.to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        prediction_values
    ).all():
        raise ValueError(
            "Predictions contain "
            "non-finite values"
        )

    if not np.isfinite(
        target_values
    ).all():
        raise ValueError(
            "Targets contain "
            "non-finite values"
        )

    prediction_ranks = (
        pd.DataFrame(
            prediction_values
        )
        .rank(
            axis=1,
            method="average",
        )
        .to_numpy(dtype=np.float64)
    )

    target_ranks = (
        pd.DataFrame(
            target_values
        )
        .rank(
            axis=1,
            method="average",
        )
        .to_numpy(dtype=np.float64)
    )

    prediction_centered = (
        prediction_ranks
        - prediction_ranks.mean(
            axis=1,
            keepdims=True,
        )
    )

    target_centered = (
        target_ranks
        - target_ranks.mean(
            axis=1,
            keepdims=True,
        )
    )

    numerator = (
        prediction_centered
        * target_centered
    ).sum(axis=1)

    denominator = np.sqrt(
        np.square(
            prediction_centered
        ).sum(axis=1)
        * np.square(
            target_centered
        ).sum(axis=1)
    )

    rank_ic = np.divide(
        numerator,
        denominator,
        out=np.full(
            len(numerator),
            np.nan,
            dtype=np.float64,
        ),
        where=denominator > 0,
    )

    result = pd.DataFrame(
        {
            "timestamp": (
                prediction_matrix.index
            ),
            "rank_ic": rank_ic,
        }
    )

    valid_count = int(
        np.isfinite(
            result[
                "rank_ic"
            ].to_numpy(
                dtype=np.float64
            )
        ).sum()
    )

    if valid_count == 0:
        raise ValueError(
            "No valid timestamp Rank IC "
            "observations"
        )

    return result.reset_index(
        drop=True
    )


def summarize_bootstrap_distribution(
    samples: np.ndarray | pd.Series,
    observed: float,
    confidence_level: float,
) -> dict[str, float]:
    values = np.asarray(
        samples,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "Bootstrap samples must be "
            "one-dimensional"
        )

    if len(values) == 0:
        raise ValueError(
            "Bootstrap samples are empty"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Bootstrap samples contain "
            "non-finite values"
        )

    if not np.isfinite(observed):
        raise ValueError(
            "Observed statistic must be finite"
        )

    if not (
        0.0
        < confidence_level
        < 1.0
    ):
        raise ValueError(
            "confidence_level must lie "
            "between zero and one"
        )

    alpha = (
        1.0 - confidence_level
    ) / 2.0

    return {
        "observed": float(observed),
        "bootstrap_mean": float(
            values.mean()
        ),
        "bootstrap_std": float(
            values.std(ddof=1)
        ),
        "confidence_lower": float(
            np.quantile(
                values,
                alpha,
            )
        ),
        "confidence_upper": float(
            np.quantile(
                values,
                1.0 - alpha,
            )
        ),
        "probability_positive": float(
            (values > 0).mean()
        ),
    }


def build_portfolio_bootstrap_distribution(
    periods: pd.DataFrame,
    resamples: int,
    block_length_periods: int,
    seed: int,
    cost_bps: float,
    annualization_periods: int,
) -> tuple[
    pd.DataFrame,
    dict[str, float],
]:
    required_columns = {
        "cohort",
        "timestamp",
        "gross_return",
        "turnover",
    }

    missing_columns = required_columns - set(
        periods.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing portfolio bootstrap "
            f"columns: {sorted(missing_columns)}"
        )

    if annualization_periods <= 0:
        raise ValueError(
            "annualization_periods must "
            "be positive"
        )

    if not np.isfinite(cost_bps):
        raise ValueError(
            "cost_bps must be finite"
        )

    if cost_bps < 0:
        raise ValueError(
            "cost_bps cannot be negative"
        )

    cohort_gross_samples = []
    cohort_turnover_samples = []
    observed_gross_means = []
    observed_turnover_means = []

    cohorts = sorted(
        int(value)
        for value in periods[
            "cohort"
        ].unique()
    )

    if not cohorts:
        raise ValueError(
            "No portfolio cohorts found"
        )

    for cohort in cohorts:
        group = (
            periods.loc[
                periods["cohort"]
                == cohort
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        paired_values = group[
            [
                "gross_return",
                "turnover",
            ]
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(
            paired_values
        ).all():
            raise ValueError(
                f"Non-finite values in "
                f"cohort {cohort}"
            )

        if (
            paired_values[:, 1] < 0
        ).any():
            raise ValueError(
                f"Negative turnover in "
                f"cohort {cohort}"
            )

        if (
            block_length_periods
            > len(group)
        ):
            raise ValueError(
                "Portfolio block length "
                f"exceeds cohort {cohort} "
                "observation count"
            )

        samples = (
            circular_block_bootstrap_means(
                values=paired_values,
                resamples=resamples,
                block_length=(
                    block_length_periods
                ),
                seed=(
                    seed
                    + cohort
                    * 100_003
                ),
            )
        )

        cohort_gross_samples.append(
            samples[:, 0]
        )

        cohort_turnover_samples.append(
            samples[:, 1]
        )

        observed_gross_means.append(
            float(
                paired_values[
                    :,
                    0,
                ].mean()
            )
        )

        observed_turnover_means.append(
            float(
                paired_values[
                    :,
                    1,
                ].mean()
            )
        )

    gross_samples = np.column_stack(
        cohort_gross_samples
    ).mean(axis=1)

    turnover_samples = np.column_stack(
        cohort_turnover_samples
    ).mean(axis=1)

    if np.any(
        turnover_samples <= 0
    ):
        raise ValueError(
            "Bootstrapped turnover is "
            "non-positive"
        )

    net_samples = (
        gross_samples
        - turnover_samples
        * cost_bps
        / 10_000.0
    )

    break_even_samples = (
        gross_samples
        / turnover_samples
        * 10_000.0
    )

    observed_gross = float(
        np.mean(
            observed_gross_means
        )
    )

    observed_turnover = float(
        np.mean(
            observed_turnover_means
        )
    )

    if observed_turnover <= 0:
        raise ValueError(
            "Observed mean turnover is "
            "non-positive"
        )

    observed = {
        "gross_mean_return": (
            observed_gross
        ),
        "mean_turnover": (
            observed_turnover
        ),
        "net_mean_return": (
            observed_gross
            - observed_turnover
            * cost_bps
            / 10_000.0
        ),
        "break_even_cost_bps": (
            observed_gross
            / observed_turnover
            * 10_000.0
        ),
        "gross_annualized_return": (
            observed_gross
            * annualization_periods
        ),
        "net_annualized_return": (
            (
                observed_gross
                - observed_turnover
                * cost_bps
                / 10_000.0
            )
            * annualization_periods
        ),
    }

    distribution = pd.DataFrame(
        {
            "resample": np.arange(
                resamples,
                dtype=np.int64,
            ),
            "gross_mean_return": (
                gross_samples
            ),
            "mean_turnover": (
                turnover_samples
            ),
            "net_mean_return": (
                net_samples
            ),
            "break_even_cost_bps": (
                break_even_samples
            ),
            "gross_annualized_return": (
                gross_samples
                * annualization_periods
            ),
            "net_annualized_return": (
                net_samples
                * annualization_periods
            ),
        }
    )

    return distribution, observed
