# Block-bootstrap uncertainty protocol

## Purpose

Point estimates do not show how much sampling uncertainty remains after
accounting for temporal dependence.

This stage attaches deterministic percentile confidence intervals to the
frozen out-of-sample predictive and economic results.

No model is retrained and no hyperparameter is selected using bootstrap output.

## Predictive bootstrap

For every model, cross-sectional Rank IC is first calculated separately at
each out-of-sample timestamp.

A circular moving-block bootstrap resamples that timestamp series using blocks
of:

| Bars | Minutes |
|---:|---:|
| 12 | 60 |
| 24 | 120 |
| 48 | 240 |

The 60-minute block is the primary specification because the original label
horizon overlaps across 12 consecutive five-minute timestamps.

Longer blocks test sensitivity to more persistent dependence.

## Undefined timestamp correlations

A timestamp-level Rank IC is undefined when either the prediction ranks or
target ranks have zero cross-sectional variance.

Such timestamps remain in their original temporal positions and are represented
as missing correlations. Each bootstrap resample jointly resamples the
correlation values and a valid-observation indicator, then divides the sampled
correlation sum by the sampled valid count.

This preserves the original time spacing without assigning an artificial zero
correlation or compressing the series by deleting timestamps.

## Economic bootstrap

The original 12 hourly cohorts are non-overlapping within each cohort.

Each cohort is resampled independently using circular blocks of:

| Periods | Duration |
|---:|---:|
| 24 | 1 day |
| 72 | 3 days |
| 168 | 1 week |

Gross return and turnover are resampled as paired observations so transaction
costs remain aligned with the corresponding portfolio transition.

For every bootstrap replicate:

1. each cohort is resampled independently;
2. gross mean return and turnover are calculated within each cohort;
3. the 12 cohort means are equally weighted;
4. net return at one basis point is calculated;
5. break-even cost is calculated from gross return divided by turnover.

## Confidence intervals

The analysis uses:

- 1,000 deterministic resamples;
- a fixed recorded random seed;
- two-sided 95% percentile confidence intervals.

Reported probabilities are empirical bootstrap frequencies, including:

- probability that mean Rank IC is positive;
- probability that net mean return at one basis point is positive;
- probability that break-even cost exceeds one basis point.

These probabilities are descriptive bootstrap frequencies, not posterior
probabilities.

## Reproduction requirement

Before uncertainty estimates are accepted, the implementation must reproduce:

- committed aggregate out-of-sample predictive metrics;
- committed portfolio cohort metrics;
- turnover;
- cost-adjusted returns;
- volatility and Sharpe.

## Interpretation

A positive point estimate with a confidence interval crossing zero is not
statistically stable under the chosen dependence structure.

A break-even interval below one basis point confirms that realistic execution
cost remains unsupported even if the gross point estimate is positive.

Bootstrap intervals do not address:

- fixed-universe survivorship bias;
- exchange or market-impact modelling;
- funding;
- latency;
- model-selection uncertainty outside the frozen experiment;
- structural change beyond the observed sample.
