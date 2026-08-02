# Break-even transaction-cost protocol

## Purpose

The gross portfolio results are positive for several models, while all mean
results are negative at five basis points.

This stage computes the exact cost threshold at which mean out-of-sample net
return becomes zero.

## Cost convention

Transaction cost is measured in basis points per unit of one-way traded
notional.

For a portfolio period:

    net return
    = gross return
    - turnover * cost_bps / 10,000

Turnover is the sum of absolute changes in portfolio weights.

## Cohort break-even cost

For each model and staggered hourly cohort:

    break_even_bps
    = cumulative gross log return
      / total traded notional
      * 10,000

Equivalently:

    break_even_bps
    = mean gross return
      / mean turnover
      * 10,000

A negative break-even value means the cohort loses money even before
transaction costs.

## Model-level break-even values

Two aggregate values are reported.

### Equal-weighted break-even

Each of the 12 start-time cohorts receives equal weight:

    mean cohort gross return
    / mean cohort turnover
    * 10,000

### Pooled break-even

All cohort periods are pooled by observation count:

    total gross log return
    / total traded notional
    * 10,000

The pooled value is the exact cost that makes aggregate cumulative net log
return zero across the complete evaluated population.

## Dense cost curve

Mean-return quantities are calculated from zero to ten basis points in
increments of 0.1 basis points.

The curve reports:

- mean, median, worst and best cohort net return;
- annualized versions of those mean-return quantities;
- fraction of cohorts with positive mean net return;
- pooled mean and cumulative net return;
- mean turnover.

These quantities are exact because mean return is linear in transaction cost.

Sharpe ratios are not interpolated. Their volatility denominator can change
with cost because turnover varies through time.

## Interpretation

The break-even cost is a sensitivity threshold, not an estimate of achievable
execution cost.

A model with a positive but very small threshold remains economically fragile,
particularly when spread, fees, market impact, funding, and latency are not
separately modelled.
