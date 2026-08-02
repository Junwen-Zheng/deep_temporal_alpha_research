# Portfolio-breadth sensitivity protocol

## Purpose

The primary economic evaluation holds the four highest-ranked and four
lowest-ranked contracts.

This stage tests whether that result depends on the chosen number of contracts
on each side.

## Portfolio widths

The analysis evaluates the top and bottom:

- 1 contract;
- 2 contracts;
- 3 contracts;
- 4 contracts;
- 5 contracts;
- 6 contracts;
- 8 contracts;
- 10 contracts.

The 20-contract universe therefore produces portfolios containing between 2
and 20 active positions.

## Weighting

Every portfolio remains dollar-neutral and unit-gross.

For breadth `k`:

    long weight per contract = 0.5 / k

    short weight per contract = -0.5 / k

The highest `k` predictions are long and the lowest `k` predictions are short.

Ties are resolved deterministically using the existing alphabetical symbol
ordering.

## Execution and holding period

The execution convention remains unchanged:

    entry = open[t + 1]

    exit = close[t + 12]

The holding period remains 60 minutes.

The 12 staggered hourly cohorts and discontinuity resets are identical to the
primary economic evaluation.

## Transaction-cost sensitivity

Each breadth is evaluated at:

- 0 basis points;
- 0.25 basis points;
- 0.5 basis points;
- 1 basis point;
- 2 basis points;
- 5 basis points;
- 10 basis points.

Costs are applied per unit of one-way traded notional.

## Reported evidence

For every model and breadth, the evidence includes:

- cohort-level mean return and Sharpe;
- turnover;
- pooled and equal-weighted break-even cost;
- worst and best cohort break-even cost;
- fraction of profitable cohorts;
- zero-cost performance;
- cost-adjusted portfolio summaries.

The four-contract-per-side result must exactly reproduce the previously
committed portfolio evidence.

## Interpretation

A narrow portfolio may concentrate the strongest ranks but usually increases
idiosyncratic risk.

A broader portfolio may reduce volatility and turnover, but can dilute weak
cross-sectional ordering.

A credible result should not depend on one arbitrarily selected portfolio
width.
