# Fold and monthly stability protocol

## Purpose

Aggregate out-of-sample metrics can hide periods in which a signal reverses
sign or loses economic value.

This stage decomposes the original 60-minute evaluation by walk-forward fold
and UTC calendar month.

## Frozen inputs

No models are retrained.

The analysis uses the committed predictions from all three test folds and
reconstructs the original next-bar-open execution returns.

The aggregate predictive and portfolio metrics must reproduce their previously
committed evidence before the stability decomposition is accepted.

## Predictive stability

For every model, the following are calculated separately by fold and by
fold-month segment:

- mean cross-sectional Rank IC;
- median Rank IC;
- Rank IC standard deviation;
- Rank IC information ratio;
- positive-timestamp fraction;
- RMSE and MAE;
- row and timestamp counts.

A calendar month is assigned using the UTC prediction timestamp.

Month segments remain associated with their original fold and are never joined
across fold boundaries.

## Economic stability

The original portfolio construction is retained:

- 60-minute holding period;
- top four and bottom four contracts;
- zero net and unit gross exposure;
- next-bar-open entry;
- 12 staggered hourly cohorts;
- turnover resets across test-period discontinuities.

Fold and monthly economic metrics are calculated at:

- zero basis points;
- 0.5 basis points;
- 1 basis point.

For each temporal group, the evidence includes:

- pooled and equal-weighted mean return;
- annualized mean return;
- turnover;
- break-even transaction cost;
- worst and best cohort mean return;
- fraction of cohorts with positive mean return.

## Month-boundary treatment

A portfolio period is assigned to the month of its prediction timestamp.

Turnover is not reset merely because the calendar month changes. The monthly
decomposition therefore represents an ongoing strategy rather than a strategy
that liquidates and re-enters at every month boundary.

## Sharpe treatment

Monthly Sharpe ratios are not aggregated.

The 12 hourly sleeves overlap asynchronously in calendar time. Treating their
observations as independent would overstate effective sample size.

The monthly economic comparison therefore uses linear quantities: mean return,
turnover, cumulative return, and break-even cost.

## Interpretation

A robust signal should exhibit:

- positive Rank IC in all or most folds;
- positive Rank IC across most months;
- no isolated month dominating aggregate performance;
- positive gross break-even cost across most months;
- limited deterioration in the worst fold and worst month.

Passing this stage does not establish profitability. It identifies whether the
aggregate result is temporally broad or concentrated.
