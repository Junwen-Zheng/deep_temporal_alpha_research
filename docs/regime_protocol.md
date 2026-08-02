# Market-regime analysis protocol

## Purpose

Aggregate out-of-sample results can conceal environments in which a signal
weakens, reverses, or becomes economically unusable.

This stage decomposes the frozen 60-minute evaluation across causal market
states.

## Market-state variables

Three variables are calculated at every five-minute timestamp using only bars
that have closed by that timestamp.

### Market direction

For each contract:

    trailing return
    = log(close[t] / close[t - 12])

The market return is the cross-sectional mean of these trailing 60-minute
returns.

### Market-wide realized volatility

For each contract, trailing realized volatility is:

    sqrt(sum of squared five-minute returns
         over the previous 12 bars))

The market-wide value is the cross-sectional mean across all 20 contracts.

### Cross-sectional dispersion

Dispersion is the population standard deviation across the 20 trailing
60-minute contract returns.

## Fold-specific thresholds

Each regime dimension uses two tertile thresholds.

For a test fold, thresholds are estimated using only market-state timestamps at
or before that fold's validation boundary.

No test timestamp contributes to its own threshold definition.

The regimes are:

| Dimension | Regimes |
|---|---|
| Direction | down, flat, up |
| Volatility | low, medium, high |
| Dispersion | low, medium, high |

## Predictive evaluation

For every model, fold, regime dimension, and regime, the analysis reports:

- mean and median Rank IC;
- Rank IC standard deviation and information ratio;
- positive-timestamp fraction;
- RMSE and MAE;
- row and timestamp counts.

No model is retrained.

## Economic evaluation

The original portfolio implementation is retained:

- top four and bottom four contracts;
- zero net and unit gross exposure;
- next-bar-open entry;
- 60-minute exit;
- 12 staggered hourly cohorts;
- discontinuity-aware turnover.

Portfolio metrics are calculated at:

- zero basis points;
- 0.5 basis points;
- 1 basis point.

Each regime cell reports:

- pooled and equal-weighted mean return;
- annualized mean return;
- turnover;
- break-even cost;
- worst and best cohort return;
- positive-cohort fraction.

## Reproduction requirement

Before the regime decomposition is accepted, the implementation must reproduce:

- aggregate committed out-of-sample prediction metrics;
- aggregate committed portfolio cohort metrics;
- turnover;
- transaction-cost results;
- volatility and Sharpe.

## Interpretation

A regime-robust signal should remain positive across most folds and market
states.

A negative result isolated to one state may indicate conditional signal decay.
A result that is positive only in one state is not broadly stable.

Regime thresholds are descriptive. They are not used to select, retrain, or
trade the models.
