# Forecast and holding-horizon sensitivity protocol

## Purpose

The trained models optimize a 60-minute cross-sectional return target.

This stage tests how their frozen out-of-sample forecasts relate to shorter and
longer forward-return horizons.

The models are not retrained. The analysis is therefore a signal-decay and
holding-horizon audit, not a comparison of separately optimized models.

## Horizons

The following close-to-close targets are reconstructed directly from the
normalized bars:

| Bars | Minutes |
|---:|---:|
| 3 | 15 |
| 6 | 30 |
| 12 | 60 |
| 24 | 120 |

For horizon `h`, the raw target is:

    log(close[t + h] / close[t])

The cross-sectional target subtracts the mean raw return of all 20 symbols at
the same timestamp.

## Execution convention

Economic returns use:

    entry = open[t + 1]

    exit = close[t + h]

Therefore:

    execution return
    = log(close[t + h] / open[t + 1])

The next-bar entry prevents execution at the close used to construct the final
feature observation.

## Fold containment

Alternate targets are retained only when their label-end timestamp remains
inside the original test fold.

The 120-minute horizon consequently drops prediction timestamps near the end of
each fold when the longer target would cross the fold boundary.

No target may borrow information from the following validation or test period.

## Portfolio construction

For every horizon:

- long the top four predictions;
- short the bottom four predictions;
- allocate 0.5 gross exposure to each side;
- maintain zero net and unit gross exposure;
- use `h` staggered non-overlapping cohorts;
- reset turnover after test-period discontinuities.

Transaction-cost scenarios are:

- 0 basis points;
- 0.25 basis points;
- 0.5 basis points;
- 1 basis point;
- 2 basis points;
- 5 basis points;
- 10 basis points.

## Annualization

Each cohort holds a non-overlapping position for the relevant horizon.

Annualization therefore uses:

    periods per year
    = 365 * 24 * 60
      / horizon minutes

This gives 35,040 periods for 15 minutes, 17,520 for 30 minutes, 8,760 for 60
minutes, and 4,380 for 120 minutes.

## Reproduction requirement

The 60-minute horizon must exactly reproduce:

- the committed out-of-sample Rank IC metrics;
- all committed portfolio cohort results;
- turnover;
- cost-adjusted returns;
- annualized volatility and Sharpe.

## Interpretation

A model whose Rank IC decays smoothly across nearby horizons may be capturing a
persistent ordering effect.

A result that exists only at one horizon is more timing-specific.

A longer horizon may improve break-even cost by spreading turnover over a
larger expected move, but this does not establish executable profitability.
