# Rebalance-frequency sensitivity protocol

## Purpose

The primary portfolio evaluation forms a new 60-minute portfolio every five
minutes and evaluates the 12 resulting hourly sleeves separately.

This stage tests whether the economic conclusion depends on how frequently the
signal is refreshed or on a particular clock phase.

## Fixed forecast and holding horizon

The prediction target and holding period remain fixed at 12 five-minute bars.

Changing the refresh cadence does not change the forecast horizon. This keeps
rebalance sensitivity separate from the later forecast-horizon analysis.

## Cadences

Four refresh cadences are evaluated:

| Refresh cadence | Concurrent hourly sleeves | Clock phases |
|---|---:|---:|
| 5 minutes | 12 | 1 |
| 15 minutes | 4 | 3 |
| 30 minutes | 2 | 6 |
| 60 minutes | 1 | 12 |

For cadence `s` bars and clock phase `p`, the selected hourly cohorts satisfy:

    cohort modulo s = p

Each active sleeve receives equal capital:

    capital per sleeve = 1 / sleeve count

## Reported quantities

For every model, cadence, clock phase, and transaction-cost level, the audit
reports:

- selected hourly cohorts;
- sleeve count;
- gross and net mean return;
- annualized mean return;
- pooled traded notional;
- break-even transaction cost;
- fraction of component sleeves with positive mean return.

Results are summarized across all possible clock phases using mean, median,
worst and best phase outcomes.

## Sharpe treatment

Sharpe ratios are deliberately not aggregated here.

The hourly sleeves begin at different five-minute offsets, so their returns are
asynchronous and overlapping in calendar time. Averaging their individual
Sharpes or treating launches as independent would overstate precision.

This stage therefore focuses on linear quantities that can be combined without
such an independence assumption: mean return, turnover and break-even cost.

## Interpretation

A signal is refresh-robust only when its result remains positive across
multiple cadences and clock phases at a plausible transaction-cost level.

A good result at one hourly phase but not the others indicates timing
dependence rather than a broadly stable executable signal.
