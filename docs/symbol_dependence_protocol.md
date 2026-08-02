# Symbol-dependence protocol

## Purpose

A cross-sectional result can appear stable while being driven by one or two
contracts.

This stage measures how predictive and economic conclusions change when each
symbol is removed individually.

## Leave-one-symbol-out prediction analysis

For each model and each of the 20 symbols:

1. remove that symbol from every out-of-sample timestamp;
2. retain the remaining 19-symbol cross-section;
3. recompute timestamp-level Rank IC;
4. recompute aggregate Rank IC statistics;
5. compare the result with the full-universe metric.

Cross-sectional demeaning does not affect this Rank IC comparison because
subtracting a common value from every target in a timestamp does not change
target ranks.

## Leave-one-symbol-out portfolio analysis

For each exclusion:

- long the top four remaining predictions;
- short the bottom four remaining predictions;
- retain zero net and unit gross exposure;
- use next-bar-open entry and the 60-minute exit;
- retain all 12 staggered hourly cohorts;
- evaluate zero and one basis point transaction costs;
- calculate pooled and cohort-level break-even costs.

The portfolio is reconstructed after the symbol is removed. Existing weights
are not merely rescaled.

## Full-universe contribution decomposition

For the original 20-symbol portfolio, each model reports:

- long-selection frequency by symbol;
- short-selection frequency by symbol;
- active-selection frequency;
- average absolute portfolio weight;
- cumulative gross log-return contribution;
- absolute contribution share;
- contribution concentration rank.

Absolute contribution shares sum to one within each model.

The summary additionally reports:

- largest single-symbol absolute contribution share;
- top-three absolute contribution share;
- Herfindahl concentration of absolute contributions.

## Reproduction requirement

Before any exclusion result is accepted, the full-universe implementation must
reproduce:

- committed predictive metrics;
- committed portfolio cohort metrics;
- turnover;
- cost-adjusted returns;
- volatility and Sharpe.

## Interpretation

A result is symbol-robust when:

- Rank IC remains positive after every exclusion;
- no exclusion produces a large metric discontinuity;
- gross break-even cost remains positive after every exclusion;
- no single contract dominates absolute portfolio contribution.

Leave-one-out robustness does not eliminate fixed-universe survivorship bias.
It only tests dependence within the selected 20-contract universe.
