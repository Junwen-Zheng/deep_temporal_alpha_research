# Deep Temporal Alpha Research

A reproducible deep-learning quantitative research project testing whether
temporal neural models add stable out-of-sample predictive information beyond
strong linear and tree-based baselines.

The project will include:

- verified public-market data ingestion;
- point-in-time features and labels;
- purged walk-forward validation;
- Ridge and LightGBM baselines;
- MLP, TCN, and Transformer comparisons;
- null tests and transaction-cost diagnostics;
- reproducible evidence and a self-contained research report.

## Market data

The initial dataset consists of verified monthly Binance USD-M futures
5-minute kline archives from July 2025 through June 2026. Every raw archive
is validated against its publisher-provided SHA-256 checksum and represented
in a versioned manifest.

## Normalized data

Verified CSV archives are normalized into typed, compressed Parquet files.
The pipeline preserves separate event and availability timestamps and records
row counts, gaps, file hashes, and output locations in a versioned manifest.

## Research panel

The causal feature pipeline constructs lagged return, volatility, activity,
order-flow, and cyclical-time inputs from completed bars. Labels represent
the following 60-minute cross-sectionally demeaned log return.

Every row records its label end time. Walk-forward splits remove observations
whose forward target crosses a train, validation, or test boundary.

## Research baselines

The baseline layer compares 12-bar momentum and reversal scores with
train-only-scaled Ridge regression and deterministic LightGBM. Ridge
regularization is selected on validation Rank IC, while all test predictions
remain untouched until selection is complete.

## Neural baseline

A compact pointwise MLP tests nonlinear interactions among the causal feature
set before sequence architectures are introduced. Training uses fold-specific
robust scaling, validation Rank-IC checkpoint selection, early stopping, and
local MLflow experiment tracking.

## Temporal sequence representation

The TCN and Transformer share a memory-efficient 128-bar dataset. The complete
panel is stored once as a timestamp-by-symbol-by-feature tensor, and overlapping
windows are sliced on demand. A dedicated audit verifies temporal continuity,
split alignment, complete cross-sections, and exact history exclusions before
any sequence model is trained.

## Temporal convolution baseline

The causal TCN consumes the full 128-bar sequence through a dilation schedule
whose receptive field is exactly 128 bars. Training deterministically
subsamples highly overlapping endpoints, while validation and test predictions
continue to cover every eligible five-minute timestamp.

## Causal patch Transformer

The Transformer consumes the full 128-bar history as 32 causal four-bar
patches. Masked self-attention prevents future-patch access, while patching
reduces quadratic attention cost enough to evaluate every validation and test
endpoint.

## Cost-aware portfolio evaluation

All models are compared using next-bar execution and 12 staggered,
non-overlapping hourly cohorts. Each portfolio is dollar-neutral and
unit-gross, holding the top four and bottom four contracts. Results include
turnover and transaction-cost sensitivity from zero to ten basis points.

## Null and alignment tests

Every frozen out-of-sample prediction series is evaluated against deterministic
within-timestamp target permutations and multiple temporal target shifts. The
audit also verifies exact zero-shift metric reproduction and the numerical
identity between the momentum and reversal reference scores.

## Label and execution audit

All forward labels are reconstructed directly from normalized bars, including
their exact 12-bar end timestamps and cross-sectional demeaning. The audit also
reconstructs next-bar-open execution returns, verifies split containment, and
checks every frozen prediction file against the canonical research panel.

## Break-even transaction costs

The economic analysis computes exact cohort-level, equal-weighted, and pooled
break-even transaction costs. A dense 0–10 basis-point curve shows how quickly
mean returns deteriorate with traded notional without interpolating nonlinear
Sharpe-ratio behavior.

## Rebalance-frequency sensitivity

The fixed 60-minute forecast is evaluated with signal refresh every 5, 15, 30,
and 60 minutes. Every clock phase is included through equal-capital staggered
hourly sleeves. The analysis reports phase-level mean return, turnover and
break-even cost without treating overlapping launches as independent Sharpe
observations.

## Portfolio-breadth sensitivity

Dollar-neutral portfolios are evaluated using between one and ten contracts on
each side of the cross-section. Every breadth retains next-bar execution, the
60-minute holding period, staggered non-overlapping cohorts, turnover
accounting, and cost sensitivity. The four-per-side configuration is required
to reproduce the primary portfolio evidence exactly.

## Forecast and holding-horizon sensitivity

Frozen out-of-sample predictions are evaluated against 15-, 30-, 60-, and
120-minute forward returns. Each horizon reconstructs its labels and
next-bar-open execution returns directly from normalized bars, preserves test
fold boundaries, and uses horizon-matched non-overlapping cohorts. The
60-minute configuration must reproduce the primary predictive and portfolio
evidence exactly.

## Fold and monthly stability

Frozen 60-minute predictions are decomposed by walk-forward fold and UTC
calendar month. Predictive Rank IC, portfolio mean return, turnover, and
break-even cost are reported for every temporal segment. Aggregate predictive
and portfolio evidence must be reproduced before the decomposition is
accepted, and overlapping monthly sleeves are not treated as independent
Sharpe observations.

## Symbol-dependence analysis

Every contract is removed individually from the frozen out-of-sample
evaluation. Rank IC and portfolio economics are recomputed on each remaining
19-symbol universe. The full 20-symbol portfolio is also decomposed into
per-symbol selection frequency and gross-return contribution, with
single-symbol, top-three, and Herfindahl concentration measures.

## Market-regime analysis

Frozen out-of-sample predictions are decomposed by trailing market direction,
market-wide realized volatility, and cross-sectional return dispersion.
Fold-specific regime thresholds use only timestamps at or before the validation
boundary. Predictive Rank IC, turnover, break-even cost, and cost-adjusted mean
return are reported for every fold-regime cell.

## Block-bootstrap uncertainty

Timestamp-level Rank IC is evaluated with circular blocks from 60 to 240
minutes. Portfolio gross return and turnover are resampled jointly within each
non-overlapping hourly cohort using blocks from one day to one week. The
resulting deterministic percentile intervals cover Rank IC, gross return,
one-basis-point net return, turnover, and break-even transaction cost.
