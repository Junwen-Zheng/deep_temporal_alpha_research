# Out-of-sample null-test protocol

## Purpose

Positive Rank IC can result from genuine predictive ordering, accidental
alignment, metric implementation errors, or leakage.

This stage applies deterministic falsification tests to all seven models using
only their frozen out-of-sample predictions.

## Cross-sectional target permutation

For each test timestamp, the 20 target ranks are independently permuted across
symbols.

This preserves:

- the timestamp population;
- the target distribution;
- the number of contracts;
- the model predictions;
- the cross-sectional rank values.

It destroys the model-to-contract target alignment.

Fifty deterministic permutations are evaluated. The observed Rank IC is
compared with the resulting null distribution using a corrected empirical
two-sided p-value.

These p-values are diagnostics rather than definitive inference because
adjacent five-minute observations and 60-minute targets remain serially
dependent.

## Temporal alignment shifts

Predictions are compared with targets shifted by:

- minus 24, 12, 6 and 1 bars;
- zero bars;
- plus 1, 6, 12 and 24 bars.

Positive shift `k` compares the prediction at `t` with the target recorded at
`t + k`.

Shifts are performed independently inside each test fold so that observations
cannot cross test-period boundaries.

The zero-shift result must reproduce the committed out-of-sample Rank IC.

## Momentum/reversal identity

The 12-bar momentum and reversal predictions must be exact numerical
opposites at every timestamp and symbol.

Their observed Rank IC values must therefore have equal magnitude and opposite
sign.

## Interpretation

A model passes this falsification stage when its observed result is clearly
distinguishable from the permutation-null distribution.

Passing does not establish profitability or causal economic value. It only
shows that the measured ordering is not reproduced after target identity is
destroyed.

## Undefined cross-sectional correlations

Spearman Rank IC is undefined when either the prediction ranks or target ranks
have zero variance across all contracts at a timestamp.

Such timestamps are retained in the underlying prediction evidence but are
excluded from Rank IC means and timestamp counts. They are not assigned a
correlation of zero. This matches the treatment used by the primary
out-of-sample metric evaluator.
