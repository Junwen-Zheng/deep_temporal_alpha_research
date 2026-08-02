# Model-family and feature-family ablation protocol

## Purpose

The project contains several model families and sixteen causal input features.

This stage separates two questions:

1. whether increasingly complex model families improve upon pointwise and
   non-learned references;
2. whether deterministic learned baselines depend disproportionately on one
   feature family.

## Feature families

The sixteen features are partitioned into four non-overlapping families.

### Returns

- one-bar return;
- three-bar return;
- twelve-bar return;
- forty-eight-bar return.

### Range and volatility

- intrabar log range;
- close-to-open return;
- twelve-bar realized volatility;
- forty-eight-bar realized volatility.

### Activity and flow

- log quote volume;
- forty-eight-bar volume z-score;
- forty-eight-bar trade-count z-score;
- taker-buy ratio.

### Calendar

- time-of-day sine and cosine;
- weekday sine and cosine.

Every feature belongs to exactly one family.

## Leave-one-family-out retraining

Ridge and LightGBM are evaluated using:

- all sixteen features;
- all features except returns;
- all features except range and volatility;
- all features except activity and flow;
- all features except calendar variables.

Each reduced model is retrained independently for every walk-forward fold.

Ridge retains:

- train-only robust scaling;
- validation-only alpha selection;
- the original alpha grid.

LightGBM retains:

- the original deterministic seed;
- the original hyperparameters;
- validation early stopping;
- raw unscaled features.

The test split never participates in feature-set or hyperparameter selection.

## Full-feature control

The full-feature rows use the already committed frozen Ridge and LightGBM
predictions.

Those rows must exactly reproduce the committed predictive and portfolio
evidence before ablation deltas are accepted.

## Economic evaluation

Every feature variant uses the unchanged primary portfolio protocol:

- top four and bottom four contracts;
- zero net and unit gross exposure;
- next-bar-open entry;
- sixty-minute exit;
- twelve non-overlapping hourly cohorts;
- discontinuity-aware turnover;
- zero, one, two, five, and ten basis-point cost scenarios.

## Model-family comparison

The seven frozen models are compared without retraining:

- momentum reference;
- reversal reference;
- Ridge;
- LightGBM;
- pointwise MLP;
- temporal convolutional network;
- patch Transformer.

The comparison includes:

- aggregate Rank IC;
- block-bootstrap Rank IC interval;
- break-even transaction cost;
- block-bootstrap break-even interval;
- deltas relative to reversal;
- deltas relative to the pointwise MLP.

This is an architecture-family comparison, not a claim that the models are
strictly nested statistical specifications.

## Interpretation

A large negative leave-one-family-out delta indicates dependence on that
feature family.

A positive delta after removing a family indicates that the removed variables
were unhelpful under the frozen training procedure.

An architecture with higher Rank IC but lower break-even cost has improved
ranking accuracy without improving economic efficiency.

All results remain subject to the fixed-universe and public-OHLCV limitations
documented elsewhere in the project.
