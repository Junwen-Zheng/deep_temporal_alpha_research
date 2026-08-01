# Research contract

## Research question

Do compact temporal deep-learning models add stable out-of-sample
cross-sectional return-prediction information beyond Ridge and LightGBM?

## Market and sample

- Binance USD-M perpetual futures
- 20 predeclared contracts
- 5-minute completed bars
- July 2025 through June 2026
- 128-bar input sequences
- 60-minute prediction horizon

The fixed universe is a public-data research convenience. It is not claimed
to be a fully point-in-time reconstruction of all historically eligible
contracts.

## Prediction time

Features for bar `t` become available only at that bar's recorded
`available_time`.

No feature may use a bar after `t`.

## Target

For instrument `i` at timestamp `t`:

    raw_target(i, t) = log(close(i, t + 12) / close(i, t))

The training label is the timestamp-level cross-sectionally demeaned return:

    target(i, t) =
        raw_target(i, t)
        - mean_j(raw_target(j, t))

The dataset records both `label_end_timestamp` and `label_end_time`.

## Feature families

- lagged log returns;
- intrabar range and close-to-open return;
- rolling realized volatility;
- quote-volume level and surprise;
- trade-count surprise;
- taker-buy ratio;
- cyclical time-of-day and weekday encodings.

For a completed bar with zero quote volume, `taker_buy_ratio` is assigned
the neutral value `0.5`. The zero-activity state remains visible through
the volume and trade-count features, while the timestamp is retained
rather than silently removed.

Feature normalization must be fitted separately on each fold's training
period. This milestone does not apply full-sample scaling.

## Walk-forward design

Three expanding folds are defined.

A row belongs to a split only when its full forward label ends inside that
split. This removes observations whose target window crosses a train,
validation, or test boundary.

Test windows do not overlap:

- Fold 1 test: January-February 2026
- Fold 2 test: March-April 2026
- Fold 3 test: May-June 2026

## Primary metric

Mean timestamp-level cross-sectional Spearman Rank IC.

## Secondary diagnostics

- median Rank IC;
- Rank IC information ratio;
- positive-IC timestamp fraction;
- top-minus-bottom portfolio return;
- transaction-cost sensitivity;
- turnover;
- symbol and subperiod concentration;
- seed stability;
- null-test results.

## Interpretation limit

This is a public-data research exercise. Positive results will not be
described as deployable alpha or evidence of expected live profitability.
