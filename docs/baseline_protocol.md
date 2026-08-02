# Baseline protocol

## Purpose

The baseline stage establishes whether later temporal neural models add
incremental predictive information beyond inexpensive reference models.

## Models

### Twelve-bar momentum

Uses the lagged 60-minute log return directly as the prediction score.

### Twelve-bar reversal

Uses the negative lagged 60-minute log return.

### Ridge regression

Uses all 16 features after robust scaling fitted exclusively on each fold's
training period.

The regularization parameter is selected using validation mean
cross-sectional Rank IC. The test period is not inspected during selection.

### LightGBM

Uses the unscaled 16-feature vector. Training is deterministic and uses
validation-loss early stopping. Its test predictions are produced once using
the selected boosting iteration.

## Evaluation

Each model is evaluated separately on validation and test periods using:

- mean timestamp-level cross-sectional Spearman Rank IC;
- median Rank IC;
- Rank IC standard deviation and information ratio;
- positive-IC timestamp fraction;
- RMSE;
- MAE.

No claim is made that the baseline scores constitute deployable alpha.
Transaction costs and portfolio diagnostics are added in a later milestone.
