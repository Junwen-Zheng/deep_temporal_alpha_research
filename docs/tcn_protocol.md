# Temporal convolutional network protocol

## Purpose

The TCN is the first model that consumes the complete 128-bar temporal input.

Its purpose is to test whether causal nonlinear temporal structure adds
out-of-sample information beyond the pointwise MLP, Ridge, LightGBM, and the
12-bar reversal reference score.

## Architecture

The network uses:

- a pointwise input projection;
- seven causal residual convolution blocks;
- dilations of 1, 2, 4, 8, 16, 32, and 64;
- kernel size 2;
- per-timestamp channel LayerNorm;
- GELU activation;
- dropout;
- a scalar endpoint prediction head.

The receptive field is exactly 128 bars:

    1 + (2 - 1) * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 128

No convolution reads a timestamp after the output timestamp.

## Scaling

The robust feature scaler is fitted using only each fold's purged training
period. The same frozen scaler is applied to historical context, validation,
and test sequences.

## Overlapping-window sampling

Adjacent five-minute endpoints share 127 of their 128 input bars and most of
their 60-minute target horizon.

Training therefore uses every sixth eligible endpoint, corresponding to one
endpoint every 30 minutes. All 20 symbols are retained for each selected
timestamp.

Validation and test predictions remain evaluated at every eligible five-minute
endpoint. Training subsampling therefore changes optimization cost, not the
reported evaluation population.

## Training and model selection

Training minimizes mean squared error with AdamW and gradient clipping.

The selected checkpoint is the epoch with the highest validation mean
cross-sectional Rank IC. Test predictions are generated only after selection.

Automatic mixed precision is enabled only on CUDA. MPS and CPU execution use
float32.

## Reproducibility evidence

Each fold records:

- training and evaluation row counts;
- endpoint sampling stride;
- train-only scaler;
- architecture and optimizer configuration;
- selected epoch;
- validation selection score;
- MLflow run ID;
- checkpoint hash;
- prediction hashes;
- evidence hashes.
