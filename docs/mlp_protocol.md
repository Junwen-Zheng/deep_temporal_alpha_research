# Pointwise MLP protocol

## Purpose

The pointwise MLP determines whether nonlinear interactions among the 16
causal bar features improve on Ridge and LightGBM before temporal sequence
models are introduced.

It is deliberately not presented as a temporal architecture. The later TCN
and Transformer experiments consume 128-bar sequences.

## Inputs

The MLP consumes the current row's 16 causal features.

Robust scaling is fitted exclusively on each fold's purged training period.
Validation and test rows use the frozen training scaler.

## Architecture

- input layer normalization;
- two GELU hidden layers;
- dropout;
- scalar return-score output.

## Selection

Training minimizes mean squared error.

The selected checkpoint is the epoch with the highest validation
cross-sectional mean Rank IC. Test predictions are generated only after the
checkpoint has been selected.

## Reproducibility

Each fold records:

- random seed;
- architecture and optimizer parameters;
- train-only scaler;
- selected epoch;
- MLflow run ID;
- checkpoint hash;
- prediction hashes;
- evidence-file hashes.

MPS execution is seed-controlled but is not claimed to be byte-for-byte
deterministic across hardware or PyTorch versions.

## MLflow storage

Experiment and run metadata are stored in a local SQLite database. MLflow
artifacts are stored separately under the ignored `mlartifacts` directory.
The database and generated artifacts are excluded from Git; reproducible
run identifiers and file hashes are retained in the committed evidence
manifest.
