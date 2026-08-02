# Causal patch Transformer protocol

## Purpose

The Transformer is the second temporal architecture and the final primary
model family in the predefined comparison.

It tests whether causal attention over the complete 128-bar history improves
on the TCN, pointwise MLP, LightGBM, Ridge, and reversal benchmark.

## Patch representation

The model consumes all 128 five-minute bars.

Four adjacent bars are flattened into each patch, producing 32 temporal
tokens:

    128 bars / 4 bars per patch = 32 tokens

Patching reduces quadratic attention cost without discarding the input
history.

## Architecture

The model uses:

- linear patch projection;
- learned positional embeddings;
- two pre-normalized Transformer encoder layers;
- four attention heads;
- model dimension 64;
- feedforward dimension 128;
- GELU activation;
- dropout;
- final LayerNorm;
- scalar endpoint prediction head.

## Causality

An upper-triangular attention mask prevents token `t` from attending to any
later patch.

The final prediction uses only the last patch representation, which may attend
to the current and all preceding patches.

## Scaling and splits

The robust scaler is fitted exclusively on each fold's purged training rows.

Validation and test sequences may include earlier feature history that was
already observable at prediction time. Their forward targets remain fully
contained inside their assigned split.

## Endpoint sampling

Training uses every sixth eligible endpoint to limit duplication among heavily
overlapping 128-bar windows.

Validation and test evaluation retain every eligible five-minute endpoint.

## Selection and reproducibility

Training minimizes mean squared error with AdamW and gradient clipping.

The selected epoch maximizes validation mean cross-sectional Rank IC. Test
predictions are generated after checkpoint selection.

The evidence manifest records architecture parameters, train-only scalers,
row counts, selected epochs, MLflow run IDs, checkpoint hashes, prediction
hashes, and evidence hashes.
