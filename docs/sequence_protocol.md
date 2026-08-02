# Sequence dataset protocol

## Purpose

The sequence layer provides the shared input representation for the temporal
convolutional network and Transformer experiments.

## Representation

The complete research panel is validated and reshaped into:

    timestamp x symbol x feature

The underlying feature tensor is stored once. Individual 128-bar windows are
constructed by slicing this tensor when a sample is requested, rather than
materializing millions of overlapping copies.

## Sequence endpoint

A sample ending at timestamp `t` contains:

    t - 127, ..., t - 1, t

No timestamp after `t` is included.

The target remains the cross-sectionally demeaned return from `t` through
`t + 12` bars.

## Split handling

Train, validation, and test membership is determined by the existing purged
walk-forward masks.

Validation and test sequences may use historical feature rows preceding their
split boundary. This is permitted because those observations were already
available at prediction time.

Targets remain fully contained inside their assigned split.

## Initial history exclusion

The first 127 research-panel timestamps cannot form a complete 128-bar
sequence and are excluded from sequence training.

This removes 2,540 rows from each expanding training fold:

    127 timestamps x 20 symbols

Validation and test counts are unaffected because they occur after sufficient
historical context exists.

## Leakage audit

The audit verifies:

- a complete 20-symbol cross-section at every timestamp;
- identical symbol ordering across timestamps;
- contiguous five-minute timestamps;
- finite feature and target tensors;
- cross-sectionally uniform split masks;
- strictly valid sequence endpoints;
- exact fold and split sample counts.
