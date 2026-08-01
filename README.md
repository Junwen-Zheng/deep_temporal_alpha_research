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
