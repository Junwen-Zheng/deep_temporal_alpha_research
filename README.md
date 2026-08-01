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
