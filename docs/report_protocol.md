# Deterministic research-report protocol

## Purpose

The final research evidence must be reviewable without executing Python,
opening notebooks, or relying on external web assets.

This stage generates one self-contained HTML report from committed evidence
tables.

## Source policy

The report reads only committed project inputs:

- configuration files;
- protocol documentation;
- evidence CSV files;
- evidence manifests;
- the project README.

Prediction Parquet files and normalized market data are represented indirectly
through the cryptographic hashes already stored in their producing manifests.

## Determinism

The report deliberately excludes:

- generation timestamps;
- random identifiers;
- machine-specific absolute paths;
- external stylesheets;
- external scripts;
- external images;
- external fonts;
- network resources.

Source files are sorted lexicographically before inventory generation.

CSV outputs use a fixed newline convention and numeric representation. JSON
uses sorted keys and fixed indentation.

Running the report builder repeatedly with identical inputs must produce
identical SHA-256 hashes for:

- the HTML report;
- the key-findings table;
- the source inventory;
- the report manifest.

## Report structure

The report contains:

1. executive conclusion;
2. primary and exploratory findings;
3. research-design summary;
4. frozen predictive and economic evidence;
5. bootstrap uncertainty;
6. horizon sensitivity;
7. monthly and fold stability;
8. symbol dependence;
9. causal market regimes;
10. feature-family ablations;
11. limitations;
12. cryptographic source inventory.

## Primary and exploratory separation

Frozen model results are labelled primary.

Feature-family ablations are labelled exploratory because multiple reduced
feature sets were inspected using the same frozen test sample.

A reduced-feature result may define a future hypothesis but cannot revise the
primary conclusion without a new untouched evaluation period.

## Interpretation boundary

The report must not describe public-OHLCV simulation as live, executable, or
realized profitability.

Break-even cost remains a simplified one-way threshold excluding impact,
funding, latency, adverse selection, partial fills, and operational risk.
