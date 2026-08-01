# Data contract

## Source

Binance public monthly USD-M futures kline archives.

## Sample

- Interval: 5 minutes
- Start: July 2025
- End: June 2026
- Candidate universe: 20 predeclared perpetual-futures contracts
- Expected archives: 240

The candidate universe is frozen before model fitting. It is not claimed to be
a fully point-in-time reconstruction of all historically tradable contracts.
That limitation must remain visible in the final report.

## Raw-data integrity

Every downloaded ZIP must:

1. have a corresponding publisher-provided CHECKSUM file;
2. match the published SHA-256 digest;
3. pass ZIP CRC validation;
4. contain exactly one CSV file;
5. be represented in the versioned download manifest.

## Raw-data immutability

Raw ZIP, CHECKSUM, and extracted CSV files must not be edited manually.
Subsequent transformations will write separate Parquet datasets.

## Timing

The raw kline schema contains both open time and close time.

A completed bar may only be used after its recorded close time. Later
milestones must preserve this distinction through an explicit
`available_time` field.
