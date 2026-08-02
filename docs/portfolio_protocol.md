# Cost-aware portfolio protocol

## Purpose

Rank IC measures prediction ordering but does not establish whether a signal
survives portfolio formation, turnover, and transaction costs.

This stage compares every model on the same non-overlapping out-of-sample test
periods.

## Execution convention

A prediction is formed after bar `t` has completed.

The portfolio enters at the open of bar `t + 1` and exits at the close of bar
`t + 12`.

This avoids assuming execution at the same close used to construct the final
feature observation.

## Portfolio construction

At each rebalance timestamp:

- rank all 20 contracts by prediction;
- long the top four;
- short the bottom four;
- equal-weight each side;
- allocate 0.5 gross exposure to longs;
- allocate 0.5 gross exposure to shorts;
- maintain zero net exposure and unit gross exposure.

Ties are resolved deterministically by symbol order.

## Non-overlapping cohorts

The target horizon is 12 five-minute bars.

Evaluating a newly formed 60-minute portfolio every five minutes would create
overlapping holdings. The timestamps are therefore partitioned into 12
staggered cohorts.

Each cohort rebalances once per hour and contains non-overlapping holding
periods. Metrics are computed separately for every cohort and then summarized
across the 12 start-time phases.

This avoids presenting overlapping observations as independent portfolio
returns.

## Turnover and costs

Traded notional is:

    sum_i(abs(weight(i, t) - weight(i, t - 1)))

After a discontinuity between test periods, the portfolio is treated as
re-entering from cash.

Costs are charged per unit of one-way traded notional at:

- 0 basis points;
- 1 basis point;
- 2 basis points;
- 5 basis points;
- 10 basis points.

## Metrics

For every model, cost level, and cohort, the evidence records:

- period count;
- mean net return;
- annualized net return;
- annualized volatility;
- annualized Sharpe ratio;
- positive-return fraction;
- mean traded notional;
- cumulative net log return.

Annualization uses 8,760 hourly periods because Binance perpetual futures
trade continuously.

## Limitations

This remains a public-data research simulation. It does not model:

- exchange-specific fee tiers;
- bid-ask spread separately from the cost sensitivity;
- market impact;
- funding payments;
- liquidation or margin constraints;
- order-size capacity;
- exchange outages;
- live execution latency.

The output is therefore an economic sensitivity analysis, not a claim of
realisable trading performance.
