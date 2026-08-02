# Label and execution-alignment audit

## Purpose

This audit reconstructs the research labels and portfolio execution returns
directly from the normalized source bars.

It is separate from the model evaluation so that a shared alignment error
cannot silently affect every model in the same way.

## Label reconstruction

For each symbol and prediction timestamp `t`, the raw label is reconstructed
as:

    log(close[t + 12] / close[t])

The recorded label-end timestamp must equal exactly:

    t + 12 five-minute bars

The cross-sectional target is reconstructed by subtracting the mean raw return
of all 20 symbols at each timestamp.

## Execution reconstruction

The economic simulation assumes that the final feature bar has already closed
before an order can be placed.

Execution return is therefore reconstructed as:

    log(close[t + 12] / open[t + 1])

The difference between this value and the close-to-close research label is
expected. It represents the move between `close[t]` and the executable
`open[t + 1]`.

## Split containment

For every walk-forward fold:

- training labels must end no later than the training boundary;
- validation labels must end no later than the validation boundary;
- test labels must end no later than the test boundary;
- validation timestamps must occur after the training boundary;
- test timestamps must occur after the validation boundary.

## Prediction alignment

Every frozen test prediction file is compared with the canonical research
panel.

The audit requires exact timestamp and symbol alignment and numerically
identical raw and cross-sectionally demeaned targets.

## Interpretation

Passing this audit establishes that the recorded labels, model predictions,
split boundaries, and economic execution convention are mutually aligned.

It does not establish profitability or remove the fixed-universe limitation.
