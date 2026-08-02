from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_alpha.data.download import sha256_file

REFERENCE_MODELS = {
    "momentum_12",
    "reversal_12",
}


def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    table_name: str,
) -> None:
    missing = set(columns) - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"{table_name} is missing columns: "
            f"{sorted(missing)}"
        )


def format_value(
    value: Any,
    precision: int = 6,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return "Yes" if bool(value) else "No"

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        return f"{int(value):,}"

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        numeric = float(value)

        if not np.isfinite(numeric):
            return ""

        return f"{numeric:.{precision}f}"

    if pd.isna(value):
        return ""

    return str(value)


def render_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    labels: Mapping[str, str] | None = None,
    precision: int = 6,
    table_class: str = "data-table",
) -> str:
    require_columns(
        frame=frame,
        columns=columns,
        table_name="Rendered table",
    )

    column_labels = {
        column: (
            labels[column]
            if labels
            and column in labels
            else column.replace(
                "_",
                " ",
            ).title()
        )
        for column in columns
    }

    header = "".join(
        "<th>"
        + escape(
            column_labels[column]
        )
        + "</th>"
        for column in columns
    )

    body_rows = []

    for row in frame[
        list(columns)
    ].itertuples(
        index=False,
        name=None,
    ):
        cells = "".join(
            "<td>"
            + escape(
                format_value(
                    value,
                    precision=precision,
                )
            )
            + "</td>"
            for value in row
        )

        body_rows.append(
            f"<tr>{cells}</tr>"
        )

    body = "".join(body_rows)

    return (
        f'<table class="{escape(table_class)}">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def render_symmetric_bar_chart(
    frame: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    value_suffix: str = "",
    precision: int = 4,
) -> str:
    require_columns(
        frame=frame,
        columns=[
            label_column,
            value_column,
        ],
        table_name="Chart frame",
    )

    chart = frame[
        [
            label_column,
            value_column,
        ]
    ].copy()

    values = chart[
        value_column
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Chart values contain "
            "non-finite values"
        )

    row_height = 34
    label_width = 150
    plot_width = 520
    value_width = 120
    chart_width = (
        label_width
        + plot_width
        + value_width
    )
    chart_height = (
        52
        + row_height
        * len(chart)
    )
    zero_x = (
        label_width
        + plot_width / 2.0
    )

    maximum_absolute = float(
        np.max(
            np.abs(values)
        )
    )

    if maximum_absolute <= 0:
        maximum_absolute = 1.0

    scale = (
        plot_width / 2.0
        / maximum_absolute
    )

    elements = [
        (
            f'<svg class="bar-chart" '
            f'viewBox="0 0 {chart_width} '
            f'{chart_height}" '
            f'role="img" '
            f'aria-label="{escape(title)}">'
        ),
        (
            f'<text x="0" y="22" '
            f'class="chart-title">'
            f"{escape(title)}</text>"
        ),
        (
            f'<line x1="{zero_x:.3f}" '
            f'y1="38" '
            f'x2="{zero_x:.3f}" '
            f'y2="{chart_height - 8}" '
            f'class="zero-line"/>'
        ),
    ]

    for index, row in enumerate(
        chart.itertuples(
            index=False,
            name=None,
        )
    ):
        label, raw_value = row
        value = float(raw_value)
        y = 48 + index * row_height
        bar_width = abs(value) * scale

        if value >= 0:
            x = zero_x
            bar_class = "positive-bar"
        else:
            x = zero_x - bar_width
            bar_class = "negative-bar"

        elements.extend(
            [
                (
                    f'<text x="0" '
                    f'y="{y + 17}" '
                    f'class="chart-label">'
                    f"{escape(str(label))}"
                    "</text>"
                ),
                (
                    f'<rect x="{x:.3f}" '
                    f'y="{y}" '
                    f'width="{bar_width:.3f}" '
                    f'height="20" '
                    f'class="{bar_class}"/>'
                ),
                (
                    f'<text '
                    f'x="{label_width + plot_width + 12}" '
                    f'y="{y + 16}" '
                    f'class="chart-value">'
                    f"{value:.{precision}f}"
                    f"{escape(value_suffix)}"
                    "</text>"
                ),
            ]
        )

    elements.append("</svg>")

    return "".join(elements)


def build_source_inventory(
    paths: Iterable[Path],
) -> pd.DataFrame:
    unique_paths = sorted(
        {
            Path(path)
            for path in paths
        },
        key=lambda path: path.as_posix(),
    )

    rows = []

    for path in unique_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        rows.append(
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    if not rows:
        raise ValueError(
            "Source inventory is empty"
        )

    return pd.DataFrame(rows)


def build_key_findings(
    tables: Mapping[
        str,
        pd.DataFrame,
    ],
    focus_cost_bps: float,
) -> pd.DataFrame:
    required_tables = {
        "oos",
        "break_even",
        "bootstrap_prediction",
        "bootstrap_portfolio",
        "ablation_features",
        "ablation_families",
    }

    missing_tables = (
        required_tables
        - set(tables)
    )

    if missing_tables:
        raise ValueError(
            "Missing finding tables: "
            f"{sorted(missing_tables)}"
        )

    oos = tables["oos"]
    break_even = tables["break_even"]
    bootstrap_prediction = tables[
        "bootstrap_prediction"
    ]
    bootstrap_portfolio = tables[
        "bootstrap_portfolio"
    ]
    ablation_features = tables[
        "ablation_features"
    ]
    ablation_families = tables[
        "ablation_families"
    ]

    require_columns(
        oos,
        [
            "model",
            "mean_rank_ic",
        ],
        "OOS metrics",
    )

    require_columns(
        break_even,
        [
            "model",
            "pooled_break_even_cost_bps",
        ],
        "Break-even summary",
    )

    require_columns(
        bootstrap_prediction,
        [
            "model",
            "block_length_timestamps",
            "confidence_lower",
            "confidence_upper",
        ],
        "Prediction bootstrap",
    )

    require_columns(
        bootstrap_portfolio,
        [
            "model",
            "block_length_periods",
            "probability_break_even_above_focus_cost",
            "break_even_confidence_upper",
        ],
        "Portfolio bootstrap",
    )

    require_columns(
        ablation_features,
        [
            "model",
            "variant",
            "mean_rank_ic",
            "pooled_break_even_cost_bps",
            (
                "pooled_annualized_net_return_"
                "at_focus_cost"
            ),
        ],
        "Feature ablations",
    )

    require_columns(
        ablation_families,
        [
            "model",
            "mean_rank_ic",
            "model_family",
        ],
        "Model-family summary",
    )

    best_rank = oos.loc[
        oos["mean_rank_ic"].idxmax()
    ]

    learned = (
        ablation_families.loc[
            ~ablation_families[
                "model"
            ].isin(REFERENCE_MODELS)
        ]
    )

    best_learned = learned.loc[
        learned["mean_rank_ic"].idxmax()
    ]

    best_break_even = break_even.loc[
        break_even[
            "pooled_break_even_cost_bps"
        ].idxmax()
    ]

    primary_prediction = (
        bootstrap_prediction.loc[
            bootstrap_prediction[
                "block_length_timestamps"
            ]
            == 12
        ]
    )

    positive_interval_models = (
        primary_prediction.loc[
            primary_prediction[
                "confidence_lower"
            ]
            > 0,
            "model",
        ]
        .sort_values()
        .tolist()
    )

    primary_portfolio = (
        bootstrap_portfolio.loc[
            bootstrap_portfolio[
                "block_length_periods"
            ]
            == 24
        ]
    )

    strongest_survival = (
        primary_portfolio.loc[
            primary_portfolio[
                (
                    "probability_break_even_"
                    "above_focus_cost"
                )
            ].idxmax()
        ]
    )

    reduced = (
        ablation_features.loc[
            ablation_features[
                "variant"
            ]
            != "full"
        ]
    )

    best_reduced = reduced.loc[
        reduced[
            "pooled_break_even_cost_bps"
        ].idxmax()
    ]

    rows = [
        {
            "category": "predictive",
            "finding": (
                "Highest frozen out-of-sample "
                "Rank IC"
            ),
            "model": str(
                best_rank["model"]
            ),
            "variant": "",
            "value": float(
                best_rank[
                    "mean_rank_ic"
                ]
            ),
            "unit": "Rank IC",
            "status": "primary",
            "interpretation": (
                "The non-learned reversal "
                "reference remains the strongest "
                "predictive benchmark."
            ),
        },
        {
            "category": "model_family",
            "finding": (
                "Highest learned-model "
                "out-of-sample Rank IC"
            ),
            "model": str(
                best_learned["model"]
            ),
            "variant": str(
                best_learned[
                    "model_family"
                ]
            ),
            "value": float(
                best_learned[
                    "mean_rank_ic"
                ]
            ),
            "unit": "Rank IC",
            "status": "primary",
            "interpretation": (
                "Temporal attention improves "
                "upon the pointwise MLP but does "
                "not surpass reversal."
            ),
        },
        {
            "category": "economics",
            "finding": (
                "Highest frozen pooled "
                "break-even cost"
            ),
            "model": str(
                best_break_even["model"]
            ),
            "variant": "",
            "value": float(
                best_break_even[
                    "pooled_break_even_cost_bps"
                ]
            ),
            "unit": "bps",
            "status": "primary",
            "interpretation": (
                "The strongest frozen result "
                f"remains below the {focus_cost_bps:g} "
                "bps focus cost."
            ),
        },
        {
            "category": "uncertainty",
            "finding": (
                "Models with primary Rank IC "
                "interval above zero"
            ),
            "model": ",".join(
                positive_interval_models
            ),
            "variant": (
                "60-minute circular blocks"
            ),
            "value": float(
                len(
                    positive_interval_models
                )
            ),
            "unit": "models",
            "status": "primary",
            "interpretation": (
                "Predictive evidence is broader "
                "than economic evidence."
            ),
        },
        {
            "category": "uncertainty",
            "finding": (
                "Highest bootstrap frequency "
                "of break-even above focus cost"
            ),
            "model": str(
                strongest_survival["model"]
            ),
            "variant": (
                "24-hour cohort blocks"
            ),
            "value": float(
                strongest_survival[
                    (
                        "probability_break_even_"
                        "above_focus_cost"
                    )
                ]
            ),
            "unit": "frequency",
            "status": "primary",
            "interpretation": (
                "The empirical frequency remains "
                "too small to support the focus "
                "transaction-cost assumption."
            ),
        },
        {
            "category": "exploratory_ablation",
            "finding": (
                "Highest reduced-feature "
                "break-even cost"
            ),
            "model": str(
                best_reduced["model"]
            ),
            "variant": str(
                best_reduced["variant"]
            ),
            "value": float(
                best_reduced[
                    "pooled_break_even_cost_bps"
                ]
            ),
            "unit": "bps",
            "status": "exploratory",
            "interpretation": (
                "This post-hoc feature variant "
                "requires independent validation "
                "and cannot replace the frozen "
                "primary result."
            ),
        },
        {
            "category": "exploratory_ablation",
            "finding": (
                "One-basis-point annualized "
                "return of best reduced variant"
            ),
            "model": str(
                best_reduced["model"]
            ),
            "variant": str(
                best_reduced["variant"]
            ),
            "value": float(
                best_reduced[
                    (
                        "pooled_annualized_net_"
                        "return_at_focus_cost"
                    )
                ]
            ),
            "unit": "log return/year",
            "status": "exploratory",
            "interpretation": (
                "Positive simulated return is "
                "not confirmatory because the "
                "variant was inspected on the "
                "same test sample."
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_model_evidence_table(
    tables: Mapping[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    oos = tables["oos"]
    break_even = tables["break_even"]

    prediction = (
        tables[
            "bootstrap_prediction"
        ]
        .loc[
            tables[
                "bootstrap_prediction"
            ][
                "block_length_timestamps"
            ]
            == 12,
            [
                "model",
                "confidence_lower",
                "confidence_upper",
                "probability_positive",
                "undefined_timestamp_count",
            ],
        ]
        .rename(
            columns={
                "confidence_lower": (
                    "rank_ic_ci_lower"
                ),
                "confidence_upper": (
                    "rank_ic_ci_upper"
                ),
                "probability_positive": (
                    "rank_ic_bootstrap_"
                    "positive_frequency"
                ),
            }
        )
    )

    portfolio = (
        tables[
            "bootstrap_portfolio"
        ]
        .loc[
            tables[
                "bootstrap_portfolio"
            ][
                "block_length_periods"
            ]
            == 24,
            [
                "model",
                "break_even_confidence_lower",
                "break_even_confidence_upper",
                (
                    "probability_break_even_"
                    "above_focus_cost"
                ),
                "net_probability_positive",
                "observed_mean_turnover",
            ],
        ]
        .rename(
            columns={
                "observed_mean_turnover": (
                    "mean_turnover"
                ),
            }
        )
    )

    result = (
        oos[
            [
                "model",
                "mean_rank_ic",
                "rank_ic_ir",
                "positive_timestamp_fraction",
            ]
        ]
        .merge(
            break_even[
                [
                    "model",
                    "pooled_break_even_cost_bps",
                    (
                        "worst_individual_"
                        "break_even_cost_bps"
                    ),
                ]
            ],
            on="model",
            validate="one_to_one",
        )
        .merge(
            prediction,
            on="model",
            validate="one_to_one",
        )
        .merge(
            portfolio,
            on="model",
            validate="one_to_one",
        )
        .sort_values(
            "mean_rank_ic",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result


def build_report_html(
    title: str,
    subtitle: str,
    report_version: int,
    milestones_covered: int,
    focus_cost_bps: float,
    tables: Mapping[
        str,
        pd.DataFrame,
    ],
    findings: pd.DataFrame,
    inventory: pd.DataFrame,
) -> str:
    model_evidence = (
        build_model_evidence_table(
            tables
        )
    )

    stability = tables["stability"].copy()

    require_columns(
        stability,
        [
            "model",
            "worst_fold_rank_ic",
            "worst_month_rank_ic",
            (
                "positive_month_rank_ic_"
                "fraction"
            ),
            (
                "worst_month_break_even_"
                "cost_bps"
            ),
            (
                "positive_month_fraction_"
                "at_focus_cost"
            ),
        ],
        "Stability summary",
    )

    symbol_dependence = tables[
        "symbol_dependence"
    ].copy()

    require_columns(
        symbol_dependence,
        [
            "model",
            "worst_exclusion_rank_ic",
            (
                "maximum_absolute_rank_ic_"
                "delta"
            ),
            (
                "worst_exclusion_break_even_"
                "cost_bps"
            ),
            (
                "largest_absolute_"
                "contribution_share"
            ),
            (
                "top_three_absolute_"
                "contribution_share"
            ),
        ],
        "Symbol-dependence summary",
    )

    regimes = tables["regimes"].copy()

    require_columns(
        regimes,
        [
            "model",
            "regime_dimension",
            "worst_cell_rank_ic",
            (
                "positive_rank_ic_cell_"
                "fraction"
            ),
            "worst_break_even_cost_bps",
            (
                "positive_cell_fraction_"
                "at_focus_cost"
            ),
        ],
        "Regime summary",
    )

    regime_summary = (
        regimes.groupby(
            "model",
            as_index=False,
            sort=True,
        )
        .agg(
            worst_regime_rank_ic=(
                "worst_cell_rank_ic",
                "min",
            ),
            minimum_positive_rank_ic_cell_fraction=(
                (
                    "positive_rank_ic_"
                    "cell_fraction"
                ),
                "min",
            ),
            worst_regime_break_even_cost_bps=(
                "worst_break_even_cost_bps",
                "min",
            ),
            maximum_positive_regime_fraction_at_focus_cost=(
                (
                    "positive_cell_fraction_"
                    "at_focus_cost"
                ),
                "max",
            ),
        )
        .sort_values(
            "worst_regime_rank_ic",
            ascending=False,
        )
    )

    horizon = tables["horizon"].copy()

    require_columns(
        horizon,
        [
            "model",
            "horizon_minutes",
            "pooled_break_even_cost_bps",
            (
                "worst_individual_"
                "break_even_cost_bps"
            ),
            "mean_zero_cost_sharpe",
        ],
        "Horizon summary",
    )

    best_horizon = (
        horizon.sort_values(
            [
                "horizon_minutes",
                "pooled_break_even_cost_bps",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "horizon_minutes",
            as_index=False,
            sort=True,
        )
        .first()
    )

    ablation_features = tables[
        "ablation_features"
    ].copy()

    require_columns(
        ablation_features,
        [
            "model",
            "variant",
            "feature_count",
            "mean_rank_ic",
            "mean_rank_ic_delta_vs_full",
            "worst_fold_rank_ic",
            "pooled_break_even_cost_bps",
            (
                "pooled_break_even_cost_bps_"
                "delta_vs_full"
            ),
            (
                "pooled_annualized_net_return_"
                "at_focus_cost"
            ),
        ],
        "Feature-ablation summary",
    )

    ablation_features = (
        ablation_features.sort_values(
            [
                "model",
                "mean_rank_ic",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    rank_chart = (
        render_symmetric_bar_chart(
            frame=model_evidence,
            label_column="model",
            value_column="mean_rank_ic",
            title=(
                "Frozen out-of-sample "
                "mean Rank IC"
            ),
        )
    )

    break_even_chart = (
        render_symmetric_bar_chart(
            frame=model_evidence,
            label_column="model",
            value_column=(
                "pooled_break_even_cost_bps"
            ),
            title=(
                "Frozen pooled break-even "
                "transaction cost"
            ),
            value_suffix=" bps",
        )
    )

    findings_table = render_table(
        findings,
        columns=[
            "status",
            "category",
            "finding",
            "model",
            "variant",
            "value",
            "unit",
            "interpretation",
        ],
    )

    model_table = render_table(
        model_evidence,
        columns=[
            "model",
            "mean_rank_ic",
            "rank_ic_ci_lower",
            "rank_ic_ci_upper",
            "rank_ic_ir",
            "pooled_break_even_cost_bps",
            (
                "break_even_confidence_"
                "lower"
            ),
            (
                "break_even_confidence_"
                "upper"
            ),
            (
                "probability_break_even_"
                "above_focus_cost"
            ),
            "mean_turnover",
            "undefined_timestamp_count",
        ],
    )

    horizon_table = render_table(
        best_horizon,
        columns=[
            "horizon_minutes",
            "model",
            "pooled_break_even_cost_bps",
            (
                "worst_individual_"
                "break_even_cost_bps"
            ),
            "mean_zero_cost_sharpe",
        ],
    )

    stability_table = render_table(
        stability.sort_values(
            "worst_month_rank_ic",
            ascending=False,
        ),
        columns=[
            "model",
            "worst_fold_rank_ic",
            "worst_month_rank_ic",
            (
                "positive_month_rank_ic_"
                "fraction"
            ),
            (
                "worst_month_break_even_"
                "cost_bps"
            ),
            (
                "positive_month_fraction_"
                "at_focus_cost"
            ),
        ],
    )

    symbol_table = render_table(
        symbol_dependence.sort_values(
            "worst_exclusion_rank_ic",
            ascending=False,
        ),
        columns=[
            "model",
            "worst_exclusion_rank_ic",
            (
                "maximum_absolute_rank_ic_"
                "delta"
            ),
            (
                "worst_exclusion_break_even_"
                "cost_bps"
            ),
            (
                "largest_absolute_"
                "contribution_share"
            ),
            (
                "top_three_absolute_"
                "contribution_share"
            ),
        ],
    )

    regime_table = render_table(
        regime_summary,
        columns=[
            "model",
            "worst_regime_rank_ic",
            (
                "minimum_positive_rank_ic_"
                "cell_fraction"
            ),
            (
                "worst_regime_break_even_"
                "cost_bps"
            ),
            (
                "maximum_positive_regime_"
                "fraction_at_focus_cost"
            ),
        ],
    )

    ablation_table = render_table(
        ablation_features,
        columns=[
            "model",
            "variant",
            "feature_count",
            "mean_rank_ic",
            "mean_rank_ic_delta_vs_full",
            "worst_fold_rank_ic",
            "pooled_break_even_cost_bps",
            (
                "pooled_break_even_cost_bps_"
                "delta_vs_full"
            ),
            (
                "pooled_annualized_net_return_"
                "at_focus_cost"
            ),
        ],
    )

    inventory_table = render_table(
        inventory,
        columns=[
            "path",
            "bytes",
            "sha256",
        ],
        precision=0,
        table_class="inventory-table",
    )

    css = """
:root {
  color-scheme: light;
  --ink: #17202a;
  --muted: #5d6874;
  --line: #d9e0e7;
  --panel: #f6f8fa;
  --primary: #315b7d;
  --positive: #4f7896;
  --negative: #a65d5d;
  --warning: #8a641f;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: #ffffff;
  color: var(--ink);
  font-family:
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    sans-serif;
  line-height: 1.5;
}

main {
  width: min(1240px, calc(100% - 48px));
  margin: 0 auto;
  padding: 44px 0 72px;
}

header {
  border-bottom: 3px solid var(--primary);
  margin-bottom: 34px;
  padding-bottom: 22px;
}

h1 {
  font-size: 2.25rem;
  line-height: 1.12;
  margin: 0 0 8px;
}

.subtitle {
  color: var(--muted);
  font-size: 1.08rem;
  margin: 0;
}

.metadata {
  color: var(--muted);
  font-size: 0.9rem;
  margin-top: 14px;
}

section {
  margin-top: 42px;
}

h2 {
  border-bottom: 1px solid var(--line);
  font-size: 1.45rem;
  padding-bottom: 8px;
}

h3 {
  font-size: 1.1rem;
  margin-top: 30px;
}

.callout {
  background: var(--panel);
  border-left: 5px solid var(--primary);
  padding: 18px 20px;
}

.callout.warning {
  border-left-color: var(--warning);
}

.grid {
  display: grid;
  gap: 24px;
  grid-template-columns:
    repeat(auto-fit, minmax(440px, 1fr));
}

.chart-panel {
  border: 1px solid var(--line);
  overflow-x: auto;
  padding: 14px;
}

.bar-chart {
  display: block;
  min-width: 760px;
  width: 100%;
}

.chart-title {
  fill: var(--ink);
  font-size: 17px;
  font-weight: 650;
}

.chart-label,
.chart-value {
  fill: var(--ink);
  font-size: 13px;
}

.zero-line {
  stroke: #8a949e;
  stroke-width: 1;
}

.positive-bar {
  fill: var(--positive);
}

.negative-bar {
  fill: var(--negative);
}

table {
  border-collapse: collapse;
  display: block;
  font-size: 0.82rem;
  overflow-x: auto;
  width: 100%;
}

th,
td {
  border: 1px solid var(--line);
  padding: 7px 9px;
  text-align: right;
  vertical-align: top;
  white-space: nowrap;
}

th {
  background: var(--panel);
  font-weight: 650;
}

th:first-child,
td:first-child {
  text-align: left;
}

.inventory-table {
  font-family:
    ui-monospace,
    SFMono-Regular,
    Menlo,
    monospace;
  font-size: 0.72rem;
}

.inventory-table td:first-child {
  max-width: 460px;
  white-space: normal;
}

.inventory-table td:last-child {
  font-size: 0.67rem;
}

footer {
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.82rem;
  margin-top: 50px;
  padding-top: 18px;
}

code {
  background: var(--panel);
  padding: 1px 4px;
}

@media print {
  main {
    width: 100%;
    padding: 0;
  }

  section {
    break-inside: avoid;
  }
}
"""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta
  name="viewport"
  content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>{css}</style>
</head>
<body>
<main>
<header>
  <h1>{escape(title)}</h1>
  <p class="subtitle">{escape(subtitle)}</p>
  <p class="metadata">
    Report version {report_version} ·
    Evidence through Milestone {milestones_covered} ·
    Focus transaction cost {focus_cost_bps:g} bps
  </p>
</header>

<section id="executive-conclusion">
  <h2>Executive conclusion</h2>
  <div class="callout">
    <strong>Primary result:</strong>
    The frozen experiment contains statistically persistent
    cross-sectional ranking information, but no frozen model supports
    a one-basis-point transaction-cost assumption. The simple
    reversal reference remains stronger than every learned model.
  </div>
  <p>
    Several learned models have block-bootstrap Rank IC intervals
    above zero. That predictive evidence does not translate into
    sufficiently high break-even transaction costs. The distinction
    between ranking accuracy and economically usable turnover-adjusted
    return is central to this report.
  </p>
  <div class="callout warning">
    <strong>Exploratory exception:</strong>
    a reduced-feature LightGBM variant exceeds one basis point after
    removing calendar features. This variant was identified through
    post-hoc test-sample comparison and has not received independent
    holdout, bootstrap, monthly, regime, or symbol-dependence
    confirmation. It is a follow-up hypothesis, not a revised primary
    conclusion.
  </div>
</section>

<section id="key-findings">
  <h2>Key findings</h2>
  {findings_table}
</section>

<section id="research-design">
  <h2>Research design</h2>
  <p>
    The study uses a fixed universe of twenty Binance perpetual-futures
    contracts, five-minute public OHLCV data, causal point-in-time
    features, a demeaned sixty-minute cross-sectional return target,
    three purged walk-forward folds, and validation-only model
    selection.
  </p>
  <p>
    Portfolio evaluation enters at the next bar open, exits after
    twelve five-minute bars, holds the top four and bottom four
    contracts with zero net and unit gross exposure, and separates the
    overlapping horizon into twelve non-overlapping hourly cohorts.
    Transaction cost is modelled as one-way basis points multiplied by
    turnover.
  </p>
</section>

<section id="model-evidence">
  <h2>Frozen model evidence</h2>
  <div class="grid">
    <div class="chart-panel">{rank_chart}</div>
    <div class="chart-panel">{break_even_chart}</div>
  </div>
  <h3>Predictive, economic, and uncertainty metrics</h3>
  {model_table}
</section>

<section id="horizon">
  <h2>Holding-horizon sensitivity</h2>
  <p>
    Frozen sixty-minute-trained forecasts were transferred to
    alternative holding horizons without retraining. This is a signal
    decay audit rather than horizon-specific model optimization.
  </p>
  {horizon_table}
</section>

<section id="temporal-stability">
  <h2>Fold and monthly stability</h2>
  <p>
    Positive Rank IC is persistent for several models, but positive
    one-basis-point monthly performance is rare. Monthly results use
    linear return quantities rather than treating overlapping sleeves
    as independent Sharpe observations.
  </p>
  {stability_table}
</section>

<section id="symbol-dependence">
  <h2>Symbol dependence</h2>
  <p>
    Leave-one-symbol-out tests show that the predictive ordering is not
    generated by a single contract. They do not remove fixed-universe
    survivorship bias.
  </p>
  {symbol_table}
</section>

<section id="regimes">
  <h2>Causal market regimes</h2>
  <p>
    Regime thresholds are calibrated using only information available
    by each fold's validation boundary. Reversal is the only model
    whose Rank IC remains positive in every tested fold-regime cell.
  </p>
  {regime_table}
</section>

<section id="ablations">
  <h2>Feature-family ablations</h2>
  <p>
    Ridge and LightGBM were retrained after removing returns,
    range/volatility, activity/flow, or calendar features. Full-feature
    controls reproduce the original frozen evidence. Reduced-feature
    rows are exploratory because multiple variants were inspected on
    the same test sample.
  </p>
  {ablation_table}
</section>

<section id="limitations">
  <h2>Limitations</h2>
  <ul>
    <li>
      The universe is fixed ex post and therefore subject to
      survivorship and selection bias.
    </li>
    <li>
      Public OHLCV simulation does not model queue position, latency,
      market impact, adverse selection, partial fills, funding, or
      exchange-specific execution failure.
    </li>
    <li>
      Break-even cost is a simplified one-way threshold, not an
      achievable live execution estimate.
    </li>
    <li>
      Feature-ablation improvements are test-sample hypotheses and
      require a new untouched evaluation period.
    </li>
    <li>
      Bootstrap intervals address sampling dependence within the
      observed period, not future structural change.
    </li>
  </ul>
</section>

<section id="evidence-inventory">
  <h2>Evidence inventory</h2>
  <p>
    Every listed source file is identified by byte size and SHA-256.
    The report contains no external stylesheet, script, image, font,
    or network dependency.
  </p>
  {inventory_table}
</section>

<footer>
  Deterministic research artifact. No generation timestamp is embedded;
  identical committed inputs produce identical report bytes.
</footer>
</main>
</body>
</html>
"""

    return html


def assert_self_contained_html(
    html: str,
) -> None:
    lowered = html.lower()

    forbidden = [
        "http://",
        "https://",
        "<script src=",
        "<link ",
        "<img ",
        "@import",
    ]

    present = [
        token
        for token in forbidden
        if token in lowered
    ]

    if present:
        raise ValueError(
            "Report contains external-resource "
            f"tokens: {present}"
        )

    required = [
        "<style>",
        "<svg",
        "Executive conclusion",
        "Evidence inventory",
    ]

    missing = [
        token
        for token in required
        if token not in html
    ]

    if missing:
        raise ValueError(
            "Report is missing required "
            f"content: {missing}"
        )
