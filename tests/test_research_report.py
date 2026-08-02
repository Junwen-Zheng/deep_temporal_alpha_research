from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deep_alpha.reporting.research_report import (
    assert_self_contained_html,
    build_key_findings,
    build_model_evidence_table,
    build_source_inventory,
    format_value,
    render_symmetric_bar_chart,
    render_table,
)


def sample_tables() -> dict[
    str,
    pd.DataFrame,
]:
    return {
        "oos": pd.DataFrame(
            {
                "model": [
                    "reversal_12",
                    "ridge",
                ],
                "mean_rank_ic": [
                    0.04,
                    0.01,
                ],
            }
        ),
        "break_even": pd.DataFrame(
            {
                "model": [
                    "reversal_12",
                    "ridge",
                ],
                "pooled_break_even_cost_bps": [
                    0.6,
                    0.3,
                ],
            }
        ),
        "bootstrap_prediction": (
            pd.DataFrame(
                {
                    "model": [
                        "reversal_12",
                        "ridge",
                    ],
                    "block_length_timestamps": [
                        12,
                        12,
                    ],
                    "confidence_lower": [
                        0.03,
                        0.002,
                    ],
                    "confidence_upper": [
                        0.05,
                        0.02,
                    ],
                }
            )
        ),
        "bootstrap_portfolio": (
            pd.DataFrame(
                {
                    "model": [
                        "reversal_12",
                        "ridge",
                    ],
                    "block_length_periods": [
                        24,
                        24,
                    ],
                    (
                        "probability_break_even_"
                        "above_focus_cost"
                    ): [
                        0.0,
                        0.01,
                    ],
                    (
                        "break_even_confidence_"
                        "upper"
                    ): [
                        0.8,
                        1.1,
                    ],
                }
            )
        ),
        "ablation_features": (
            pd.DataFrame(
                {
                    "model": [
                        "ridge",
                        "ridge",
                    ],
                    "variant": [
                        "full",
                        "without_calendar",
                    ],
                    "mean_rank_ic": [
                        0.01,
                        0.02,
                    ],
                    (
                        "pooled_break_even_"
                        "cost_bps"
                    ): [
                        0.3,
                        1.2,
                    ],
                    (
                        "pooled_annualized_net_"
                        "return_at_focus_cost"
                    ): [
                        -0.5,
                        0.1,
                    ],
                }
            )
        ),
        "ablation_families": (
            pd.DataFrame(
                {
                    "model": [
                        "reversal_12",
                        "ridge",
                    ],
                    "model_family": [
                        "reversal_reference",
                        "linear_pointwise",
                    ],
                    "mean_rank_ic": [
                        0.04,
                        0.01,
                    ],
                }
            )
        ),
    }


def test_format_value_is_deterministic() -> None:
    assert format_value(
        0.12345678,
        precision=4,
    ) == "0.1235"

    assert format_value(1234) == "1,234"
    assert format_value(True) == "Yes"


def test_table_escapes_html() -> None:
    frame = pd.DataFrame(
        {
            "value": [
                "<script>alert(1)</script>",
            ]
        }
    )

    rendered = render_table(
        frame,
        columns=[
            "value",
        ],
    )

    assert "<script>" not in rendered

    assert (
        "&lt;script&gt;"
        in rendered
    )


def test_bar_chart_handles_negative_values() -> None:
    frame = pd.DataFrame(
        {
            "model": [
                "positive",
                "negative",
            ],
            "value": [
                1.0,
                -0.5,
            ],
        }
    )

    rendered = (
        render_symmetric_bar_chart(
            frame=frame,
            label_column="model",
            value_column="value",
            title="Example",
        )
    )

    assert "<svg" in rendered
    assert "positive-bar" in rendered
    assert "negative-bar" in rendered


def test_source_inventory_is_sorted(
    tmp_path: Path,
) -> None:
    second = tmp_path / "b.txt"
    first = tmp_path / "a.txt"

    second.write_text(
        "second\n",
        encoding="utf-8",
    )

    first.write_text(
        "first\n",
        encoding="utf-8",
    )

    inventory = build_source_inventory(
        [
            second,
            first,
        ]
    )

    assert inventory[
        "path"
    ].tolist() == [
        first.as_posix(),
        second.as_posix(),
    ]

    assert inventory[
        "sha256"
    ].str.len().eq(64).all()


def test_key_findings_select_expected_rows() -> None:
    findings = build_key_findings(
        tables=sample_tables(),
        focus_cost_bps=1.0,
    )

    primary_rank = findings.loc[
        findings["finding"]
        == (
            "Highest frozen out-of-sample "
            "Rank IC"
        )
    ].iloc[0]

    exploratory = findings.loc[
        findings["finding"]
        == (
            "Highest reduced-feature "
            "break-even cost"
        )
    ].iloc[0]

    assert (
        primary_rank["model"]
        == "reversal_12"
    )

    assert (
        exploratory["variant"]
        == "without_calendar"
    )

    assert np.isclose(
        exploratory["value"],
        1.2,
    )


def test_self_contained_report_is_accepted() -> None:
    html = (
        "<!doctype html>"
        "<html><head><style></style></head>"
        "<body>"
        "<h1>Executive conclusion</h1>"
        "<svg></svg>"
        "<h2>Evidence inventory</h2>"
        "</body></html>"
    )

    assert_self_contained_html(html)


def test_external_resource_is_rejected() -> None:
    html = (
        "<!doctype html>"
        "<html><head><style></style></head>"
        "<body>"
        "<h1>Executive conclusion</h1>"
        "<svg></svg>"
        "<h2>Evidence inventory</h2>"
        '<script src="https://example.com/a.js">'
        "</script>"
        "</body></html>"
    )

    with pytest.raises(
        ValueError,
        match="external-resource",
    ):
        assert_self_contained_html(html)

def test_model_evidence_uses_bootstrap_turnover() -> None:
    tables = {
        "oos": pd.DataFrame(
            {
                "model": [
                    "ridge",
                ],
                "mean_rank_ic": [
                    0.01,
                ],
                "rank_ic_ir": [
                    0.1,
                ],
                "positive_timestamp_fraction": [
                    0.55,
                ],
            }
        ),
        "break_even": pd.DataFrame(
            {
                "model": [
                    "ridge",
                ],
                "pooled_break_even_cost_bps": [
                    0.3,
                ],
                (
                    "worst_individual_"
                    "break_even_cost_bps"
                ): [
                    0.1,
                ],
            }
        ),
        "bootstrap_prediction": pd.DataFrame(
            {
                "model": [
                    "ridge",
                ],
                "block_length_timestamps": [
                    12,
                ],
                "confidence_lower": [
                    0.002,
                ],
                "confidence_upper": [
                    0.02,
                ],
                "probability_positive": [
                    0.99,
                ],
                "undefined_timestamp_count": [
                    0,
                ],
            }
        ),
        "bootstrap_portfolio": pd.DataFrame(
            {
                "model": [
                    "ridge",
                ],
                "block_length_periods": [
                    24,
                ],
                "break_even_confidence_lower": [
                    0.1,
                ],
                "break_even_confidence_upper": [
                    0.5,
                ],
                (
                    "probability_break_even_"
                    "above_focus_cost"
                ): [
                    0.0,
                ],
                "net_probability_positive": [
                    0.0,
                ],
                "observed_mean_turnover": [
                    1.25,
                ],
            }
        ),
    }

    result = build_model_evidence_table(
        tables
    )

    assert len(result) == 1

    assert np.isclose(
        result.loc[
            0,
            "mean_turnover",
        ],
        1.25,
    )
