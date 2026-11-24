"""Discrimination and ranking tests: triangle, paired, duo-trio, tetrad, ranking."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from statsmodels.stats.multitest import multipletests

from modules.stat_format import (
    format_alpha,
    format_alpha_percent,
    format_p_value,
    select_significance_level,
    with_formatted_p_values,
)

from ._shared import _columns, _compact_letters, _interpret_p, _need
from .discrimination_stats import (
    METHODS,
    answers_outside_samples,
    discrimination_summary,
    score_responses,
)

_PER_ROW = "Answer + correct answer columns"
_SINGLE = "One correct/incorrect column"


def _guess_column(columns: list[str], include: str, exclude: str | None = None) -> int:
    for index, column in enumerate(columns):
        lowered = column.lower()
        if include in lowered and (exclude is None or exclude not in lowered):
            return index
    return 0


def _score_per_row(df: pd.DataFrame, key: str) -> pd.Series | None:
    columns = df.columns.astype(str).tolist()
    left, right = st.columns(2)
    answer = left.selectbox(
        "Assessor's answer column",
        columns,
        index=_guess_column(columns, "answer", exclude="correct"),
        key=f"{key}_answer",
    )
    correct = right.selectbox(
        "Correct answer column",
        columns,
        index=_guess_column(columns, "correct"),
        key=f"{key}_correct_col",
        help="The code of the sample the assessor should have picked in that row "
        "(for a triangle test, the odd sample).",
    )
    if not _need(answer != correct, "Choose two different columns for the answer and the correct answer."):
        return None
    others = [c for c in columns if c not in {answer, correct}]
    samples = st.multiselect(
        "Sample columns (optional, used to check the codes)",
        others,
        default=[c for c in others if "sample" in c.lower()],
        key=f"{key}_samples",
    )
    if samples:
        for label, column in (("answer", answer), ("correct answer", correct)):
            outside = answers_outside_samples(df, samples, column)
            if outside.any():
                st.warning(
                    f"{int(outside.sum())} row(s) have an {label} that is not one of the samples "
                    "served in that row. Check the codes."
                )
                st.dataframe(df.loc[outside], width="stretch")
    return score_responses(df[answer], df[correct])


def _score_single_column(df: pd.DataFrame, key: str) -> pd.Series | None:
    outcome = st.selectbox("Response / correct column", df.columns.tolist(), key=f"{key}_outcome")
    values = df[outcome].dropna().unique().tolist()
    if not _need(len(values) >= 1, "The response column must contain at least one observed outcome."):
        return None
    if len(values) == 1:
        marks_correct = st.segmented_control(
            f"Does the value '{values[0]}' represent a correct response?",
            ["Yes", "No"],
            default="Yes",
            key=f"{key}_single_outcome",
        )
        if marks_correct is None:
            return None
        correct_value = values[0] if marks_correct == "Yes" else object()
    else:
        correct_value = st.selectbox("Value meaning correct", values, key=f"{key}_correct")
    scored = (df[outcome] == correct_value).astype("boolean")
    scored[df[outcome].isna()] = pd.NA
    return scored


def _discrimination(df: pd.DataFrame, name: str, method_key: str, key: str) -> None:
    method = METHODS[method_key]
    chance = method.guess
    st.header(name)
    st.write(
        f"Tests whether the number of correct responses exceeds chance ({chance:.3g}) "
        "using an exact one-sided binomial test, and estimates d-prime with the "
        "Thurstonian model."
    )
    if not _need(df.shape[1] > 0 and len(df) > 0, "Upload response-level data first."):
        return
    columns = df.columns.astype(str).tolist()
    has_correct_column = len(columns) >= 2 and any("correct" in c.lower() for c in columns)
    layout = st.segmented_control(
        "Data layout",
        [_PER_ROW, _SINGLE],
        default=_PER_ROW if has_correct_column else _SINGLE,
        key=f"{key}_layout",
        help="Use the first option when each row holds the assessor's answer and the correct "
        "answer for that row (as in an XLSTAT design sheet). Use the second when a column "
        "already says whether each answer was right.",
    )
    if layout is None:
        st.info("Choose a data layout.")
        return
    scored = _score_per_row(df, key) if layout == _PER_ROW else _score_single_column(df, key)
    if scored is None:
        return
    valid = scored.notna()
    total = int(valid.sum())
    if not _need(total > 0, "No rows with a usable answer."):
        return
    missing = int((~valid).sum())
    if missing:
        st.caption(f"{missing} row(s) with a missing value were left out.")
    successes = int(scored[valid].sum())

    if layout == _PER_ROW:
        with st.expander(f"Per-assessor results ({successes} correct of {total})"):
            marks = scored.map({True: "+", False: "-"}, na_action="ignore")
            st.dataframe(df.assign(**{"Correct / Incorrect": marks}), width="stretch")

    alpha = select_significance_level()
    test, parameters = discrimination_summary(successes, total, method, alpha)
    critical = test["Critical correct"]
    st.dataframe(
        with_formatted_p_values(
            pd.DataFrame(
                {
                    "Correct": [successes],
                    "Total": [total],
                    "Observed proportion": [successes / total],
                    "Chance proportion": [chance],
                    "d-prime": [test["d-prime"]],
                    f"Critical correct (α={format_alpha(alpha)})": [
                        critical if critical is not None else "Not attainable"
                    ],
                    "Exact p-value": [test["p-value"]],
                    "Power": [test["Power"]],
                }
            ),
            ["Exact p-value"],
        ),
        width="stretch",
    )
    confidence = f"{(1 - alpha) * 100:g}%"
    st.subheader(f"Estimated parameters ({method.name}, Clopper-Pearson {confidence} CI)")
    st.dataframe(
        parameters.rename(
            columns={
                "Lower bound": f"Lower bound ({confidence})",
                "Upper bound": f"Upper bound ({confidence})",
            }
        ),
        width="stretch",
    )
    note = "Power is the binomial power of the test at the observed proportion of correct answers."
    if method_key == "paired":
        note += " The paired comparison uses the directional (2-AFC) model."
    st.caption(note)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(["Observed", "Chance"], [successes / total, chance], color=["#2a9d8f", "#bdbdbd"])
    ax.set(ylim=(0, 1), ylabel="Correct response proportion", title=name)
    st.pyplot(fig)
    plt.close(fig)
    if test["p-value"] < alpha:
        st.success("The panel discriminated above chance.")
    else:
        st.info("The observed correct rate is not convincingly above chance.")


def _ranking(df: pd.DataFrame) -> None:
    st.header("Sensory Ranking Test")
    st.write("Friedman's test compares product ranks while accounting for each assessor.")
    numeric, _ = _columns(df)
    if not _need(bool(numeric) and df.shape[1] >= 3, "Need assessor, product, and numeric rank columns."):
        return
    columns = df.columns.astype(str).tolist()
    assessor = st.selectbox("Assessor", columns, key="rank_assessor")
    product = st.selectbox("Product", [c for c in columns if c != assessor], key="rank_product")
    rank_candidates = [c for c in numeric if c not in {assessor, product}]
    if not _need(bool(rank_candidates), "Select Assessor/Product columns that leave a numeric Rank column available."):
        return
    rank = st.selectbox("Rank", rank_candidates, key="rank_value")
    alpha = select_significance_level()
    pivot = df.pivot_table(index=assessor, columns=product, values=rank, aggfunc="mean").dropna()
    if not _need(pivot.shape[0] >= 2 and pivot.shape[1] >= 3, "Need at least 2 complete assessors and 3 products."):
        return
    statistic, p_value = stats.friedmanchisquare(*(pivot[column] for column in pivot))
    summary = pd.DataFrame({"Mean rank": pivot.mean(), "Median rank": pivot.median()}).sort_values("Mean rank")
    st.write(f"Friedman χ²({pivot.shape[1] - 1}) = **{statistic:.3f}**, p = **{format_p_value(p_value)}**")
    _interpret_p(float(p_value), alpha)

    pair_metadata: list[tuple[str, str, float, float]] = []
    raw_p_values: list[float] = []
    products = [str(column) for column in pivot.columns]
    for first_index, first in enumerate(pivot.columns):
        for second in pivot.columns[first_index + 1 :]:
            difference = pivot[first] - pivot[second]
            if np.allclose(difference, 0):
                pair_statistic, pair_p = 0.0, 1.0
            else:
                pair_statistic, pair_p = stats.wilcoxon(difference)
            raw_p_values.append(float(pair_p))
            pair_metadata.append(
                (str(first), str(second), float(pair_statistic), float(difference.mean()))
            )
    adjusted = multipletests(raw_p_values, alpha=alpha, method="holm")[1]
    significant_pairs = {
        tuple(sorted((first, second)))
        for (first, second, _, _), adjusted_p in zip(pair_metadata, adjusted, strict=False)
        if adjusted_p < alpha
    }
    letters = _compact_letters(products, significant_pairs)
    summary["Grouping letters"] = [letters[str(product_name)] for product_name in summary.index]
    st.subheader("Holm-adjusted pairwise Wilcoxon comparisons")
    st.dataframe(
        with_formatted_p_values(
            pd.DataFrame(
                [
                    {
                        "Product 1": first,
                        "Product 2": second,
                        "W": pair_statistic,
                        "Mean rank difference": mean_difference,
                        "Adjusted p-value": adjusted_p,
                        f"Significant ({format_alpha_percent(alpha)})": adjusted_p < alpha,
                    }
                    for (first, second, pair_statistic, mean_difference), adjusted_p in zip(
                        pair_metadata, adjusted, strict=False
                    )
                ]
            ),
            ["Adjusted p-value"],
        ),
        width="stretch",
    )
    st.dataframe(summary, width="stretch")
    st.caption("Products sharing a grouping letter are not significantly different.")
    fig, ax = plt.subplots(figsize=(8, 4))
    summary["Mean rank"].plot.bar(ax=ax, color="#457b9d")
    ax.set(ylabel="Mean rank", title="Lower mean rank indicates the more highly ranked product")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

