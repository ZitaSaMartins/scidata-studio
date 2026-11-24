"""Descriptive sensory profiling: QDA, temporal (TDS/TI-style) curves, product characterization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from statsmodels.formula.api import ols

from modules.stat_format import format_alpha_percent, select_significance_level, with_formatted_p_values

from . import characterization_stats
from ._shared import _columns, _compact_letters, _frame, _heatmap, _need
from .characterization_report import design_inputs, run_product_characterization


def _qda(df: pd.DataFrame, title: str = "QDA / Sensory Profile") -> None:
    st.header(title)
    st.write(
        "Summarises product profiles and runs an ANOVA per attribute. With an assessor column the model is "
        "Y = Product + Assessor (as in XLSTAT), so differences in how assessors use the scale are not counted as error."
    )
    design = design_inputs(df, f"{title}_qda", with_session=False)
    if design is None:
        return
    alpha = select_significance_level()
    attributes = list(design.descriptors)
    clean = _frame(df, [*design.factors, *attributes])
    if not _need(clean[design.product].nunique() >= 2, "Need at least two products."):
        return
    try:
        res = characterization_stats.analyse(clean, design)
    except ValueError as exc:
        st.error(str(exc))
        return
    profile = clean.groupby(design.product, observed=True)[attributes].agg(["mean", "std", "count"])
    st.subheader("Product summaries")
    st.dataframe(profile, width="stretch")

    st.subheader("Inferential product comparisons")
    st.caption(f"Model: {design.model_label}. Test value = normal quantile of the p-value (larger = more discriminating).")
    tests = res["power"].rename(columns={"Descriptors": "Attribute", "p-values": "p-value", "Test values": "Test value"})
    tests[f"Significant ({format_alpha_percent(alpha)})"] = np.where(tests["p-value"] < alpha, "Yes", "No")
    tests = tests[["Attribute", "F", "p-value", "Test value", f"Significant ({format_alpha_percent(alpha)})"]]
    st.dataframe(with_formatted_p_values(tests, ["p-value"]), width="stretch", hide_index=True)

    letters = _tukey_letters(res, alpha)
    if not letters.empty:
        st.subheader("Tukey HSD grouping letters")
        st.dataframe(letters, width="stretch", hide_index=True)
        st.caption(
            "Within each attribute, products sharing a letter are not significantly different "
            f"(Tukey-Kramer on the {'adjusted ' if design.assessor else ''}means, error term of the model above)."
        )
    means = res["means"]
    _heatmap(means, "Adjusted mean sensory profile" if design.assessor else "Mean sensory profile")
    significant = tests.loc[tests["p-value"] < alpha, "Attribute"].tolist()
    alpha_percent = format_alpha_percent(alpha)
    st.write(
        f"At the {alpha_percent} level, product means differ for: **{', '.join(significant)}**."
        if significant else f"No selected attribute showed a product difference at the {alpha_percent} level."
    )


def _tukey_letters(res: dict, alpha: float) -> pd.DataFrame:
    pairs = res["pairs"]
    rows: list[dict[str, object]] = []
    for attribute, means in res["means"].items():
        subset = pairs[(pairs["Descriptor"] == attribute) & (pairs["p-value (Tukey)"] < alpha)]
        significant = {tuple(sorted((a, b))) for a, b in zip(subset["Product 1"], subset["Product 2"], strict=True)}
        products = [str(p) for p in means.dropna().index]
        letters = _compact_letters(products, significant)
        rows.extend(
            {"Attribute": attribute, "Product": product, "Mean": float(means[product]), "Grouping letters": letters[product]}
            for product in products
        )
    return pd.DataFrame(rows)


def _temporal(df: pd.DataFrame) -> None:
    st.header("Temporal Sensory Analysis")
    st.write("Displays trajectories over time and tests product, time, and product×time effects.")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 2 and df.shape[1] >= 3, "Need time, product, and numeric response columns."):
        return
    product = st.selectbox("Product", df.columns.astype(str).tolist(), key="temporal_product")
    time_candidates = [c for c in numeric if c != product]
    if not _need(bool(time_candidates), "Select a Product column that leaves a numeric Time column available."):
        return
    time = st.selectbox("Time", time_candidates, key="temporal_time")
    response_candidates = [c for c in numeric if c not in {product, time}]
    if not _need(bool(response_candidates), "Select Product/Time columns that leave a numeric response column available."):
        return
    response = st.selectbox("Intensity / dominance score", response_candidates, key="temporal_y")
    alpha = select_significance_level()
    clean = _frame(df, [product, time, response])
    summary = clean.groupby([product, time], observed=True)[response].agg(["mean", "std", "count"]).reset_index()
    st.dataframe(summary, width="stretch")
    cell_counts = clean.groupby([product, time], observed=True).size().unstack(fill_value=0)
    if (
        cell_counts.shape[0] >= 2
        and cell_counts.shape[1] >= 2
        and (cell_counts > 0).all().all()
        and len(clean) > cell_counts.size
    ):
        model_data = clean.rename(
            columns={product: "_product", time: "_time", response: "_response"}
        )
        model = ols("_response ~ C(_product) * C(_time)", data=model_data).fit()
        anova = sm.stats.anova_lm(model, typ=2).reset_index()
        anova.columns = ["Effect", *anova.columns[1:]]
        st.subheader("Product × time ANOVA")
        st.dataframe(with_formatted_p_values(anova, ["PR(>F)"]), width="stretch")
        interaction_p = float(
            anova.loc[anova["Effect"] == "C(_product):C(_time)", "PR(>F)"].iloc[0]
        )
        if not np.isfinite(interaction_p):
            st.error("The product × time interaction produced an invalid (non-finite) p-value.")
        elif interaction_p < alpha:
            st.success("Temporal trajectories differ significantly among products.")
        else:
            st.info("No significant product × time interaction was detected.")
    else:
        st.warning(
            "The product × time ANOVA requires every selected product-time combination; "
            "the trajectory and product-specific trends are still shown."
        )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=summary, x=time, y="mean", hue=product, marker="o", ax=ax)
    ax.set(ylabel=f"Mean {response}", title="Temporal sensory curves")
    st.pyplot(fig)
    plt.close(fig)
    trend_rows: list[dict[str, object]] = []
    for product_name, group in clean.groupby(product, observed=True):
        if len(group) >= 3 and group[time].nunique() >= 2:
            trend = stats.linregress(group[time], group[response])
            trend_rows.append(
                {
                    "Product": product_name,
                    "Slope per time unit": trend.slope,
                    "R²": trend.rvalue**2,
                    "Slope p-value": trend.pvalue,
                }
            )
    st.subheader("Product-specific temporal trends")
    st.dataframe(with_formatted_p_values(pd.DataFrame(trend_rows), ["Slope p-value"]), width="stretch")
    st.caption(
        "Positive slopes indicate increasing intensity and negative slopes indicate decay. "
        "Separated or non-parallel curves suggest product-specific temporal evolution."
    )


def _product_characterization(df: pd.DataFrame) -> None:
    run_product_characterization(df)

