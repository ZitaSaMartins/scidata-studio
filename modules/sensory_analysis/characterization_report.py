"""Product characterization report, laid out like XLSTAT's "Product characterization" sheet.

All numbers come from ``characterization_stats``; this module only handles
inputs, caching and display.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.stat_format import format_alpha_percent, select_significance_level

from . import characterization_stats as cs
from . import panel_stats as ps
from ._shared import _columns, _need
from .panel_report import _fmt_p, _guess

_NONE = "— none —"
_HIGHER = "#00C7D1"   # significantly above the average product (XLSTAT's cyan)
_LOWER = "#FF0000"    # significantly below (XLSTAT's red)
_FACTOR_LIKE = r"^(obs|id|rank|order|session|rep|replicate|panel|assess|judge|subject|consumer)"


# --------------------------------------------------------------------------- #
# Inputs and cached computation
# --------------------------------------------------------------------------- #

def default_descriptors(df: pd.DataFrame, factors: list[str]) -> list[str]:
    """Numeric columns that are neither a selected factor nor look like an id / rank / session column."""
    numeric, _ = _columns(df)
    return [c for c in numeric if c not in factors and not re.match(_FACTOR_LIKE, c, flags=re.IGNORECASE)]


def design_inputs(df: pd.DataFrame, key: str, with_session: bool = True) -> cs.CharacterizationDesign | None:
    columns = df.columns.astype(str).tolist()
    if not _need(len(columns) >= 2, "Need a product column and numeric sensory descriptors."):
        return None
    c1, c2, c3 = st.columns(3)
    product = c1.selectbox("Products", columns, index=_guess(df, [r"product", r"sample|prod"], 0), key=f"{key}_product")
    others = [c for c in columns if c != product]
    assessor_guess = _guess(df[others], [r"assess", r"panel|judge|subject|taster|consumer"], -1) if others else -1
    has_assessor = bool(others) and re.search(r"assess|panel|judge|subject|taster|consumer", others[assessor_guess], re.I)
    assessor_options = [_NONE, *others]
    assessor = c2.selectbox("Assessors (recommended)", assessor_options,
                            index=assessor_options.index(others[assessor_guess]) if has_assessor else 0,
                            key=f"{key}_assessor")
    assessor = None if assessor == _NONE else assessor
    session = None
    if with_session:
        session_options = [_NONE, *[c for c in others if c != assessor]]
        session = c3.selectbox("Sessions (optional)", session_options, index=0, key=f"{key}_session")
        session = None if session == _NONE else session
    factors = [c for c in (product, assessor, session) if c]
    numeric, _ = _columns(df)
    candidates = [c for c in numeric if c not in factors]
    descriptors = st.multiselect("Descriptors", candidates, default=default_descriptors(df, factors), key=f"{key}_descriptors")
    if not _need(bool(descriptors), "Select at least one descriptor."):
        return None
    interaction = False
    if assessor and cs.interaction_estimable(df.dropna(subset=factors), product, assessor):
        interaction = st.toggle("Include the product × assessor interaction (P*A)", value=False, key=f"{key}_interaction",
                                help="With P*A the product effect is tested against the interaction (assessors random).")
    return cs.CharacterizationDesign(product, tuple(descriptors), assessor, session, interaction)


@st.cache_data(show_spinner=False, max_entries=10)
def _compute(data: pd.DataFrame, design: cs.CharacterizationDesign, confidence: float) -> dict:
    return cs.analyse(data, design, confidence)


@st.cache_data(show_spinner=False, max_entries=10)
def _bootstrap(data: pd.DataFrame, design: cs.CharacterizationDesign, means: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    return cs.bootstrap_scores(data, design, cs.profile_pca(means), n_boot, seed)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def _summary(df: pd.DataFrame, design: cs.CharacterizationDesign) -> None:
    st.subheader("Summary statistics")
    quant = ps.summary_quantitative(df, list(design.descriptors))
    st.dataframe(quant.style.format(precision=3), hide_index=True, width="stretch")
    qual = ps.summary_qualitative(df, design.factors)
    st.dataframe(qual.style.format({"%": "{:.2f}"}), hide_index=True, width="stretch", height=250)


def _power(res: dict, design: cs.CharacterizationDesign, alpha: float) -> None:
    st.subheader("Discriminating power by descriptor")
    against = "the P*A interaction" if design.interaction and design.assessor else "the error"
    st.caption(f"Model: {design.model_label}, sum-to-zero contrasts, product effect tested against {against}. "
               "Test value = normal quantile of the product effect's p-value, so a larger value means a more "
               "discriminating descriptor. Sorted from most to least discriminating.")
    power = res["power"]
    st.dataframe(power[["Descriptors", "Test values", "p-values"]].style
                 .format({"Test values": "{:.3f}", "p-values": _fmt_p}),
                 hide_index=True, width="stretch")
    colours = np.where(power["p-values"] < alpha, "#4285F4", "#B0B0B0")
    fig = go.Figure(go.Bar(x=power["Descriptors"], y=power["Test values"], marker_color=colours))
    fig.update_layout(title="Discriminating power by descriptor", yaxis_title="Test value", height=380,
                      margin={"l": 10, "r": 10, "t": 50, "b": 10})
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Blue: the product effect is significant at the {format_alpha_percent(alpha)} level.")


def _coefficients(res: dict, alpha: float, confidence: float) -> None:
    st.subheader("Model coefficients")
    st.caption("Coefficient = the product's adjusted mean minus the average over products. "
               f"Test value = signed normal quantile of the t test's p-value. Bounds: {confidence:.0%} confidence interval.")
    coefficients = res["coefficients"]
    product = st.selectbox("Product", res["products"], key="characterization_coef_product")
    table = coefficients[coefficients["Product"] == product].drop(columns="Product").set_index("Descriptor")
    lower, upper = f"Lower bound {confidence:.0%}", f"Upper bound {confidence:.0%}"
    table = table.rename(columns={"Lower bound": lower, "Upper bound": upper})
    shown = table[["Coefficient", "Estimated mean", "p-value", "Test value", lower, upper]]
    st.dataframe(shown.style.format(precision=3).format({"p-value": _fmt_p})
                 .apply(lambda row: [_cell_colour(row["Coefficient"], row["p-value"], alpha)] * len(row), axis=1),
                 width="stretch")
    c1, c2 = st.columns(2, vertical_alignment="bottom")
    value = c1.segmented_control("Bars show", ["Coefficient", "Test value"], default="Coefficient",
                                 key="characterization_coef_value") or "Coefficient"
    every_product = c2.toggle("One chart per product", value=False, key="characterization_coef_all")
    st.caption(f"Blue: significantly positive, red: significantly negative, white: not significant "
               f"({format_alpha_percent(alpha)}). Descriptors sorted from the largest to the smallest value.")
    if every_product:
        grid = st.columns(2)
        for i, name in enumerate(res["products"]):
            with grid[i % 2]:
                _coefficient_chart(coefficients[coefficients["Product"] == name].set_index("Descriptor"), name, value, alpha)
    else:
        _coefficient_chart(table, product, value, alpha)
    with st.expander("All products side by side"):
        columns = ["Coefficient", "p-value", "Test value"]
        tables = {c: cs.coefficients_wide(coefficients, c) for c in columns}
        wide = pd.concat({(p, c): tables[c][p] for p in res["products"] for c in columns}, axis=1)
        st.dataframe(wide.style.format(precision=3), width="stretch")


def _coefficient_chart(table: pd.DataFrame, product: str, value: str, alpha: float) -> None:
    """XLSTAT-style bar chart of one product's coefficients (or test values) by descriptor."""
    ordered = table.sort_values(value, ascending=False)
    significant = ordered["p-value"] < alpha
    fill = np.where(~significant, "white", np.where(ordered["Coefficient"] > 0, "#4285F4", "#DB4437"))
    line = np.where(~significant, "#555555", fill)
    fig = go.Figure(go.Bar(
        x=ordered.index, y=ordered[value], marker={"color": fill, "line": {"color": line, "width": 1}},
        customdata=ordered["p-value"], hovertemplate="%{x}<br>" + value + ": %{y:.3f}<br>p-value: %{customdata:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#555555", line_width=1)
    fig.update_layout(title=product, xaxis_title="Descriptors", yaxis_title=f"{value}s", height=380,
                      margin={"l": 10, "r": 10, "t": 50, "b": 10})
    st.plotly_chart(fig, width="stretch", key=f"characterization_coef_chart_{product}")


def _cell_colour(coefficient: float, p_value: float, alpha: float) -> str:
    if not np.isfinite(p_value) or p_value >= alpha:
        return ""
    return f"background-color: {_HIGHER if coefficient > 0 else _LOWER}; color: black"


def _means(res: dict, pca: cs.ProfilePCA, alpha: float) -> None:
    st.subheader("Adjusted means by product")
    st.caption("Least-squares means. Products are sorted by their F1 score and descriptors by their correlation with F1. "
               f"Blue: significantly above the average product, red: significantly below, white: not significant "
               f"({format_alpha_percent(alpha)}, t test of the product's coefficient).")
    means = cs.sorted_means(res["means"], pca)
    coefficients = res["coefficients"].set_index(["Product", "Descriptor"])

    def colour(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [[_cell_colour(coefficients.loc[(p, d), "Coefficient"], coefficients.loc[(p, d), "p-value"], alpha)
              for d in frame.columns] for p in frame.index],
            index=frame.index, columns=frame.columns)

    styled = means.style.format(precision=3).apply(colour, axis=None)
    st.dataframe(styled, width="stretch")
    buffer = io.BytesIO()
    styled.to_excel(buffer, sheet_name="Adjusted means", engine="openpyxl")
    st.download_button("Download as Excel (with colours)", buffer.getvalue(), "adjusted_means.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       icon=":material/download:", key="characterization_means_download")


def _pca(res: dict, design: cs.CharacterizationDesign, pca: cs.ProfilePCA) -> None:
    st.subheader("Sensory profiles (PCA on the adjusted means)")
    if pca.scores.shape[1] < 2:
        st.info("The map needs at least three products and two descriptors that vary between products.")
        return
    st.caption("Normalized PCA (correlation matrix) of the product × descriptor table of adjusted means.")
    st.markdown("**Eigenvalues**")
    st.dataframe(pca.eigen_table.style.format(precision=3), width="stretch")
    ratio = pca.eigen_table.loc["Variability (%)"]

    show_ellipses = False
    if design.assessor:
        c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
        show_ellipses = c1.toggle("Show 95% confidence ellipses", value=True, key="characterization_ellipses",
                                  help="Assessors are resampled with replacement; each virtual panel's product means are projected on the map.")
        n_boot = c2.number_input("Bootstrap samples", 100, 2000, 500, step=100, key="characterization_nboot",
                                 disabled=not show_ellipses)
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    colour = {p: palette[i % len(palette)] for i, p in enumerate(pca.scores.index)}
    if show_ellipses:
        boot = _bootstrap(res["data"], design, res["means"], int(n_boot), cs.DEFAULT_SEED)
        for product, group in boot.groupby("Product", sort=False):
            outline = cs.ellipse_points(group[["F1", "F2"]].to_numpy())
            fig.add_trace(go.Scatter(x=outline[:, 0], y=outline[:, 1], mode="lines", line={"color": colour[product]},
                                     name=f"{product} (95%)", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pca.scores["F1"], y=pca.scores["F2"], mode="markers+text", text=pca.scores.index,
                             textposition="top center", marker={"size": 10, "color": [colour[p] for p in pca.scores.index]},
                             showlegend=False))
    fig.add_hline(y=0, line_color="grey", line_width=0.7)
    fig.add_vline(x=0, line_color="grey", line_width=0.7)
    fig.update_layout(title="Products (axes F1 and F2)", xaxis_title=f"F1 ({ratio['F1']:.2f} %)",
                      yaxis_title=f"F2 ({ratio['F2']:.2f} %)", height=520, margin={"l": 10, "r": 10, "t": 50, "b": 10})
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, width="stretch")

    circle = go.Figure()
    angle = np.linspace(0, 2 * np.pi, 200)
    circle.add_trace(go.Scatter(x=np.cos(angle), y=np.sin(angle), mode="lines", line={"color": "grey", "width": 1},
                                showlegend=False, hoverinfo="skip"))
    for descriptor, (x_value, y_value) in pca.loadings[["F1", "F2"]].iterrows():
        circle.add_annotation(x=x_value, y=y_value, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                              showarrow=True, arrowhead=2, arrowcolor="#4285F4", text="")
        circle.add_annotation(x=x_value, y=y_value, text=descriptor, showarrow=False, yshift=10)
    circle.update_layout(title="Descriptors (correlation circle)", xaxis_title=f"F1 ({ratio['F1']:.2f} %)",
                         yaxis_title=f"F2 ({ratio['F2']:.2f} %)", height=520, margin={"l": 10, "r": 10, "t": 50, "b": 10},
                         xaxis={"range": [-1.1, 1.1]}, yaxis={"range": [-1.1, 1.1], "scaleanchor": "x", "scaleratio": 1})
    st.plotly_chart(circle, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Factor scores**")
        st.dataframe(pca.scores.style.format(precision=3), width="stretch")
    with c2:
        st.markdown("**Squared cosines of the observations**")
        cos2 = pca.squared_cosines
        st.dataframe(cos2.style.format(precision=3).highlight_max(axis=1, props="font-weight: 700"), width="stretch")
    st.caption("Bold: the axis on which each product is best represented.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_product_characterization(df: pd.DataFrame) -> None:
    st.header("Product characterization")
    st.write("Finds the descriptors that discriminate the products and what characterizes each product "
             "(ANOVA per descriptor with product and assessor effects, then a PCA of the adjusted means).")
    design = design_inputs(df, "characterization")
    if design is None:
        return
    c1, c2 = st.columns(2)
    with c1:
        alpha = select_significance_level()
    confidence = c2.number_input("Confidence interval (%)", 50.0, 99.9, 95.0, step=1.0, key="characterization_ci") / 100
    data = df[design.factors + list(design.descriptors)]
    if not _need(data[design.product].nunique() >= 2, "Need at least two products."):
        return
    try:
        with st.spinner("Fitting one model per descriptor…"):
            res = _compute(data, design, confidence)
    except ValueError as exc:
        st.error(str(exc))
        return
    if not design.assessor:
        st.warning("No assessor column selected, so the model is Y = P and differences between assessors stay in the error. "
                   "Select the assessor column to match XLSTAT's default model (Y = P + A).")

    tabs = st.tabs(["Summary", "Discriminating power", "Coefficients", "Adjusted means", "Sensory map"])
    with tabs[0]:
        _summary(data, design)
    with tabs[1]:
        _power(res, design, alpha)
    with tabs[2]:
        _coefficients(res, alpha, confidence)
    pca = cs.profile_pca(res["means"])
    with tabs[3]:
        _means(res, pca, alpha)
    with tabs[4]:
        _pca(res, design, pca)

