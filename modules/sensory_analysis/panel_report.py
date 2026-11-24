"""Full panel analysis report, laid out like 's "Panel analysis" sheet.

All numbers come from ``panel_stats``; this module
only handles inputs, caching and display.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

from . import panel_stats as ps
from ._shared import heatmap_height

_NONE = "— none —"


# --------------------------------------------------------------------------- #
# Cached computation
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, max_entries=10)
def _compute(data: pd.DataFrame, assessor: str, product: str, session: str | None, descriptors: tuple[str, ...], filter_descriptors: bool) -> dict:
    design = ps.PanelDesign(assessor, product, descriptors, session)
    anovas = ps.anova_all(data, design)
    effect = ps.product_effect(anovas)
    kept = descriptors
    dropped: list[str] = []
    if filter_descriptors:
        keep = ps.discriminating_descriptors(anovas)
        dropped = [d for d in descriptors if d not in keep]
        kept = tuple(d for d in descriptors if d in keep)
    result = {"anovas": anovas, "effect": effect, "kept": kept, "dropped": dropped}
    if not kept:
        return result
    design = ps.PanelDesign(assessor, product, kept, session)
    anovas_kept = {d: anovas[d] for d in kept}
    tests = ps.assessor_tests(data, design, anovas_kept)
    cap, rank_f = ps.cap_table(data, design, anovas_kept, tests)
    means = ps.assessor_product_means(data, design)
    pca_corr, pca_ratio = ps.pca_correlations(ps.product_by_assessor_descriptor(means))
    mfa_corr, mfa_ratio = ps.mfa_correlations(means)
    result.update({
        "tests": tests,
        "cap": cap,
        "rank_f": rank_f,
        "counts": ps.significant_counts(tests),
        "score": ps.performance_score(tests),
        "means": means,
        "means_wide": ps.means_wide(means),
        "distance": ps.distance_to_consensus(means),
        "repeatability": ps.session_repeatability(data, design),
        "max_range": ps.max_session_range(data, design),
        "pca": (pca_corr, pca_ratio),
        "mfa": (mfa_corr, mfa_ratio),
        "cluster": ps.assessor_clustering(means, standardize=False),
        "cluster_std": ps.assessor_clustering(means, standardize=True),
        "kendall": ps.kendall_w_all(means),
        "usage": ps.scale_usage(data, design),
        "within_sd": ps.within_session_sd(data, design),
    })
    return result


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def _guess(df: pd.DataFrame, patterns: list[str], fallback: int) -> int:
    """Index of the first column matching the patterns (in priority order), skipping ID-like columns."""
    columns = df.columns.astype(str).tolist()
    n_rows = len(df)
    for pattern in patterns:
        for i, col in enumerate(columns):
            if re.search(pattern, col, flags=re.IGNORECASE) and df.iloc[:, i].nunique() < n_rows:
                return i
    return min(fallback, len(columns) - 1)


def _fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def _bold_if(condition):
    return lambda v: "font-weight: 700" if condition(v) else ""


def _table_heatmap(table: pd.DataFrame, title: str, colorscale: str, fmt: str = ".2f",
                   zmid: float | None = None, zrange: tuple[float, float] | None = None) -> None:
    """Annotated heatmap sized to the page (rows = assessors)."""
    frame = table.copy()
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    kwargs = {"zmin": zrange[0], "zmax": zrange[1]} if zrange else {}
    fig = px.imshow(frame, text_auto=fmt, aspect="auto", color_continuous_scale=colorscale,
                    color_continuous_midpoint=zmid, **kwargs)
    fig.update_xaxes(type="category", side="bottom")
    fig.update_yaxes(type="category")
    fig.update_layout(title=title, height=heatmap_height(frame.shape[0]), margin={"l": 10, "r": 10, "t": 50, "b": 10})
    st.plotly_chart(fig, width="stretch")


def _cap_colour(value) -> str:
    if not isinstance(value, str) or not value:
        return ""
    colour = {"+": "rgba(46,160,67,0.25)", "-": "rgba(218,54,51,0.25)", "=": "rgba(140,140,140,0.18)"}.get(value[0], "")
    style = f"background-color: {colour}; text-align: center"
    return style + ("; font-weight: 700" if "!" in value else "")


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def _summary(data: pd.DataFrame, assessor: str, product: str, session: str | None, descriptors: list[str]) -> None:
    st.subheader("Summary statistics (quantitative data)")
    quant = ps.summary_quantitative(data, descriptors)
    st.dataframe(quant.style.format(precision=3), hide_index=True, width="stretch")
    st.subheader("Summary statistics (qualitative data)")
    factors = [c for c in (product, assessor, session) if c]
    qual = ps.summary_qualitative(data, factors)
    st.dataframe(qual.style.format({"%": "{:.2f}"}), hide_index=True, width="stretch", height=300)


def _anova(res: dict, session: str | None) -> None:
    sources = next(iter(res["anovas"].values()))["Source"].tolist()
    has_interaction = "Product*Assessor" in sources
    model = "Y = P + A" + (" + S" if session else "") + (" + P*A" if has_interaction else "")
    tested = ("Product and Assessor are tested against the Product*Assessor interaction; Session and the interaction against the error."
              if has_interaction else "All effects are tested against the error.")
    st.caption(f"Model: {model}. Product fixed, the other effects random. Type III sums of squares. {tested}")
    for descriptor, table in res["anovas"].items():
        with st.expander(f"ANOVA with {descriptor} as dependent variable", expanded=False):
            show = table.drop(columns="Denominator")
            st.dataframe(
                show.style.format({"Sum of squares": "{:.3f}", "Mean squares": "{:.3f}", "F": "{:.3f}", "Pr > F": _fmt_p}, na_rep="")
                .map(_bold_if(lambda v: isinstance(v, float) and v < ps.ALPHA), subset=["Pr > F"]),
                hide_index=True, width="stretch",
            )
    st.subheader("Product effect for each descriptor")
    effect = res["effect"].sort_values("F", ascending=False)
    st.dataframe(
        effect.style.format({"F": "{:.3f}", "Pr > F": _fmt_p})
        .map(_bold_if(lambda v: isinstance(v, float) and v < ps.ALPHA), subset=["Pr > F"]),
        hide_index=True, width="stretch",
    )
    fig = px.bar(effect, x="Descriptors", y="F", title="Product effect (F) by descriptor")
    st.plotly_chart(fig, width="stretch")
    if res["dropped"]:
        st.warning(f"Non-discriminating descriptors (product p ≥ {ps.FILTER_ALPHA}) left out of the next analyses: {', '.join(res['dropped'])}.")


def _cap(res: dict) -> None:
    st.subheader("CAP table (Control of Assessor Performances)")
    cap = res["cap"].copy()
    # Display-only table: everything as text so the RankF row can sit under the codes
    num_cols = ["FProd", "FInter", "Mean", "RMSE"]
    for col in num_cols:
        cap[col] = cap[col].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "")
    rank_row = pd.DataFrame([[""] * 4 + [f"{v:.2f}" for v in res["rank_f"]]], columns=cap.columns, index=["RankF"])
    full = pd.concat([cap, rank_row])
    assessor_cols = [c for c in full.columns if c not in num_cols]
    styler = full.style.map(_cap_colour, subset=pd.IndexSlice[cap.index, assessor_cols])
    st.dataframe(styler, width="stretch")
    st.markdown(
        "- **FProd / FInter**: F of the product effect and of the Product*Assessor interaction in the model\n"
        "- **Mean**: average rating; **RMSE**: square root of the model's residual mean square\n"
        "- **RankF**: average rank of the assessor's product F across descriptors (lower = more discriminating)\n"
        "- Assessors and descriptors are sorted by discriminating power (RankF and FProd)\n"
        "- **+** discriminating and agreeing with the panel · **−** discriminating but disagreeing · "
        "**=** not discriminating · **!** not repeatable"
    )
    st.caption(f"Discriminating: per-assessor product effect p < {ps.ALPHA}. Disagreeing: the assessor's share of the interaction "
               f"is significant at p < {ps.AGREEMENT_ALPHA}. Not repeatable: per-assessor session effect, or residual variance "
               f"larger than the panel's, significant at p < {ps.ALPHA}.")


def _plots(data: pd.DataFrame, assessor: str, product: str, descriptors: list[str], means: pd.DataFrame) -> None:
    descriptor = st.selectbox("Descriptor", descriptors, key="panel_report_plot_descriptor")
    by = st.radio("Group by", [product, assessor], horizontal=True, key="panel_report_plot_by")
    ordered = sorted(data[by].astype(str).unique(), key=lambda v: (float(v) if v.replace(".", "", 1).isdigit() else np.inf, v))
    plot_data = data.assign(**{by: data[by].astype(str)})
    st.subheader("Box plots")
    fig = px.box(plot_data, x=by, y=descriptor, category_orders={by: ordered}, points=False)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Strip plots")
    other = assessor if by == product else product
    fig = px.strip(plot_data.assign(**{other: data[other].astype(str)}), x=by, y=descriptor, color=other, category_orders={by: ordered})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Each assessor vs the panel average")
    st.caption("One line per assessor (mean over sessions) and the panel average in black. "
               "Lines that cross the black line often or sit far from it show assessors who rank or score the products differently. "
               "Click names in the legend to hide or isolate assessors.")
    profile = means[descriptor].unstack(level=0)  # products x assessors
    profile = profile.reindex(columns=list(means.index.get_level_values(0).unique()))
    products = [str(p) for p in profile.index]
    fig = go.Figure()
    for who in profile.columns:
        fig.add_trace(go.Scatter(x=products, y=profile[who], mode="lines+markers", name=str(who), opacity=0.55,
                                 line={"width": 1.5}, marker={"size": 5}))
    fig.add_trace(go.Scatter(x=products, y=profile.mean(axis=1), mode="lines+markers", name="Panel average",
                             line={"color": "black", "width": 4}, marker={"size": 8}))
    fig.update_layout(xaxis={"title": product, "type": "category"}, yaxis_title=descriptor, height=480, hovermode="x unified",
                      legend_title_text=assessor)
    st.plotly_chart(fig, width="stretch")


def _unit_circle(fig: go.Figure) -> None:
    t = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line={"color": "grey", "width": 1}, hoverinfo="skip", showlegend=False))
    fig.add_hline(y=0, line_width=0.5, line_color="grey")
    fig.add_vline(x=0, line_width=0.5, line_color="grey")
    fig.update_layout(xaxis={"range": [-1.1, 1.1], "scaleanchor": "y", "constrain": "domain"}, yaxis={"range": [-1.1, 1.1], "constrain": "domain"},
                      width=680, height=600)


def _multivariate(res: dict, descriptors: list[str]) -> None:
    corr, ratio = res["pca"]
    st.subheader("Correlation circle of the PCA on all the descriptors")
    st.caption("PCA on products × (assessor, descriptor) means. Red: the selected descriptor for every assessor. Grey: all other descriptors.")
    descriptor = st.selectbox("Descriptor to highlight", descriptors, key="panel_report_pca_descriptor")
    frame = corr.reset_index()
    frame.columns = ["Assessor", "Descriptor", "F1", "F2"]
    selected = frame["Descriptor"] == descriptor
    fig = go.Figure()
    _unit_circle(fig)
    fig.add_trace(go.Scatter(x=frame.loc[~selected, "F1"], y=frame.loc[~selected, "F2"], mode="markers",
                             marker={"color": "rgba(80,80,80,0.35)", "size": 6}, name="Other descriptors",
                             text=frame.loc[~selected, "Assessor"] + " | " + frame.loc[~selected, "Descriptor"], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=frame.loc[selected, "F1"], y=frame.loc[selected, "F2"], mode="markers+text",
                             marker={"color": "#d62728", "size": 9}, name=descriptor, text=frame.loc[selected, "Assessor"],
                             textposition="top center"))
    fig.update_layout(xaxis_title=f"F1 ({ratio[0]:.1%})", yaxis_title=f"F2 ({ratio[1]:.1%})")
    st.plotly_chart(fig, width="content")

    st.subheader("% of variance plots")
    st.caption("For each assessor, the share of this descriptor's variance carried by the first axis (dark) and the second axis (light).")
    share = frame[selected].assign(**{"F1 %": lambda f: 100 * f["F1"] ** 2, "F2 %": lambda f: 100 * f["F2"] ** 2})
    fig = go.Figure([
        go.Bar(x=share["Assessor"], y=share["F1 %"], name="F1", marker_color="#555555"),
        go.Bar(x=share["Assessor"], y=share["F2 %"], name="F2", marker_color="#bbbbbb"),
    ])
    fig.update_layout(barmode="stack", yaxis={"title": "% of variance", "range": [0, 100]}, xaxis_title="Assessor",
                      xaxis={"type": "category"})
    st.plotly_chart(fig, width="stretch")

    st.subheader("MFA plot of the descriptors")
    st.caption("Multiple Factor Analysis with one group per assessor. Each point is one assessor's descriptor; tight clusters per colour mean the panel agrees on that descriptor.")
    mcorr, mratio = res["mfa"]
    mframe = mcorr.reset_index()
    mframe.columns = ["Assessor", "Descriptor", "F1", "F2"]
    fig = px.scatter(mframe, x="F1", y="F2", color="Descriptor", hover_data=["Assessor"])
    _unit_circle(fig)
    fig.update_layout(xaxis_title=f"F1 ({mratio[0]:.1%})", yaxis_title=f"F2 ({mratio[1]:.1%})")
    st.plotly_chart(fig, width="content")


def _assessors(res: dict) -> None:
    tests = res["tests"]
    st.subheader("Ability to discriminate products")
    st.caption("Per-assessor ANOVA (Product + Session). Bold: the assessor discriminates the products for that descriptor (p < 0.05).")
    assessors = list(res["score"].columns)
    who = st.selectbox("Assessor", assessors, key="panel_report_assessor")
    table = ps.ability_to_discriminate(tests, who)
    sig_cols = [c for c in table.columns if table.loc["Pr > F", c] < ps.ALPHA]
    styler = table.style.format(lambda v: _fmt_p(v) if v < 0.01 else f"{v:.3f}", subset=pd.IndexSlice[["Pr > F"], :]) \
        .format("{:.3f}", subset=pd.IndexSlice[["MSE", "F"], :]) \
        .map(lambda _: "font-weight: 700", subset=pd.IndexSlice[:, sig_cols])
    st.dataframe(styler, width="stretch")

    st.subheader(f"Number of descriptors with p < {ps.ALPHA} for the product effect")
    counts = res["counts"]
    st.dataframe(counts.style.format({"%": "{:.0%}"}, na_rep="").format(precision=0, subset=[c for c in counts.columns if c != "%"]),
                 width="stretch")

    st.subheader("Assessors' performance score based on the ANOVA model")
    st.caption("Number of descriptors on which each assessor discriminates the products, is repeatable, and shows no significant interaction. Higher is better.")
    score = res["score"]
    st.dataframe(score.style.background_gradient(cmap="Greens", axis=1, subset=pd.IndexSlice[["Total"], :]), width="stretch")

    st.subheader("Scale usage")
    st.caption("How each assessor uses the scale. **Level vs panel**: the assessor's mean minus the panel mean "
               "(red = scores higher than the panel, blue = lower). **SD**: spread of the assessor's ratings "
               "(low = uses only a small part of the scale).")
    usage = res["usage"]
    assessors_order = list(res["score"].columns)
    descriptors = list(res["kept"])
    level = ps.scale_usage_pivot(usage, "Level vs panel", assessors_order, descriptors)
    bound = float(np.nanmax(np.abs(level.to_numpy()))) or 1.0
    _table_heatmap(level, "Level vs panel (assessor mean − panel mean)", "RdBu_r", zmid=0, zrange=(-bound, bound))
    _table_heatmap(ps.scale_usage_pivot(usage, "SD", assessors_order, descriptors), "Spread of ratings (SD)", "Purples")
    with st.expander("Full scale-usage table"):
        st.dataframe(usage.style.format({c: "{:.2f}" for c in ["Mean", "SD", "Min", "Max", "Level vs panel", "Range"]}),
                     hide_index=True, width="stretch")


def _agreement(res: dict, descriptors: list[str]) -> None:
    st.subheader("Kendall's coefficient of concordance (W)")
    st.caption("Do the assessors rank the products in the same order? Computed on each assessor's product means. "
               "W = 1: identical rankings; W = 0: no agreement. Bold: agreement significantly above chance (p < 0.05).")
    kendall = res["kendall"]
    st.dataframe(
        kendall.style.format({"W": "{:.3f}", "Chi-square": "{:.2f}", "Pr > Chi²": _fmt_p})
        .map(_bold_if(lambda v: isinstance(v, float) and v < ps.ALPHA), subset=["Pr > Chi²"]),
        hide_index=True, width="stretch",
    )
    fig = px.bar(kendall.sort_values("W", ascending=False), x="Descriptor", y="W", range_y=[0, 1],
                 title="Panel agreement by descriptor")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Correlation between assessors (Spearman)")
    st.caption("How similarly each pair of assessors ranks the products for one descriptor. "
               "An assessor with mostly low or negative correlations disagrees with the rest of the panel.")
    descriptor = st.selectbox("Descriptor", descriptors, key="panel_report_agreement_descriptor")
    corr = ps.assessor_correlations(res["means"], descriptor)
    _table_heatmap(corr, f"Spearman correlation between assessors: {descriptor}", "RdBu_r", zmid=0, zrange=(-1, 1))
    others = corr.where(~np.eye(len(corr), dtype=bool)).mean().sort_values()
    st.markdown("**Mean correlation with the other assessors** (lowest first)")
    st.dataframe(others.to_frame("Mean ρ").T.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=1, vmin=-1, vmax=1),
                 width="stretch")


def _means(res: dict) -> None:
    means = res["means"]
    st.subheader("Means for each assessor, product and descriptor")
    st.dataframe(res["means_wide"].style.format("{:.2f}"), width="stretch")

    st.subheader("Comparison of assessors for each product")
    products = list(means.index.get_level_values(1).unique())
    prod = st.selectbox("Product", products, key="panel_report_product")
    st.dataframe(ps.product_comparison(means, prod).style.format("{:.2f}"), width="stretch")

    st.subheader("Distance to consensus computed across descriptors")
    st.caption("Root mean square difference between the assessor's profile and the panel mean for each product. "
               "Blue: the closest assessor for a product (minimum). Red: the farthest (maximum).")
    dist = res["distance"]
    full = pd.concat([dist, dist.min().to_frame("Min").T, dist.max().to_frame("Max").T])

    def colour(col: pd.Series) -> list[str]:
        lo, hi = dist[col.name].min(), dist[col.name].max()
        out = []
        for idx, v in col.items():
            if idx in ("Min", "Max"):
                out.append("font-weight: 700")
            elif v == lo:
                out.append("background-color: rgba(31,119,180,0.35)")
            elif v == hi:
                out.append("background-color: rgba(214,39,40,0.35)")
            else:
                out.append("")
        return out

    st.dataframe(full.style.format("{:.3f}").apply(colour, axis=0), width="stretch")


def _sessions(res: dict, session: str | None) -> None:
    if res.get("within_sd") is not None:
        st.subheader("Spread between repeated ratings (mean within-product SD)")
        st.caption("For each assessor and descriptor, the standard deviation between the repeated ratings of the same product, "
                   "averaged over products. It shows how big the inconsistency is, in scale units: lower is more consistent.")
        _table_heatmap(res["within_sd"], "Lower SD = more consistent", "Oranges")
    if not session or res["repeatability"] is None:
        st.info("Select a Session column to see the session tests.")
        return
    rep = res["repeatability"]
    st.subheader("Session repeatability: Friedman test (or Wilcoxon if there are only 2 sessions)")
    st.caption("p-values per assessor and descriptor, products as blocks. Bold: significant difference between sessions (p < 0.05), i.e. the assessor is not repeatable.")
    st.dataframe(rep.style.format(_fmt_p).map(_bold_if(lambda v: v < ps.ALPHA)), width="stretch")
    st.subheader("Maximum range between sessions across products")
    st.caption("For each assessor and descriptor, the largest difference between sessions for the same product.")
    st.dataframe(res["max_range"].style.format("{:.2f}").background_gradient(cmap="Oranges", axis=None), width="stretch")


def _clustering(res: dict) -> None:
    for key, title in (("cluster", "Clustering of the assessors (unstandardized)"), ("cluster_std", "Clustering of the assessors (standardized)")):
        st.subheader(title)
        link, labels = res[key]
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(link, labels=labels, ax=ax, color_threshold=None)
        ax.set_ylabel("Dissimilarity (Ward)")
        ax.set_xlabel("Assessor")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("Ward agglomerative clustering on each assessor's product × descriptor means. Assessors joined low in the tree rate products in a similar way.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_panel_report(df: pd.DataFrame) -> None:
    st.header("📑 Panel Analysis Report")
    st.write("A complete panel performance report *Panel analysis*: "
             "mixed-model ANOVA, CAP table, assessor performance, consensus, session repeatability and multivariate views.")

    columns = df.columns.astype(str).tolist()
    numeric = df.select_dtypes(include=np.number).columns.astype(str).tolist()
    if len(columns) < 3:
        st.error("Need at least an assessor column, a product column and one descriptor.")
        return

    c1, c2, c3 = st.columns(3)
    assessor = c1.selectbox("Assessors", columns, index=_guess(df, [r"assess", r"panel|judge|subject|taster|consumer"], 0), key="panel_report_assessor_col")
    product = c2.selectbox("Products", columns, index=_guess(df, [r"product", r"sample|prod"], 1), key="panel_report_product_col")
    session_options = [_NONE] + columns
    session_index = next((i + 1 for i, c in enumerate(columns) if re.search(r"session|replicat|repetition|^rep$", c, re.IGNORECASE)), 0)
    session_choice = c3.selectbox("Sessions (optional)", session_options, index=session_index, key="panel_report_session_col")
    session = None if session_choice == _NONE else session_choice

    used = {assessor, product, session}
    candidates = [c for c in numeric if c not in used]
    descriptors = st.multiselect("Descriptors", candidates, default=candidates, key="panel_report_descriptors")
    filter_descriptors = st.checkbox(f"Filter out non-discriminating descriptors (product p ≥ {ps.FILTER_ALPHA})", value=True,
                                     key="panel_report_filter")

    if len({assessor, product} | ({session} if session else set())) < (3 if session else 2):
        st.error("Assessor, product and session must be different columns.")
        return
    if not descriptors:
        st.info("Select at least one descriptor.")
        return

    data = df[[c for c in (assessor, product, session) if c] + descriptors].dropna(subset=[assessor, product]).copy()
    data.columns = data.columns.astype(str)
    for factor in (assessor, product, session):
        if factor:
            data[factor] = data[factor].astype(str)
    if data[assessor].nunique() < 2 or data[product].nunique() < 2:
        st.error("Need at least two assessors and two products.")
        return
    design_check = ps.PanelDesign(assessor, product, tuple(descriptors), session)
    if not ps.interaction_estimable(data, design_check):
        st.warning("The Product*Assessor interaction can't be estimated: either some assessors did not rate every product, "
                   "or no product was rated more than once by the same assessor. Results use the model without the interaction, "
                   "so the agreement (+/−) codes and the 'No interaction' score are not available.")

    try:
        with st.spinner("Running the panel analysis…"):
            res = _compute(data, assessor, product, session, tuple(descriptors), filter_descriptors)
    except Exception as exc:  # surface model errors instead of crashing the page
        st.error(f"❌ Could not run the analysis: {exc}")
        return

    tabs = st.tabs(["📋 Summary", "📊 ANOVA", "🎯 CAP table", "📦 Box & strip plots", "🧭 PCA & MFA",
                    "👤 Assessors", "🤝 Agreement", "📐 Means & consensus", "🔁 Sessions", "🌳 Clustering"])
    with tabs[0]:
        _summary(data, assessor, product, session, descriptors)
    with tabs[1]:
        _anova(res, session)
    if not res["kept"]:
        for tab in tabs[2:]:
            with tab:
                st.warning("No descriptor discriminates the products. Untick the filter to analyse them anyway.")
        return
    kept = list(res["kept"])
    with tabs[2]:
        _cap(res)
    with tabs[3]:
        _plots(data, assessor, product, kept, res["means"])
    with tabs[4]:
        _multivariate(res, kept)
    with tabs[5]:
        _assessors(res)
    with tabs[6]:
        _agreement(res, kept)
    with tabs[7]:
        _means(res)
    with tabs[8]:
        _sessions(res, session)
    with tabs[9]:
        _clustering(res)
