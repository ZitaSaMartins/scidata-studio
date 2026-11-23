""""
Panel Performance Analysis Module - Complete Version
Complete sensory panel analysis with professional summary statistics tables.
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from ._shared import _columns, _heatmap, _need


# ============================================================
# FORMATTING HELPERS
# ============================================================

def format_p_value(p: float) -> str:
    """Format p-value for display"""
    if np.isnan(p) or p is None:
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def _melt_value_name(df: pd.DataFrame, *reserved: str) -> str:
    """Return a column name that cannot collide with existing columns."""
    name = "_rating"
    existing = set(df.columns.astype(str)) | set(reserved)
    while name in existing:
        name = f"_{name}"
    return name


# ============================================================
# HELPER: Create Summary Statistics Tables
# ============================================================

def _create_anova_summary(df: pd.DataFrame, panelist: str, product: str, 
                          replicate: str, attributes: list) -> pd.DataFrame:
    """
    Create a comprehensive ANOVA summary table (like XLSTAT output)
    Shows F-values and p-values for each descriptor
    """
    summary_rows = []
    
    for attribute in attributes:
        # Remove NaN values
        clean_data = df[[panelist, product, attribute]].dropna()
        
        if clean_data.empty:
            continue
        
        # One-way ANOVA: Product effect for this attribute
        samples = [group[attribute].values 
                  for _, group in clean_data.groupby(product)]
        
        if len(samples) >= 2:
            f_stat, p_val = stats.f_oneway(*samples)
            summary_rows.append({
                "Descriptor": attribute,
                "F": f_stat,
                "p-value": p_val,
                "Significant (α=0.05)": "✓ Yes" if p_val < 0.05 else "✗ No"
            })
    
    return pd.DataFrame(summary_rows).sort_values("F", ascending=False)


def _create_discrimination_anova_detail(df: pd.DataFrame, panelist: str, product: str,
                                        replicate: str, attribute: str) -> dict:
    """
    Create detailed ANOVA breakdown (Source, DF, Sum Sq, Mean Sq, F, Pr>F)
    Similar to XLSTAT Mixed Model Type III output
    """
    clean_data = df[[panelist, product, attribute]].dropna()
    
    if clean_data.empty:
        return None
    
    # Calculate ANOVA components
    grand_mean = clean_data[attribute].mean()
    
    # Total
    ss_total = ((clean_data[attribute] - grand_mean) ** 2).sum()
    df_total = len(clean_data) - 1
    
    # Product (Fixed effect)
    product_groups = clean_data.groupby(product)[attribute]
    ss_product = sum(len(group) * (group.mean() - grand_mean) ** 2 
                     for _, group in product_groups)
    df_product = clean_data[product].nunique() - 1
    ms_product = ss_product / df_product if df_product > 0 else 0
    
    # Assessor (Random effect)
    assessor_groups = clean_data.groupby(panelist)[attribute]
    ss_assessor = sum(len(group) * (group.mean() - grand_mean) ** 2 
                      for _, group in assessor_groups)
    df_assessor = clean_data[panelist].nunique() - 1
    ms_assessor = ss_assessor / df_assessor if df_assessor > 0 else 0
    
    # Interaction or Residual
    ss_error = ss_total - ss_product - ss_assessor
    df_error = df_total - df_product - df_assessor
    ms_error = ss_error / df_error if df_error > 0 else 0
    
    # F-values
    f_product = ms_product / ms_error if ms_error > 0 else 0
    f_assessor = ms_assessor / ms_error if ms_error > 0 else 0
    
    p_product = 1 - stats.f.cdf(f_product, df_product, df_error) if f_product > 0 else 1
    p_assessor = 1 - stats.f.cdf(f_assessor, df_assessor, df_error) if f_assessor > 0 else 1
    
    return pd.DataFrame({
        "Source": ["Product", "Assessor", "Error", "Total"],
        "Type": ["Fixed", "Random", "", ""],
        "DF": [df_product, df_assessor, df_error, df_total],
        "Sum of Squares": [f"{ss_product:.2f}", f"{ss_assessor:.2f}", f"{ss_error:.2f}", f"{ss_total:.2f}"],
        "Mean Squares": [f"{ms_product:.2f}", f"{ms_assessor:.2f}", f"{ms_error:.2f}", ""],
        "F": [f"{f_product:.3f}" if f_product > 0 else "", f"{f_assessor:.3f}" if f_assessor > 0 else "", "", ""],
        "Pr > F": [format_p_value(p_product), format_p_value(p_assessor), "", ""]
    })


# ============================================================
# 1. DISCRIMINATION with Summary Table
# ============================================================

def _panel_discrimination(df: pd.DataFrame) -> None:
    """Panel discrimination analysis with summary table"""
    st.header("📊 Panel Discrimination")
    st.write("ANOVA to test if products are significantly different for each descriptor.")
    
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 3, "Need assessor, product, and attributes."):
        return
    
    panelist = st.selectbox("Assessor", columns, key="pd_panel")
    product = st.selectbox("Product", [c for c in columns if c != panelist], key="pd_product")
    
    # Try to find replicate column
    possible_replicates = [c for c in columns if c not in {panelist, product}]
    replicate = st.selectbox("Replicate/Session (optional)", ["None"] + possible_replicates, key="pd_replicate")
    if replicate == "None":
        replicate = None
    
    attributes = st.multiselect("Descriptors", [c for c in numeric if c not in {panelist, product, replicate or ""}],
                                default=[c for c in numeric if c not in {panelist, product}][:4], key="pd_attr")
    
    if not attributes:
        st.warning("Select at least one descriptor")
        return
    
    # ========== SUMMARY TABLE ==========
    st.subheader("📋 Product Effect Summary (All Descriptors)")
    summary_df = _create_anova_summary(df, panelist, product, replicate or "", attributes)
    
    # Format for display
    summary_df_display = summary_df.copy()
    summary_df_display["F"] = summary_df_display["F"].apply(lambda x: f"{x:.3f}")
    summary_df_display["p-value"] = summary_df_display["p-value"].apply(format_p_value)
    
    st.dataframe(summary_df_display, use_container_width=True, hide_index=True)
    
    # ========== DETAILED ANOVA BREAKDOWN ==========
    st.subheader("🔬 Detailed ANOVA Breakdown")
    
    selected_descriptor = st.selectbox("Select descriptor for detailed view:", attributes, key="pd_detail")
    
    anova_detail = _create_discrimination_anova_detail(df, panelist, product, replicate or "", selected_descriptor)
    
    if anova_detail is not None:
        st.dataframe(anova_detail, use_container_width=True, hide_index=True)
        st.caption(f"Mixed Model Type III Sum of Squares ANOVA ({selected_descriptor})")
    
    # ========== HEATMAP ==========
    st.subheader("🔥 F-Values by Descriptor")
    if not summary_df.empty:
        heatmap_data = summary_df.set_index("Descriptor")[["F"]].T
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=["F-statistic"],
            text=np.round(heatmap_data.values, 2),
            texttemplate="%.2f",
            colorscale="RdYlGn"
        ))
        fig.update_layout(title="F-Statistics Heatmap", height=250)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 2. REPEATABILITY with Summary Statistics
# ============================================================

def _panel_repeatability(df: pd.DataFrame) -> None:
    """Panel repeatability analysis with comprehensive statistics"""
    st.header("🔄 Panel Repeatability")
    st.write("Consistency of assessors across replicates.")
    
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(len(columns) >= 4 and bool(numeric), "Need assessor, product, replicate, and ratings."):
        return
    
    panelist = st.selectbox("Assessor", columns, key="pr_panel")
    product = st.selectbox("Product", [c for c in columns if c != panelist], key="pr_product")
    replicate = st.selectbox("Replicate", [c for c in columns if c not in {panelist, product}], key="pr_rep")
    attributes = st.multiselect("Descriptors", [c for c in numeric if c not in {panelist, product, replicate}],
                                default=[c for c in numeric if c not in {panelist, product, replicate}][:4], key="pr_attr")
    
    rating_col = "_rating"
    long = df.melt(id_vars=[panelist, product, replicate], value_vars=attributes,
                   var_name="Descriptor", value_name=rating_col).dropna()
    
    within = long.groupby([panelist, product, "Descriptor"], observed=True)[rating_col].agg(
        ["std", "count", "mean"]
    ).reset_index()
    within = within[within["count"] > 1]
    
    if not _need(not within.empty, "No repeated ratings found."):
        return
    
    # ========== SUMMARY TABLE ==========
    st.subheader("📋 Repeatability Summary (Standard Deviation)")
    
    metrics = within.groupby([panelist, "Descriptor"], observed=True)["std"].agg(
        ["mean", "median", "std", "min", "max"]
    ).reset_index()
    metrics.columns = [panelist, "Descriptor", "Mean SD", "Median SD", "Std of SD", "Min SD", "Max SD"]
    
    # Overall statistics
    overall_stats = {
        "Mean SD (All)": within["std"].mean(),
        "Median SD (All)": within["std"].median(),
        "Best Assessor": metrics.loc[metrics["Mean SD"].idxmin(), panelist],
        "Worst Assessor": metrics.loc[metrics["Mean SD"].idxmax(), panelist],
    }
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Panel Mean SD", f"{overall_stats['Mean SD (All)']:.3f}")
    with col2:
        st.metric("Panel Median SD", f"{overall_stats['Median SD (All)']:.3f}")
    with col3:
        st.metric("Best Assessor", overall_stats['Best Assessor'])
    with col4:
        st.metric("Worst Assessor", overall_stats['Worst Assessor'])
    
    st.subheader("By Assessor and Descriptor")
    display_metrics = metrics.copy()
    for col in ["Mean SD", "Median SD", "Std of SD", "Min SD", "Max SD"]:
        display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.3f}")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)
    
    # ========== HEATMAP ==========
    st.subheader("🎯 Repeatability Heatmap (Mean Within-SD)")
    heatmap_pivot = metrics.pivot(index=panelist, columns="Descriptor", values="Mean SD")
    _heatmap(heatmap_pivot, "Lower SD = More Consistent")


# ============================================================
# 3. AGREEMENT with Summary Table
# ============================================================

def _panel_agreement(df: pd.DataFrame) -> None:
    """Panel agreement analysis with Kendall's W summary"""
    st.header("🤝 Panel Agreement")
    st.write("Kendall's W test: How similarly do assessors order products?")
    
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 3, "Need assessor, product, and rating."):
        return
    
    panelist = st.selectbox("Assessor", columns, key="pa_panel")
    product = st.selectbox("Product", [c for c in columns if c != panelist], key="pa_product")
    rating_candidates = [c for c in numeric if c not in {panelist, product}]
    
    if not _need(bool(rating_candidates), "Need a numeric rating column."):
        return
    
    rating = st.selectbox("Rating", rating_candidates, key="pa_rating")
    
    # Get matrix for each descriptor
    matrix = df.pivot_table(index=product, columns=panelist, values=rating, aggfunc="mean").dropna()
    if not _need(matrix.shape[0] >= 3 and matrix.shape[1] >= 2, "Need 3+ products rated by 2+ assessors."):
        return
    
    # Kendall's W calculation
    ranks = matrix.rank(axis=0)
    products, judges = ranks.shape
    rank_sums = ranks.sum(axis=1)
    
    tie_correction = sum(
        sum(count**3 - count for count in column.value_counts())
        for _, column in matrix.items()
    )
    denominator = judges**2 * (products**3 - products) - judges * tie_correction
    w_value = 12 * float(((rank_sums - rank_sums.mean()) ** 2).sum()) / denominator if denominator > 0 else np.nan
    chi_square = judges * (products - 1) * w_value
    p_value = float(stats.chi2.sf(chi_square, products - 1))
    
    # ========== SUMMARY TABLE ==========
    st.subheader("📋 Panel Agreement Summary")
    
    agreement_summary = pd.DataFrame([{
        "Statistic": "Kendall's W",
        "Value": f"{w_value:.4f}",
        "Interpretation": "0=No agreement, 1=Perfect agreement"
    }, {
        "Statistic": "Chi-Square",
        "Value": f"{chi_square:.3f}",
        "Interpretation": f"df={products-1}"
    }, {
        "Statistic": "p-value",
        "Value": format_p_value(p_value),
        "Interpretation": "< 0.05 = Significant agreement"
    }])
    
    st.dataframe(agreement_summary, use_container_width=True, hide_index=True)
    
    # ========== CORRELATION TABLE ==========
    st.subheader("👥 Pairwise Spearman Correlations")
    corr_matrix = matrix.corr(method="spearman")
    
    display_corr = corr_matrix.applymap(lambda x: f"{x:.3f}" if not np.isnan(x) else "N/A")
    st.dataframe(display_corr, use_container_width=True)
    
    _heatmap(corr_matrix, "Correlation Between Assessors")


# ============================================================
# 4. SCALE USAGE with Comparison Tables
# ============================================================

def _panel_scale(df: pd.DataFrame) -> None:
    """Panel scale usage analysis with comprehensive comparison tables"""
    st.header("📏 Panel Scale Usage")
    st.write("How do assessors use the rating scale differently?")
    
    numeric, _ = _columns(df)
    if not _need(bool(numeric) and df.shape[1] >= 2, "Need assessor and ratings."):
        return
    
    panelist = st.selectbox("Assessor", df.columns.astype(str).tolist(), key="ps_panel")
    attributes = st.multiselect("Descriptors", [c for c in numeric if c != panelist],
                                default=[c for c in numeric if c != panelist][:5], key="ps_attr")
    
    rating_col = "_rating"
    long = df.melt(id_vars=panelist, value_vars=attributes, 
                   var_name="Descriptor", value_name=rating_col).dropna()
    
    metrics = long.groupby([panelist, "Descriptor"], observed=True)[rating_col].agg(
        ["count", "mean", "std", "min", "max"]
    ).reset_index()
    
    if not _need(not metrics.empty, "Select at least one descriptor."):
        return
    
    metrics["Range"] = metrics["max"] - metrics["min"]
    metrics["Centered Mean"] = metrics.groupby("Descriptor")["mean"].transform(lambda x: x - x.mean())
    
    # ========== SUMMARY TABLE ==========
    st.subheader("📋 Scale Usage Statistics by Assessor and Descriptor")
    
    display_metrics = metrics[[panelist, "Descriptor", "count", "mean", "std", "min", "max", "Range", "Centered Mean"]].copy()
    display_metrics.columns = ["Assessor", "Descriptor", "N", "Mean", "SD", "Min", "Max", "Range", "Centered Mean"]
    
    # Format for display
    for col in ["Mean", "SD", "Min", "Max", "Range", "Centered Mean"]:
        display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.2f}")
    
    display_metrics["N"] = display_metrics["N"].astype(int)
    
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)
    
    # ========== COMPARISON HEATMAPS ==========
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Range Used")
        _heatmap(metrics.pivot(index=panelist, columns="Descriptor", values="Range"),
                 "Scale Range (wider = more scale usage)")
    
    with col2:
        st.subheader("Centered Mean")
        _heatmap(metrics.pivot(index=panelist, columns="Descriptor", values="Centered Mean"),
                 "Positive = rates higher than average")
    
    # ========== BOX PLOT ==========
    st.subheader("Distribution by Assessor")
    fig = px.box(long, x=panelist, y=rating_col, color="Descriptor")
    fig.update_layout(height=500, title="Scale Usage Distribution")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 5. DISCRIMINATION POWER
# ============================================================

def _discrimination_power(df: pd.DataFrame) -> None:
    """Discrimination power: which assessors can discriminate which descriptors?"""
    st.header("💪 Discrimination Power by Assessor")
    st.write("Which descriptors can each assessor effectively discriminate?")
    
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 3, "Need assessor, product, and attributes."):
        return
    
    panelist = st.selectbox("Assessor", columns, key="dpower_panel")
    product = st.selectbox("Product", [c for c in columns if c != panelist], key="dpower_product")
    
    attributes = st.multiselect("Descriptors", [c for c in numeric if c not in {panelist, product}],
                                default=[c for c in numeric if c not in {panelist, product}][:4], key="dpower_attr")
    
    if not attributes:
        st.warning("Select at least one descriptor")
        return
    
    # Calculate discrimination ability per assessor
    discrimination_data = []
    
    for attr in attributes:
        clean_data = df[[panelist, product, attr]].dropna()
        
        for assessor in clean_data[panelist].unique():
            assessor_data = clean_data[clean_data[panelist] == assessor]
            
            if len(assessor_data[product].unique()) < 2:
                continue
            
            samples = [group[attr].values for _, group in assessor_data.groupby(product)]
            if len(samples) >= 2:
                f_stat, p_val = stats.f_oneway(*samples)
                discrimination_data.append({
                    "Assessor": assessor,
                    "Descriptor": attr,
                    "F-value": f_stat,
                    "p-value": p_val,
                    "Can Discriminate": "✓ Yes" if p_val < 0.05 else "✗ No"
                })
    
    if discrimination_data:
        disc_df = pd.DataFrame(discrimination_data)
        
        st.subheader("📋 Discrimination Ability per Assessor-Descriptor Pair")
        display_df = disc_df.copy()
        display_df["F-value"] = display_df["F-value"].apply(lambda x: f"{x:.3f}")
        display_df["p-value"] = display_df["p-value"].apply(format_p_value)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Heatmap: Can each assessor discriminate each descriptor?
        st.subheader("🔥 Discrimination Ability Heatmap")
        heatmap_pivot = disc_df.pivot_table(index="Assessor", columns="Descriptor", 
                                             values="F-value", aggfunc="first")
        _heatmap(heatmap_pivot, "F-value (higher = better discrimination)")
        
        # Summary: Best assessors per descriptor
        st.subheader("⭐ Best Discriminators per Descriptor")
        best_summary = disc_df.loc[disc_df.groupby("Descriptor")["F-value"].idxmax()]
        st.dataframe(best_summary[[
            "Descriptor", "Assessor", "F-value", "p-value"
        ]].sort_values("F-value", ascending=False), use_container_width=True, hide_index=True)


# ============================================================
# 6. PRODUCT CHARACTERIZATION
# ============================================================

def _product_characterization(df: pd.DataFrame) -> None:
    """Product characterization: average rating for each product-descriptor pair"""
    st.header("🎯 Product Characterization")
    st.write("Average rating for each product and descriptor.")
    
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 2, "Need product and ratings."):
        return
    
    product = st.selectbox("Product", columns, key="pchar_product")
    
    attributes = st.multiselect("Descriptors", [c for c in numeric if c != product],
                                default=[c for c in numeric if c != product][:6], key="pchar_attr")
    
    if not attributes:
        st.warning("Select at least one descriptor")
        return
    
    # Create characterization table
    char_data = df[[product] + attributes].groupby(product)[attributes].agg(["mean", "std", "count"]).round(3)
    
    st.subheader("📋 Average Ratings by Product")
    
    # Flatten multi-level columns for display
    display_char = pd.DataFrame()
    for attr in attributes:
        display_char[f"{attr} (Mean)"] = df.groupby(product)[attr].mean()
        display_char[f"{attr} (SD)"] = df.groupby(product)[attr].std()
        display_char[f"{attr} (N)"] = df.groupby(product)[attr].count()
    
    st.dataframe(display_char.round(3), use_container_width=True)
    
    # Heatmap: Product profiles
    st.subheader("🔥 Product Profile Heatmap")
    profile_pivot = df.groupby(product)[attributes].mean()
    _heatmap(profile_pivot, "Average rating (normalized)")
    
    # Radar chart for each product (if 3+ products)
    if df[product].nunique() >= 3 and len(attributes) >= 3:
        st.subheader("📡 Product Profiles (Radar Chart)")
        
        fig = go.Figure()
        
        for prod in df[product].unique():
            prod_ratings = df[df[product] == prod][attributes].mean()
            fig.add_trace(go.Scatterpolar(
                r=prod_ratings.values,
                theta=attributes,
                fill='toself',
                name=str(prod)
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, df[attributes].max().max()])),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 7. PRODUCT PROFILES with Assessor Comparison
# ============================================================

def _product_profiles(df: pd.DataFrame) -> None:
    """Product profiles and assessor consensus"""
    st.header("👥 Product Profiles & Assessor Comparison")
    st.write("How do individual assessors compare to the panel average?")
    
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 3, "Need assessor, product, and ratings."):
        return
    
    panelist = st.selectbox("Assessor", columns, key="pp_panel")
    product = st.selectbox("Product", [c for c in columns if c != panelist], key="pp_product")
    
    attributes = st.multiselect("Descriptors", [c for c in numeric if c not in {panelist, product}],
                                default=[c for c in numeric if c not in {panelist, product}][:4], key="pp_attr")
    
    if not attributes:
        st.warning("Select at least one descriptor")
        return
    
    # Comparison plots: individual vs panel average
    st.subheader("📊 Individual vs Panel Average")
    
    products_list = df[product].unique()
    
    for attr in attributes:
        fig = go.Figure()
        
        # Panel average
        panel_avg = df.groupby(product)[attr].mean()
        fig.add_trace(go.Scatter(x=panel_avg.index.astype(str), y=panel_avg.values,
                                 name="Panel Average", mode="lines+markers",
                                 line=dict(color="black", width=3)))
        
        # Each assessor
        for assessor in df[panelist].unique():
            assessor_ratings = df[df[panelist] == assessor].groupby(product)[attr].mean()
            fig.add_trace(go.Scatter(x=assessor_ratings.index.astype(str), y=assessor_ratings.values,
                                     name=str(assessor), mode="lines+markers", opacity=0.6))
        
        fig.update_layout(title=f"{attr} - Individual vs Panel Average",
                         xaxis_title=product, yaxis_title="Rating",
                         height=450, hovermode="x")
        st.plotly_chart(fig, use_container_width=True)
    
    # Distance to consensus
    st.subheader("📏 Distance to Consensus")
    
    consensus_data = []
    
    for prod in products_list:
        prod_data = df[df[product] == prod]
        panel_profile = prod_data[attributes].mean()
        
        for assessor in prod_data[panelist].unique():
            assessor_ratings = prod_data[prod_data[panelist] == assessor][attributes].mean()
            euclidean_dist = np.sqrt(((assessor_ratings - panel_profile) ** 2).sum())
            
            consensus_data.append({
                product: prod,
                "Assessor": assessor,
                "Distance to Consensus": euclidean_dist
            })
    
    consensus_df = pd.DataFrame(consensus_data)
    
    if not consensus_df.empty:
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Distance", f"{consensus_df['Distance to Consensus'].mean():.3f}")
        with col2:
            st.metric("Max Distance", f"{consensus_df['Distance to Consensus'].max():.3f}")
        with col3:
            st.metric("Min Distance", f"{consensus_df['Distance to Consensus'].min():.3f}")
        
        # Detailed table
        st.subheader("📋 Distance to Consensus by Product and Assessor")
        display_consensus = consensus_df.copy()
        display_consensus["Distance to Consensus"] = display_consensus["Distance to Consensus"].apply(lambda x: f"{x:.3f}")
        st.dataframe(display_consensus, use_container_width=True, hide_index=True)
        
        # Heatmap
        pivot_consensus = consensus_df.pivot_table(index="Assessor", 
                                                     columns=product, 
                                                     values="Distance to Consensus")
        _heatmap(pivot_consensus, "Lower = Closer to consensus")


# ============================================================
# MAIN DISPATCHER
# ============================================================

def run_panel_analysis(df: pd.DataFrame) -> None:
    """Main panel analysis dispatcher with all tabs"""
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Discrimination",
        "🔄 Repeatability",
        "🤝 Agreement",
        "📏 Scale Usage",
        "💪 Discrimination Power",
        "🎯 Product Characterization",
        "👥 Assessor Comparison"
    ])
    
    with tab1:
        _panel_discrimination(df)
    
    with tab2:
        _panel_repeatability(df)
    
    with tab3:
        _panel_agreement(df)
    
    with tab4:
        _panel_scale(df)
    
    with tab5:
        _discrimination_power(df)
    
    with tab6:
        _product_characterization(df)
    
    with tab7:
        _product_profiles(df)