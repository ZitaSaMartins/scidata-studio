"""Panel analysis computations modelled on XLSTAT's "Panel analysis" tool.

Pure pandas / statsmodels code (no Streamlit) so every number can be tested
against the reference output in ``tuto_PAN_EN.xls``.

Model per descriptor (XLSTAT default)::

    Y = Product + Assessor + Session + Product*Assessor

Product is fixed, the other effects random. Type III sums of squares. The F
tests use the denominators implied by the expected mean squares:

* Product   -> MS(Product*Assessor)
* Assessor  -> MS(Product*Assessor)
* Session   -> MS(Error)
* Product*Assessor -> MS(Error)

Assessor-level indicators (CAP table, performance score):

* **Discrimination**: per-assessor ANOVA ``Y = Product (+ Session)``, product
  effect significant at ``alpha``.
* **Agreement**: the assessor's share of the Product*Assessor interaction,
  tested against the model's error. Significant at ``AGREEMENT_ALPHA`` ->
  disagreeing assessor ("-").
* **Repeatability**: not repeatable ("!") when the per-assessor session effect
  is significant or the assessor's residual variance is significantly larger
  than the panel's (both at ``alpha``).

On the XLSTAT tutorial data these rules reproduce every discrimination and
agreement code and 87 of the 90 repeatability flags.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.cluster.hierarchy import linkage
from statsmodels.stats.anova import anova_lm

ALPHA = 0.05
AGREEMENT_ALPHA = 0.10
FILTER_ALPHA = 0.10


@dataclass(frozen=True)
class PanelDesign:
    assessor: str
    product: str
    descriptors: tuple[str, ...]
    session: str | None = None

    @property
    def factors(self) -> list[str]:
        return [c for c in (self.product, self.assessor, self.session) if c]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _q(name: str) -> str:
    """Quote a column name for a patsy formula."""
    return 'Q("' + str(name).replace('"', '\\"') + '")'


def _prepare(df: pd.DataFrame, design: PanelDesign) -> pd.DataFrame:
    cols = design.factors + list(design.descriptors)
    data = df[cols].copy()
    for factor in design.factors:
        data[factor] = data[factor].astype(str)
    return data


def _ordered_levels(series: pd.Series) -> list[str]:
    """Factor levels in natural order (numeric labels sorted as numbers)."""
    levels = pd.unique(series.astype(str))
    try:
        return sorted(levels, key=float)
    except ValueError:
        return sorted(levels)


def has_replicates(df: pd.DataFrame, design: PanelDesign) -> bool:
    return bool(df.groupby([design.assessor, design.product]).size().max() > 1)


def interaction_estimable(df: pd.DataFrame, design: PanelDesign) -> bool:
    """Product*Assessor needs every assessor to rate every product, and some ratings repeated."""
    sizes = df.groupby([design.assessor, design.product]).size()
    n_cells = df[design.assessor].nunique() * df[design.product].nunique()
    return len(sizes) == n_cells and bool(sizes.max() > 1)


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #

def summary_quantitative(df: pd.DataFrame, descriptors: list[str]) -> pd.DataFrame:
    rows = []
    for col in descriptors:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "Variable": col,
            "Observations": len(s),
            "Obs. with missing data": int(s.isna().sum()),
            "Obs. without missing data": int(s.notna().sum()),
            "Minimum": s.min(),
            "Maximum": s.max(),
            "Mean": s.mean(),
            "Std. deviation": s.std(),
        })
    return pd.DataFrame(rows)


def summary_qualitative(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    for col in factors:
        counts = df[col].astype(str).value_counts()
        for level in _ordered_levels(df[col]):
            rows.append({
                "Variable": col,
                "Categories": level,
                "Counts": int(counts[level]),
                "%": 100 * counts[level] / counts.sum(),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Mixed-model ANOVA per descriptor
# --------------------------------------------------------------------------- #

def _full_model(data: pd.DataFrame, design: PanelDesign, descriptor: str, interaction: bool):
    terms = [f"C({_q(design.product)}, Sum)", f"C({_q(design.assessor)}, Sum)"]
    if design.session:
        terms.append(f"C({_q(design.session)}, Sum)")
    if interaction:
        terms.append(f"C({_q(design.product)}, Sum):C({_q(design.assessor)}, Sum)")
    return smf.ols(f"{_q(descriptor)} ~ " + " + ".join(terms), data=data.dropna(subset=[descriptor])).fit()


def mixed_anova(df: pd.DataFrame, design: PanelDesign, descriptor: str) -> pd.DataFrame:
    """Type III table: Source, Type, DF, Sum of squares, Mean squares, F, Pr > F."""
    data = _prepare(df, design)
    data = data.dropna(subset=[descriptor])
    interaction = interaction_estimable(data, design)
    model = _full_model(data, design, descriptor, interaction)
    if model.df_resid <= 0:
        raise ValueError(f"Not enough ratings to fit the ANOVA model for '{descriptor}'.")
    table = anova_lm(model, typ=3)

    def row(key_part: str, exclude: str | None = None):
        for idx in table.index:
            if key_part in idx and (exclude is None or exclude not in idx):
                return table.loc[idx]
        raise KeyError(key_part)

    p_key, a_key = f"C({_q(design.product)}, Sum)", f"C({_q(design.assessor)}, Sum)"
    entries = {
        "Product": row(p_key, exclude=":"),
        "Assessor": row(a_key, exclude=":"),
    }
    if design.session:
        entries["Session"] = row(f"C({_q(design.session)}, Sum)")
    if interaction:
        entries["Product*Assessor"] = row(":")
    err = table.loc["Residual"]

    ms = {k: v["sum_sq"] / v["df"] for k, v in entries.items()}
    ms_err = err["sum_sq"] / err["df"]
    denominators = {
        "Product": ("Product*Assessor" if interaction else "Error"),
        "Assessor": ("Product*Assessor" if interaction else "Error"),
        "Session": "Error",
        "Product*Assessor": "Error",
    }
    rows = []
    for source, values in entries.items():
        denom_name = denominators[source]
        if denom_name == "Error":
            denom_ms, denom_df = ms_err, err["df"]
        else:
            denom_ms, denom_df = ms[denom_name], entries[denom_name]["df"]
        f_value = ms[source] / denom_ms if denom_ms > 0 else np.nan
        rows.append({
            "Source": source,
            "Type": "Fixed" if source == "Product" else "Random",
            "DF": int(values["df"]),
            "Sum of squares": values["sum_sq"],
            "Mean squares": ms[source],
            "F": f_value,
            "Pr > F": stats.f.sf(f_value, values["df"], denom_df),
            "Denominator": denom_name,
        })
    rows.append({"Source": "Error", "Type": "", "DF": int(err["df"]), "Sum of squares": err["sum_sq"],
                 "Mean squares": ms_err, "F": np.nan, "Pr > F": np.nan, "Denominator": ""})
    return pd.DataFrame(rows)


def anova_all(df: pd.DataFrame, design: PanelDesign) -> dict[str, pd.DataFrame]:
    return {d: mixed_anova(df, design, d) for d in design.descriptors}


def product_effect(anovas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for descriptor, table in anovas.items():
        p = table.set_index("Source").loc["Product"]
        rows.append({"Descriptors": descriptor, "F": p["F"], "Pr > F": p["Pr > F"]})
    return pd.DataFrame(rows)


def discriminating_descriptors(anovas: dict[str, pd.DataFrame], threshold: float = FILTER_ALPHA) -> list[str]:
    effect = product_effect(anovas)
    return effect.loc[effect["Pr > F"] < threshold, "Descriptors"].tolist()


# --------------------------------------------------------------------------- #
# Assessor-level tests
# --------------------------------------------------------------------------- #

def assessor_tests(df: pd.DataFrame, design: PanelDesign, anovas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (assessor, descriptor) with every indicator used downstream."""
    data = _prepare(df, design)
    assessors = _ordered_levels(data[design.assessor])
    n_products = data[design.product].nunique()
    rows = []
    for descriptor in design.descriptors:
        table = anovas[descriptor].set_index("Source")
        ms_err, df_err = table.loc["Error", "Mean squares"], table.loc["Error", "DF"]
        means = data.groupby([design.assessor, design.product])[descriptor].mean().unstack()
        counts = data.groupby([design.assessor, design.product])[descriptor].count().unstack()
        # Interaction residuals: assessor x product mean minus additive fit
        resid = means.sub(means.mean(axis=1), axis=0).sub(means.mean(axis=0), axis=1) + means.stack().mean()
        for assessor in assessors:
            sub = data[data[design.assessor] == assessor].dropna(subset=[descriptor])
            terms = [f"C({_q(design.product)})"]
            if design.session and sub[design.session].nunique() > 1:
                terms.append(f"C({_q(design.session)})")
            model = smf.ols(f"{_q(descriptor)} ~ " + " + ".join(terms), data=sub).fit()
            mse = model.mse_resid if model.df_resid > 0 else np.nan
            if model.df_resid > 0:
                a_table = anova_lm(model, typ=2)
                f_prod = a_table.iloc[0]["F"]
                p_prod = a_table.iloc[0]["PR(>F)"]
                p_sess = a_table.iloc[1]["PR(>F)"] if len(terms) > 1 else np.nan
            else:
                f_prod = p_prod = p_sess = np.nan

            ss_int = float((counts.loc[assessor] * resid.loc[assessor] ** 2).sum())
            if np.isfinite(ms_err) and ms_err > 0 and "Product*Assessor" in table.index:
                f_int = ss_int / (n_products - 1) / ms_err
                p_int = stats.f.sf(f_int, n_products - 1, df_err)
            else:
                f_int = p_int = np.nan

            p_mse = stats.f.sf(mse / ms_err, model.df_resid, df_err) if np.isfinite(mse) and ms_err > 0 else np.nan

            rows.append({
                "Assessor": assessor, "Descriptor": descriptor,
                "MSE": mse, "F": f_prod, "Pr > F": p_prod,
                "p_session": p_sess, "F_interaction": f_int, "p_interaction": p_int, "p_mse": p_mse,
            })
    out = pd.DataFrame(rows)
    out["discriminating"] = out["Pr > F"] < ALPHA
    out["disagreeing"] = out["discriminating"] & (out["p_interaction"] < AGREEMENT_ALPHA)
    out["not_repeatable"] = (out["p_session"].fillna(1) < ALPHA) | (out["p_mse"].fillna(1) < ALPHA)
    out["interaction_ok"] = ~(out["p_interaction"] < AGREEMENT_ALPHA)
    out["rank_F"] = out.groupby("Descriptor")["F"].rank(ascending=False)
    return out


def ability_to_discriminate(tests: pd.DataFrame, assessor: str) -> pd.DataFrame:
    """XLSTAT layout: rows MSE / F / Pr > F, one column per descriptor."""
    sub = tests[tests["Assessor"] == assessor].set_index("Descriptor")
    return sub[["MSE", "F", "Pr > F"]].T


def significant_counts(tests: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    flags = (tests.assign(sig=(tests["Pr > F"] < alpha).astype(int))
             .pivot(index="Assessor", columns="Descriptor", values="sig"))
    flags = flags.reindex(index=_ordered_levels(tests["Assessor"]), columns=pd.unique(tests["Descriptor"]))
    flags["Sum"] = flags.sum(axis=1)
    flags["%"] = flags["Sum"] / (flags.shape[1] - 1)
    total = flags.drop(columns="%").sum(axis=0)
    flags.loc["Sum"] = list(total) + [np.nan]
    return flags


def performance_score(tests: pd.DataFrame) -> pd.DataFrame:
    g = tests.groupby("Assessor")
    score = pd.DataFrame({
        "Discrimination": g["discriminating"].sum(),
        "Repeatability": g["not_repeatable"].apply(lambda s: int((~s).sum())),
        "No interaction": g["interaction_ok"].sum(),
    }).astype(int)
    if tests["p_interaction"].isna().all():  # interaction not estimable
        score = score.drop(columns="No interaction")
    score["Total"] = score.sum(axis=1)
    return score.reindex(_ordered_levels(tests["Assessor"])).T


def cap_table(df: pd.DataFrame, design: PanelDesign, anovas: dict[str, pd.DataFrame], tests: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Control of Assessor Performances table.

    Returns ``(table, rank_f)``. Rows are descriptors sorted by FProd; the
    first columns are FProd, FInter, Mean, RMSE, then one column of codes per
    assessor sorted by RankF.
    """
    rank_f = tests.groupby("Assessor")["rank_F"].mean().sort_values(kind="stable")
    rows = []
    for descriptor in design.descriptors:
        t = anovas[descriptor].set_index("Source")
        rows.append({
            "Attributes": descriptor,
            "FProd": t.loc["Product", "F"],
            "FInter": t.loc["Product*Assessor", "F"] if "Product*Assessor" in t.index else np.nan,
            "Mean": pd.to_numeric(df[descriptor], errors="coerce").mean(),
            "RMSE": float(np.sqrt(t.loc["Error", "Mean squares"])),
        })
    summary = pd.DataFrame(rows).set_index("Attributes")

    def code(r) -> str:
        symbol = "=" if not r["discriminating"] else ("-" if r["disagreeing"] else "+")
        return symbol + ("!" if r["not_repeatable"] else "")

    codes = tests.assign(code=tests.apply(code, axis=1)).pivot(index="Descriptor", columns="Assessor", values="code")
    table = summary.join(codes[list(rank_f.index)])
    return table.sort_values("FProd", ascending=False), rank_f


# --------------------------------------------------------------------------- #
# Repeatability across sessions
# --------------------------------------------------------------------------- #

def session_repeatability(df: pd.DataFrame, design: PanelDesign) -> pd.DataFrame | None:
    """p-values of Wilcoxon (2 sessions) or Friedman (>2 sessions), blocks = products."""
    if not design.session:
        return None
    data = _prepare(df, design)
    sessions = _ordered_levels(data[design.session])
    if len(sessions) < 2:
        return None
    out = pd.DataFrame(index=_ordered_levels(data[design.assessor]), columns=list(design.descriptors), dtype=float)
    for assessor, sub in data.groupby(design.assessor):
        for descriptor in design.descriptors:
            wide = sub.pivot_table(index=design.product, columns=design.session, values=descriptor).dropna()
            try:
                if len(sessions) == 2:
                    diff = wide.iloc[:, 0] - wide.iloc[:, 1]
                    p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(wide.iloc[:, 0], wide.iloc[:, 1], zero_method="pratt", correction=True).pvalue
                else:
                    p = stats.friedmanchisquare(*[wide[c] for c in wide.columns]).pvalue
            except ValueError:
                p = np.nan
            out.loc[assessor, descriptor] = p
    out.index.name = "Assessor"
    return out


def max_session_range(df: pd.DataFrame, design: PanelDesign) -> pd.DataFrame | None:
    """For each assessor and descriptor, the largest (max - min across sessions) over products."""
    if not design.session:
        return None
    data = _prepare(df, design)
    g = data.groupby([design.assessor, design.product])[list(design.descriptors)]
    ranges = (g.max() - g.min()).groupby(level=0).max()
    return ranges.reindex(_ordered_levels(data[design.assessor]))


# --------------------------------------------------------------------------- #
# Means, consensus
# --------------------------------------------------------------------------- #

def assessor_product_means(df: pd.DataFrame, design: PanelDesign) -> pd.DataFrame:
    """Assessor x product mean for every descriptor (long index: assessor, product)."""
    data = _prepare(df, design)
    means = data.groupby([design.assessor, design.product])[list(design.descriptors)].mean()
    order_a, order_p = _ordered_levels(data[design.assessor]), _ordered_levels(data[design.product])
    return means.reindex(pd.MultiIndex.from_product([order_a, order_p], names=[design.assessor, design.product]))


def means_wide(means: pd.DataFrame) -> pd.DataFrame:
    """XLSTAT 'Means for each judge' layout: rows assessors, columns product|descriptor."""
    descriptors = list(means.columns)
    products = list(means.index.get_level_values(1).unique())
    wide = means.unstack(level=1)  # columns: (descriptor, product)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([descriptors, products]))
    wide.columns = [f"{p}|{d}" for d, p in wide.columns]
    return wide.reindex(_assessor_order(means))


def _assessor_order(means: pd.DataFrame) -> list:
    return list(means.index.get_level_values(0).unique())


def product_comparison(means: pd.DataFrame, product: str) -> pd.DataFrame:
    """Per-product table: rows assessors, columns descriptors, plus an Average row."""
    table = means.xs(product, level=1).copy()
    table.loc["Average"] = table.mean()
    return table


def distance_to_consensus(means: pd.DataFrame) -> pd.DataFrame:
    """Root mean square distance between each assessor's and the panel's product profile."""
    consensus = means.groupby(level=1, sort=False).mean()
    diff = means.sub(consensus, level=1)
    dist = np.sqrt((diff ** 2).mean(axis=1)).unstack()
    return dist.reindex(index=_assessor_order(means), columns=means.index.get_level_values(1).unique())


# --------------------------------------------------------------------------- #
# Multivariate views
# --------------------------------------------------------------------------- #

def product_by_assessor_descriptor(means: pd.DataFrame) -> pd.DataFrame:
    """Products as rows, (assessor, descriptor) as columns."""
    wide = means.unstack(level=0)  # columns: (descriptor, assessor)
    wide.columns = wide.columns.swaplevel(0, 1)
    columns = pd.MultiIndex.from_product([_assessor_order(means), list(means.columns)])
    return wide.reindex(index=means.index.get_level_values(1).unique(), columns=columns)


def pca_correlations(table: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Correlation PCA. Returns variable-factor correlations (F1, F2) and explained-variance ratios."""
    clean = table.loc[:, table.std(ddof=0) > 0].dropna(axis=1)
    z = (clean - clean.mean()) / clean.std(ddof=0)
    u, s, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    eig = s ** 2 / len(z)
    loadings = vt.T * np.sqrt(eig)  # correlations of variables with components
    corr = pd.DataFrame(loadings[:, :2], index=clean.columns, columns=["F1", "F2"])
    return corr, eig / eig.sum()


def mfa_correlations(means: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """MFA with one group per assessor (products x descriptors).

    Each group is standardised and divided by its first singular value, then a
    global PCA is run. Returns each (assessor, descriptor) variable's
    correlation with the first two global factors.
    """
    table = product_by_assessor_descriptor(means)
    blocks = []
    for assessor in table.columns.get_level_values(0).unique():
        block = table[assessor]
        block = block.loc[:, block.std(ddof=0) > 0]
        z = (block - block.mean()) / block.std(ddof=0)
        first = np.linalg.svd(z.to_numpy(), compute_uv=False)[0]
        z = z / first if first else z
        z.columns = pd.MultiIndex.from_product([[assessor], z.columns])
        blocks.append(z)
    global_table = pd.concat(blocks, axis=1)
    u, s, _ = np.linalg.svd(global_table.to_numpy(), full_matrices=False)
    scores = u[:, :2] * s[:2]
    corr = pd.DataFrame(
        {f"F{i + 1}": [np.corrcoef(table[col], scores[:, i])[0, 1] for col in global_table.columns] for i in range(2)},
        index=global_table.columns,
    )
    eig = s ** 2
    return corr, eig / eig.sum()


def assessor_clustering(means: pd.DataFrame, standardize: bool = False):
    """Ward AHC of assessors on their product x descriptor mean profile."""
    data = means.unstack(level=1).reindex(_assessor_order(means))
    if standardize:
        data = (data - data.mean()) / data.std(ddof=0).replace(0, 1)
    return linkage(data.to_numpy(), method="ward"), list(data.index)
