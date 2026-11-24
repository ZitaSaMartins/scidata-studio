"""Product characterization computations, laid out like XLSTAT's "Product characterization".

Pure pandas / statsmodels code (no Streamlit) so every number can be tested
against the reference output in ``demo-Product_characterization.xlsm``.

Per descriptor, an ANOVA with sum-to-zero contrasts::

    Y = P [+ A] [+ S] [+ P*A]

* **Discriminating power**: the product effect's F test (type III). With the
  P*A interaction in the model the product is tested against MS(P*A) (mixed
  model, assessor random), otherwise against the error. The *test value* is
  the standard normal quantile with the same upper-tail p-value
  (``Phi^-1(1 - p)``), so descriptors can be ranked even when every p-value
  prints as 0.
* **Model coefficients**: the product effect for each product (adjusted mean
  minus the mean of the adjusted means), with a two-sided t test. Test value
  = ``sign(coef) * Phi^-1(1 - p/2)``.
* **Adjusted means**: least-squares means (intercept + product coefficient).
* **Pairwise comparisons** (not part of XLSTAT's report): Tukey-Kramer tests
  of the adjusted means, using the same error term as the coefficients.
* **PCA**: normalized PCA (correlation matrix) of the product x descriptor
  table of adjusted means. Axis signs follow XLSTAT: each eigenvector's
  largest absolute loading is positive.
* **Confidence ellipses**: assessors are resampled with replacement
  (Husson, Le & Pages, 2005), each virtual panel's product means are projected
  as supplementary rows, and a 95% ellipse is drawn from their covariance.
  XLSTAT uses its own random draws, so ellipses match in size and shape but
  not to the decimal.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import special, stats
from statsmodels.stats.anova import anova_lm

from .panel_stats import _ordered_levels, _q

DEFAULT_SEED = 123456789


@dataclass(frozen=True)
class CharacterizationDesign:
    product: str
    descriptors: tuple[str, ...]
    assessor: str | None = None
    session: str | None = None
    interaction: bool = False

    @property
    def factors(self) -> list[str]:
        return [c for c in (self.product, self.assessor, self.session) if c]

    @property
    def model_label(self) -> str:
        label = "Y = P"
        if self.assessor:
            label += " + A"
        if self.session:
            label += " + S"
        if self.interaction and self.assessor:
            label += " + P*A"
        return label


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _prepare(df: pd.DataFrame, design: CharacterizationDesign) -> pd.DataFrame:
    data = df[design.factors + list(design.descriptors)].copy()
    for factor in design.factors:
        data[factor] = data[factor].astype(str)
    for descriptor in design.descriptors:
        data[descriptor] = pd.to_numeric(data[descriptor], errors="coerce")
    return data.dropna(subset=design.factors)


def test_value_from_log_p(log_p: float) -> float:
    """Normal quantile with upper-tail probability exp(log_p); stable for tiny p."""
    return float(-special.ndtri_exp(log_p))


def interaction_estimable(df: pd.DataFrame, product: str, assessor: str) -> bool:
    """P*A needs every assessor to rate every product, some of them more than once."""
    sizes = df.groupby([assessor, product]).size()
    return len(sizes) == df[assessor].nunique() * df[product].nunique() and bool(sizes.max() > 1)


def _fit(data: pd.DataFrame, design: CharacterizationDesign, descriptor: str):
    terms = [f"C({_q(design.product)}, Sum)"]
    if design.assessor:
        terms.append(f"C({_q(design.assessor)}, Sum)")
    if design.session:
        terms.append(f"C({_q(design.session)}, Sum)")
    if design.interaction and design.assessor:
        terms.append(f"C({_q(design.product)}, Sum):C({_q(design.assessor)}, Sum)")
    frame = data.dropna(subset=[descriptor])
    model = smf.ols(f"{_q(descriptor)} ~ " + " + ".join(terms), data=frame).fit()
    if model.df_resid <= 0:
        raise ValueError(f"Not enough ratings to fit the model for '{descriptor}'.")
    return model, frame


def _product_test(model, design: CharacterizationDesign) -> dict:
    """F test of the product effect, and the error term the coefficients are tested against."""
    table = anova_lm(model, typ=3)
    p_key = f"C({_q(design.product)}, Sum)"
    prod = table.loc[p_key]
    ms_err, df_err = model.mse_resid, model.df_resid
    if design.interaction and design.assessor:
        inter = next(table.loc[idx] for idx in table.index if ":" in idx)
        denom_ms, denom_df = inter["sum_sq"] / inter["df"], inter["df"]
    else:
        denom_ms, denom_df = ms_err, df_err
    ms_prod = prod["sum_sq"] / prod["df"]
    f_value = ms_prod / denom_ms if denom_ms > 0 else np.nan
    log_p = stats.f.logsf(f_value, prod["df"], denom_df)
    return {
        "F": f_value,
        "DF": prod["df"],
        "log_p": log_p,
        "denom_ms": denom_ms,
        "denom_df": denom_df,
        "scale": np.sqrt(denom_ms / ms_err) if ms_err > 0 else np.nan,
    }


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #

def analyse(df: pd.DataFrame, design: CharacterizationDesign, confidence: float = 0.95) -> dict:
    """Run every per-descriptor model. Returns the XLSTAT tables as DataFrames."""
    data = _prepare(df, design)
    products = _ordered_levels(data[design.product])
    power_rows, coef_rows, pair_rows = [], [], []
    for descriptor in design.descriptors:
        model, frame = _fit(data, design, descriptor)
        test = _product_test(model, design)
        p_value = float(np.exp(test["log_p"]))
        power_rows.append({
            "Descriptors": descriptor,
            "Test values": test_value_from_log_p(test["log_p"]),
            "p-values": p_value,
            "F": test["F"],
        })
        names = list(model.params.index)
        levels = [lvl for lvl in products if lvl in set(frame[design.product])]
        p_key = f"C({_q(design.product)}, Sum)"
        intercept = model.params["Intercept"]
        cov = model.cov_params().to_numpy()
        t_crit = stats.t.ppf(0.5 + confidence / 2, test["denom_df"])
        columns = {level: f"{p_key}[S.{level}]" for level in levels}
        estimated = [names.index(col) for col in columns.values() if col in names]
        contrasts = {}
        for level in levels:
            contrast = np.zeros(len(names))
            if columns[level] in names:
                contrast[names.index(columns[level])] = 1.0
            else:  # sum-to-zero: the omitted level = -(sum of the others)
                contrast[estimated] = -1.0
            contrasts[level] = contrast
            coef = float(contrast @ model.params.to_numpy())
            se = float(np.sqrt(contrast @ cov @ contrast)) * test["scale"]
            t_value = coef / se if se > 0 else np.nan
            log_p = np.log(2.0) + stats.t.logsf(abs(t_value), test["denom_df"])
            log_p = min(log_p, 0.0)
            coef_rows.append({
                "Descriptor": descriptor,
                "Product": level,
                "Coefficient": coef,
                "Estimated mean": intercept + coef,
                "Std. error": se,
                "p-value": float(np.exp(log_p)),
                "Test value": float(np.sign(coef) * test_value_from_log_p(log_p - np.log(2.0))),
                "Lower bound": coef - t_crit * se,
                "Upper bound": coef + t_crit * se,
            })
        for i, first in enumerate(levels):
            for second in levels[i + 1:]:
                diff_contrast = contrasts[first] - contrasts[second]
                diff = float(diff_contrast @ model.params.to_numpy())
                se = float(np.sqrt(diff_contrast @ cov @ diff_contrast)) * test["scale"]
                q = abs(diff) / se * np.sqrt(2) if se > 0 else np.nan
                pair_rows.append({
                    "Descriptor": descriptor,
                    "Product 1": first,
                    "Product 2": second,
                    "Difference": diff,
                    "p-value (Tukey)": float(stats.studentized_range.sf(q, len(levels), test["denom_df"])),
                })
    power = (pd.DataFrame(power_rows)
             .sort_values("Test values", ascending=False, kind="stable")
             .reset_index(drop=True))
    coefficients = pd.DataFrame(coef_rows)
    means = (coefficients.pivot(index="Product", columns="Descriptor", values="Estimated mean")
             .reindex(index=products, columns=list(design.descriptors)))
    return {"data": data, "products": products, "power": power, "coefficients": coefficients,
            "means": means, "pairs": pd.DataFrame(pair_rows)}


def coefficients_wide(coefficients: pd.DataFrame, value: str) -> pd.DataFrame:
    """Descriptor x product table of one coefficient statistic, in input order."""
    descriptors = list(dict.fromkeys(coefficients["Descriptor"]))
    products = list(dict.fromkeys(coefficients["Product"]))
    return coefficients.pivot(index="Descriptor", columns="Product", values=value).reindex(index=descriptors, columns=products)


# --------------------------------------------------------------------------- #
# PCA on adjusted means
# --------------------------------------------------------------------------- #

@dataclass
class ProfilePCA:
    eigenvalues: np.ndarray
    vectors: np.ndarray       # descriptors x axes (unit-norm eigenvectors)
    scores: pd.DataFrame      # products x axes
    loadings: pd.DataFrame    # descriptor-axis correlations
    center: pd.Series
    scale: pd.Series

    @property
    def eigen_table(self) -> pd.DataFrame:
        pct = 100 * self.eigenvalues / self.eigenvalues.sum()
        axes = [f"F{i + 1}" for i in range(len(self.eigenvalues))]
        return pd.DataFrame([self.eigenvalues, pct, np.cumsum(pct)],
                            index=["Eigenvalue", "Variability (%)", "Cumulative %"], columns=axes)

    @property
    def squared_cosines(self) -> pd.DataFrame:
        sq = self.scores ** 2
        return sq.div(sq.sum(axis=1), axis=0)

    def project(self, table: pd.DataFrame) -> np.ndarray:
        z = (table[self.center.index] - self.center) / self.scale
        return z.to_numpy() @ self.vectors


def profile_pca(means: pd.DataFrame) -> ProfilePCA:
    """Normalized PCA (correlation matrix) of a product x descriptor table."""
    table = means.dropna(axis=1)
    table = table.loc[:, table.std(ddof=0) > 0]
    center, scale = table.mean(), table.std(ddof=0)
    z = ((table - center) / scale).to_numpy()
    n_axes = min(z.shape[0] - 1, z.shape[1])
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    vectors = vt.T[:, :n_axes]
    signs = np.sign(vectors[np.abs(vectors).argmax(axis=0), np.arange(n_axes)])
    signs[signs == 0] = 1.0
    vectors = vectors * signs
    eigenvalues = s[:n_axes] ** 2 / z.shape[0]
    axes = [f"F{i + 1}" for i in range(n_axes)]
    scores = pd.DataFrame(z @ vectors, index=table.index, columns=axes)
    loadings = pd.DataFrame(vectors * np.sqrt(eigenvalues), index=table.columns, columns=axes)
    return ProfilePCA(eigenvalues, vectors, scores, loadings, center, scale)


def sorted_means(means: pd.DataFrame, pca: ProfilePCA) -> pd.DataFrame:
    """Adjusted means with products sorted by F1 score and descriptors by F1 correlation (both descending)."""
    rows = pca.scores["F1"].sort_values(ascending=False, kind="stable").index
    cols = pca.loadings["F1"].sort_values(ascending=False, kind="stable").index
    rest = [c for c in means.columns if c not in cols]
    return means.loc[rows, [*cols, *rest]]


# --------------------------------------------------------------------------- #
# Bootstrap confidence ellipses
# --------------------------------------------------------------------------- #

def bootstrap_scores(data: pd.DataFrame, design: CharacterizationDesign, pca: ProfilePCA,
                     n_boot: int = 500, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Product scores (F1..) for panels of assessors resampled with replacement."""
    descriptors = list(pca.center.index)
    cube = data.groupby([design.assessor, design.product])[descriptors].mean()
    assessors = cube.index.get_level_values(0).unique()
    products = list(pca.scores.index)
    array = np.stack([cube.xs(a, level=0).reindex(products).to_numpy() for a in assessors])  # A x P x D
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(assessors), size=(n_boot, len(assessors)))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a product no drawn assessor rated
        virtual = np.nanmean(array[draws], axis=1)  # B x P x D
    z = (virtual - pca.center.to_numpy()) / pca.scale.to_numpy()
    scores = z.reshape(-1, len(descriptors)) @ pca.vectors
    frame = pd.DataFrame(scores, columns=pca.scores.columns)
    frame.insert(0, "Product", np.tile(products, n_boot))
    frame["Replicate"] = np.repeat(np.arange(n_boot), len(products))
    return frame.dropna().reset_index(drop=True)


def ellipse_points(points: np.ndarray, level: float = 0.95, n: int = 100) -> np.ndarray:
    """Outline of the ellipse holding `level` of a bivariate normal fitted to `points`."""
    center = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    radius = np.sqrt(stats.chi2.ppf(level, 2))
    angle = np.linspace(0, 2 * np.pi, n)
    circle = np.c_[np.cos(angle), np.sin(angle)] * np.sqrt(np.clip(values, 0, None)) * radius
    return center + circle @ vectors.T
