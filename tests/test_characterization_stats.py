"""Product characterization numbers checked against XLSTAT's output.

The reference values are read from the "Product characterization" sheet that
XLSTAT 2025.1 produced for the "Data" sheet of demo-Product_characterization.xlsm
(model Y = P + A, 95% confidence intervals).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.sensory_analysis import characterization_stats as cs

WORKBOOK = Path(__file__).resolve().parent.parent / "demo-Product_characterization.xlsm"
SHEET = "Product characterization"
TOL = 1e-6

pytestmark = pytest.mark.skipif(not WORKBOOK.exists(), reason="demo-Product_characterization.xlsm not present")


def _block(sheet: pd.DataFrame, first_row: int, last_row: int, first_col: int = 1, last_col: int | None = None) -> pd.DataFrame:
    """Excel rows first_row..last_row (1-based), first cell = row labels; the row above holds the headers."""
    header = sheet.iloc[first_row - 2, first_col:last_col].tolist()
    block = sheet.iloc[first_row - 1:last_row, first_col:last_col]
    block.columns = header
    return block.set_index(header[0]).astype(float)


@pytest.fixture(scope="module")
def sheet():
    return pd.read_excel(WORKBOOK, sheet_name=SHEET, header=None, engine="calamine")


@pytest.fixture(scope="module")
def data():
    return pd.read_excel(WORKBOOK, sheet_name="Data", engine="calamine")


@pytest.fixture(scope="module")
def design(data):
    return cs.CharacterizationDesign("Product", tuple(data.columns[5:]), assessor="Panelist")


@pytest.fixture(scope="module")
def result(data, design):
    return cs.analyse(data, design)


@pytest.fixture(scope="module")
def pca(result):
    return cs.profile_pca(result["means"])


def test_discriminating_power(sheet, result):
    expected = _block(sheet, 73, 86, last_col=4)
    power = result["power"].set_index("Descriptors")
    assert power.index.tolist() == expected.index.tolist()  # same ranking
    np.testing.assert_allclose(power["Test values"], expected["Test values"], atol=TOL)
    np.testing.assert_allclose(power["p-values"], expected["p-values"], atol=TOL)


def test_model_coefficients(sheet, result):
    products = [p for p in sheet.iloc[112, 2:].tolist() if isinstance(p, str)]
    stats = ["Coefficient", "Estimated mean", "p-value", "Test value", "Lower bound", "Upper bound"]
    coefficients = result["coefficients"].set_index(["Descriptor", "Product"])
    for row in range(114, 128):
        descriptor = sheet.iloc[row, 1]
        for i, product in enumerate(products):
            expected = sheet.iloc[row, 2 + 6 * i: 8 + 6 * i].astype(float).to_numpy()
            actual = coefficients.loc[(descriptor, product), stats].astype(float).to_numpy()
            np.testing.assert_allclose(actual, expected, atol=TOL, err_msg=f"{descriptor} / {product}")


def test_adjusted_means_sorted_like_xlstat(sheet, result, pca):
    expected = _block(sheet, 156, 161, last_col=16)
    means = cs.sorted_means(result["means"], pca)
    assert means.index.tolist() == expected.index.tolist()
    assert means.columns.tolist() == expected.columns.tolist()
    np.testing.assert_allclose(means.to_numpy(), expected.to_numpy(), atol=TOL)


def test_pca_eigenvalues(sheet, pca):
    expected = _block(sheet, 169, 171, last_col=7)
    np.testing.assert_allclose(pca.eigen_table.to_numpy(), expected.to_numpy(), atol=TOL)


def test_pca_scores_and_signs(sheet, pca):
    np.testing.assert_allclose(pca.scores.to_numpy(), _block(sheet, 196, 201, last_col=7).to_numpy(), atol=TOL)
    np.testing.assert_allclose(pca.squared_cosines.to_numpy(), _block(sheet, 207, 212, last_col=7).to_numpy(), atol=TOL)


def test_bootstrap_ellipses_surround_products(result, design, pca):
    boot = cs.bootstrap_scores(result["data"], design, pca, n_boot=300)
    assert len(boot) == 300 * len(pca.scores)
    centres = boot.groupby("Product")[["F1", "F2"]].mean().loc[pca.scores.index]
    # resampled panels average back to the observed product positions
    np.testing.assert_allclose(centres.to_numpy(), pca.scores[["F1", "F2"]].to_numpy(), atol=0.3)
    outline = cs.ellipse_points(boot.loc[boot["Product"] == "choc3", ["F1", "F2"]].to_numpy())
    assert outline[:, 0].min() < pca.scores.loc["choc3", "F1"] < outline[:, 0].max()


def test_test_value_is_stable_for_tiny_p_values():
    assert cs.test_value_from_log_p(np.log(0.05)) == pytest.approx(1.6448536269514722)
    assert np.isfinite(cs.test_value_from_log_p(-2000.0))  # p = e^-2000 underflows to 0 as a float


def test_one_way_model_matches_tukey_hsd(data):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    res = cs.analyse(data, cs.CharacterizationDesign("Product", ("Sticky",)))
    tukey = pairwise_tukeyhsd(data["Sticky"], data["Product"])
    np.testing.assert_allclose(res["pairs"]["p-value (Tukey)"], tukey.pvalues, atol=1e-6)


def test_unbalanced_data_uses_least_squares_means(data, design):
    dropped = data.drop(index=data.index[(data["Product"] == "choc1")][:5])
    res = cs.analyse(dropped, design)
    coefficients = res["coefficients"]
    # sum-to-zero: product effects still add to zero, and adjusted means differ from raw means
    sums = coefficients.groupby("Descriptor")["Coefficient"].sum()
    np.testing.assert_allclose(sums, 0, atol=1e-9)
    raw = dropped.groupby("Product")["Cocoaa"].mean()
    assert not np.isclose(res["means"].loc["choc1", "Cocoaa"], raw["choc1"])


def test_significance_colours_match_xlstat_table(result):
    """XLSTAT fills a cell of the adjusted-means table when the product's coefficient is significant (5%)."""
    import openpyxl

    sheet = openpyxl.load_workbook(WORKBOOK)[SHEET]
    descriptors = [sheet.cell(155, c).value for c in range(3, 17)]
    coefficients = result["coefficients"].set_index(["Product", "Descriptor"])
    fills = {"FF00C7D1": 1, "FFFF0000": -1}
    for row in range(156, 162):
        product = sheet.cell(row, 2).value
        for col, descriptor in zip(range(3, 17), descriptors, strict=True):
            fill = sheet.cell(row, col).fill
            expected = fills.get(fill.fgColor.rgb, 0) if fill.fill_type == "solid" else 0
            coef, p_value = coefficients.loc[(product, descriptor), ["Coefficient", "p-value"]]
            actual = int(np.sign(coef)) if p_value < 0.05 else 0
            assert actual == expected, f"{product} / {descriptor}"
