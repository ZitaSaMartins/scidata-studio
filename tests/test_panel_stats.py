"""Panel analysis numbers checked against XLSTAT's output in tuto_PAN_EN.xls.

The reference values below were copied from the "Panel analysis" sheet that
XLSTAT 2020.4 produced for the "Data" sheet of the same workbook.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.sensory_analysis import panel_stats as ps

WORKBOOK = Path(__file__).resolve().parent.parent / "tuto_PAN_EN.xls"
DESCRIPTORS = ("Smoothness", "Touch", "Feet feeling", "Elasticity", "Hot feeling", "Closing")

pytestmark = pytest.mark.skipif(not WORKBOOK.exists(), reason="tuto_PAN_EN.xls not present")


@pytest.fixture(scope="module")
def data():
    return pd.read_excel(WORKBOOK, sheet_name="Data", engine="calamine")


@pytest.fixture(scope="module")
def design():
    return ps.PanelDesign("Assessor", "Product", DESCRIPTORS, "Session")


@pytest.fixture(scope="module")
def anovas(data, design):
    return ps.anova_all(data, design)


@pytest.fixture(scope="module")
def tests_table(data, design, anovas):
    return ps.assessor_tests(data, design, anovas)


def test_summary_statistics(data):
    quant = ps.summary_quantitative(data, list(DESCRIPTORS)).set_index("Variable")
    assert quant.loc["Smoothness", "Mean"] == pytest.approx(9.667380952380944)
    assert quant.loc["Touch", "Std. deviation"] == pytest.approx(5.343288113774425)
    assert quant.loc["Hot feeling", "Minimum"] == pytest.approx(-0.8)
    qual = ps.summary_qualitative(data, ["Product", "Assessor", "Session"])
    assert qual.loc[qual["Variable"] == "Session", "Counts"].tolist() == [210, 210]
    assert qual.loc[qual["Variable"] == "Assessor", "Categories"].tolist()[:3] == ["1", "2", "3"]


def test_mixed_anova_matches_xlstat(anovas):
    table = anovas["Smoothness"].set_index("Source")
    expected = {
        "Product": (13, 1974.2516190476194, 7.897570615239763, 2.1488380741429937e-12),
        "Assessor": (14, 913.7990119047631, 3.3943530809831794, 6.924253179366004e-05),
        "Session": (1, 0.5215238095242967, 0.03677019490029375, 0.8481198533914508),
        "Product*Assessor": (182, 3499.749988095239, 1.3557743843531473, 0.01672069146096083),
    }
    for source, (df, ss, f, p) in expected.items():
        assert table.loc[source, "DF"] == df
        assert table.loc[source, "Sum of squares"] == pytest.approx(ss, rel=1e-9)
        assert table.loc[source, "F"] == pytest.approx(f, rel=1e-9)
        assert table.loc[source, "Pr > F"] == pytest.approx(p, rel=1e-6)
    assert table.loc["Error", "Mean squares"] == pytest.approx(14.18333002961946)


def test_product_effect_matches_xlstat(anovas):
    effect = ps.product_effect(anovas).set_index("Descriptors")["F"]
    expected = [7.897570615239763, 31.346881812009865, 62.698957485143076, 47.9757721908, 20.3559838316003, 29.045843485829103]
    assert effect.tolist() == pytest.approx(expected, rel=1e-9)
    assert ps.discriminating_descriptors(anovas) == list(DESCRIPTORS)


def test_cap_summary_and_rank_f(data, design, anovas, tests_table):
    cap, rank_f = ps.cap_table(data, design, anovas, tests_table)
    assert cap.index.tolist() == ["Feet feeling", "Elasticity", "Touch", "Closing", "Hot feeling", "Smoothness"]
    assert cap.loc["Feet feeling", "FInter"] == pytest.approx(1.0853129993863386)
    assert cap.loc["Feet feeling", "RMSE"] == pytest.approx(2.1543832784166392)
    assert rank_f.index[:4].tolist() == ["11", "12", "10", "2"]
    assert rank_f["11"] == pytest.approx(2.3333333333333335)
    assert rank_f["4"] == pytest.approx(11.666666666666666)


def _xlstat_cap_codes() -> dict[tuple[str, str], str]:
    sheet = pd.read_excel(WORKBOOK, sheet_name="Panel analysis", header=None, engine="calamine")
    header = [v for v in sheet.iloc[166].tolist() if pd.notna(v)]
    assessors = [str(int(v)) for v in header[5:]]
    codes = {}
    for i in range(167, 173):
        row = [v for v in sheet.iloc[i].tolist() if pd.notna(v)]
        for assessor, code in zip(assessors, row[5:], strict=True):
            codes[(row[0], assessor)] = str(code).strip()
    return codes


def test_cap_codes_match_xlstat(data, design, anovas, tests_table):
    cap, rank_f = ps.cap_table(data, design, anovas, tests_table)
    reference = _xlstat_cap_codes()
    ours = {(d, a): cap.loc[d, a] for d in cap.index for a in rank_f.index}
    # Discrimination / agreement symbol (+ - =): identical for all 90 cells
    assert all(ours[k][0] == v[0] for k, v in reference.items())
    # Repeatability flag (!): XLSTAT's exact rule isn't documented; we match 87/90
    same_flag = sum(("!" in ours[k]) == ("!" in v) for k, v in reference.items())
    assert same_flag >= 87


def test_ability_to_discriminate_assessor_1(tests_table):
    table = ps.ability_to_discriminate(tests_table, "1")
    assert table.loc["MSE"].tolist() == pytest.approx([8.645082417582422, 13.06901098901101, 2.697499999999993, 5.257829670329661, 5.112664835164831, 3.5759340659340584])
    assert table.loc["F"].tolist() == pytest.approx([1.9152379711517373, 1.9054385846898931, 19.489728992046004, 10.205360921702352, 2.402206328821449, 9.659598660151834])


def test_significant_counts_and_discrimination_score(tests_table):
    counts = ps.significant_counts(tests_table)
    assert counts.loc["Sum", list(DESCRIPTORS)].tolist() == [7, 11, 15, 14, 11, 13]
    assert counts.loc["Sum", "Sum"] == 71
    score = ps.performance_score(tests_table)
    assert score.loc["Discrimination"].tolist() == [3, 6, 4, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 5, 5]
    # XLSTAT: [5, 5, 3, 5, 4, 6, 4, 3, 5, 5, 6, 3, 6, 5, 4]; assessor 4 is borderline (p = 0.098)
    no_interaction = score.loc["No interaction"].tolist()
    assert sum(a == b for a, b in zip(no_interaction, [5, 5, 3, 5, 4, 6, 4, 3, 5, 5, 6, 3, 6, 5, 4], strict=True)) >= 14


def test_session_repeatability_and_range(data, design):
    p = ps.session_repeatability(data, design)
    assert p.loc["1", "Closing"] == pytest.approx(0.0001220703125)  # exact Wilcoxon, no ties
    assert p.loc["3", "Smoothness"] == pytest.approx(0.42626953125)
    ranges = ps.max_session_range(data, design)
    assert ranges.loc["1"].tolist() == pytest.approx([10.1, 15.2, 7.5, 4.8, 8.1, 10.0])
    assert ranges.loc["15", "Smoothness"] == pytest.approx(16.6)


def test_means_and_distance_to_consensus(data, design):
    means = ps.assessor_product_means(data, design)
    wide = ps.means_wide(means)
    assert wide.columns[:2].tolist() == ["1|Smoothness", "2|Smoothness"]
    assert wide.loc["1", "14|Smoothness"] == pytest.approx(14.75)
    comparison = ps.product_comparison(means, "1")
    assert comparison.loc["Average"].tolist() == pytest.approx([6.888333333333333, 3.2466666666666675, 3.995, 16.450000000000003, 5.725, 14.776666666666667])
    dist = ps.distance_to_consensus(means)
    assert dist.index[:3].tolist() == ["1", "2", "3"]
    assert dist.loc["1", "1"] == pytest.approx(1.159214892358)
    assert dist.min().tolist()[:3] == pytest.approx([1.092338215836, 1.474332083552, 0.893263627048])


def test_multivariate_outputs_have_expected_shape(data, design):
    means = ps.assessor_product_means(data, design)
    corr, ratios = ps.pca_correlations(ps.product_by_assessor_descriptor(means))
    assert corr.shape == (90, 2)
    assert np.all(np.abs(corr.to_numpy()) <= 1 + 1e-9)
    assert ratios[0] > ratios[1]
    mfa, _ = ps.mfa_correlations(means)
    assert mfa.shape == (90, 2)
    link, labels = ps.assessor_clustering(means)
    assert link.shape == (14, 4)
    assert labels[:3] == ["1", "2", "3"]


def test_without_session_column(data):
    design = ps.PanelDesign("Assessor", "Product", DESCRIPTORS[:2])
    anovas = ps.anova_all(data, design)
    assert "Session" not in anovas["Smoothness"]["Source"].tolist()
    tests = ps.assessor_tests(data, design, anovas)
    assert tests["p_session"].isna().all()
    assert ps.session_repeatability(data, design) is None


def test_incomplete_design_drops_interaction():
    # 3 assessors, 3 products, assessor C never rated product z: interaction not estimable
    rows = [(a, p, r, v) for a in "AB" for p in "xyz" for r in (1, 2) for v in [hash((a, p, r)) % 7 + 1.0]]
    rows += [("C", p, r, float(r + i)) for i, p in enumerate("xy") for r in (1, 2)]
    df = pd.DataFrame(rows, columns=["Assessor", "Product", "Session", "Score"])
    design = ps.PanelDesign("Assessor", "Product", ("Score",), "Session")
    assert not ps.interaction_estimable(df, design)
    anovas = ps.anova_all(df, design)
    assert "Product*Assessor" not in anovas["Score"]["Source"].tolist()
    score = ps.performance_score(ps.assessor_tests(df, design, anovas))
    assert "No interaction" not in score.index


def test_kendall_w_per_descriptor(data, design):
    means = ps.assessor_product_means(data, design)
    kendall = ps.kendall_w_all(means).set_index("Descriptor")
    # Same value the former Agreement menu gave for Smoothness
    assert kendall.loc["Smoothness", "W"] == pytest.approx(0.3154, abs=1e-4)
    assert kendall.loc["Smoothness", "DF"] == 13
    assert (kendall["W"].between(0, 1)).all()
    assert kendall.loc["Feet feeling", "W"] > kendall.loc["Smoothness", "W"]


def test_kendall_w_is_nan_without_variation():
    df = pd.DataFrame({"A": list("aabb"), "P": list("xyxy"), "S": [1.0, 1.0, 2.0, 2.0]})
    design = ps.PanelDesign("A", "P", ("S",))
    result = ps.kendall_w(ps.assessor_product_means(df, design), "S")
    assert np.isnan(result["W"])


def test_assessor_correlations_are_symmetric(data, design):
    corr = ps.assessor_correlations(ps.assessor_product_means(data, design), "Touch")
    assert corr.shape == (15, 15)
    assert np.allclose(corr.to_numpy(), corr.to_numpy().T)
    assert np.allclose(np.diag(corr), 1)


def test_scale_usage_levels_sum_to_zero_on_balanced_data(data, design):
    usage = ps.scale_usage(data, design)
    level = ps.scale_usage_pivot(usage, "Level vs panel", [str(i) for i in range(1, 16)], list(DESCRIPTORS))
    assert np.allclose(level.sum(), 0)
    row = usage[(usage["Assessor"] == "1") & (usage["Descriptor"] == "Smoothness")].iloc[0]
    assert row["Range"] == pytest.approx(row["Max"] - row["Min"])


def test_within_session_sd(data, design):
    sd = ps.within_session_sd(data, design)
    assert sd.shape == (15, 6)
    assert sd.index[:2].tolist() == ["1", "2"]
    # two sessions: SD of a pair = |a - b| / sqrt(2)
    sub = data[(data["Assessor"] == 1) & (data["Product"] == 1)]["Smoothness"]
    assert abs(sub.iloc[0] - sub.iloc[1]) / np.sqrt(2) == pytest.approx(sub.std())
