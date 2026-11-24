"""Discrimination test numbers checked against XLSTAT's output in Demo_TRI_EN.xlsm.

The reference values come from the "Sensory discrimination tests" sheet that
XLSTAT 2021.4 produced from the triangle-test design on the first sheet.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.sensory_analysis import discrimination_stats as ds

WORKBOOK = Path(__file__).resolve().parent.parent / "Demo_TRI_EN.xlsm"
TRIANGLE = ds.METHODS["triangle"]


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Demo_TRI_EN.xlsm not present")
def test_scores_demo_sheet_row_by_row():
    data = pd.read_excel(WORKBOOK, sheet_name=0, engine="calamine")
    scored = ds.score_responses(data["Assessor's answer"], data["Correct answer"])
    assert int(scored.sum()) == 15
    assert len(scored) == 20
    # Same +/- column XLSTAT writes on the "Sensory design" sheet.
    wrong = data.loc[~scored.astype(bool), "Assessor"].tolist()
    assert wrong == ["A4", "A5", "A6", "A13", "A17"]
    samples = ["Sample 1", "Sample 2", "Sample 3"]
    assert not ds.answers_outside_samples(data, samples, "Assessor's answer").any()
    assert not ds.answers_outside_samples(data, samples, "Correct answer").any()


def test_triangle_summary_matches_xlstat():
    # 15 correct out of 20, as scored from the first sheet.
    test, params = ds.discrimination_summary(15, 20, TRIANGLE, alpha=0.05)
    assert test["p-value"] == pytest.approx(0.00016736595467080134, rel=1e-9)
    assert test["Power"] == pytest.approx(0.9861355830574515, rel=1e-9)
    assert test["Critical correct"] == 11
    expected = {
        "Prop. of correct answers": (0.75, 0.09682458365518543, 0.5089541282920425, 0.913428530898585),
        "Prop. of discriminators": (0.625, 0.1452368754827781, 0.26343119243806373, 0.8701427963478775),
        "d-prime": (2.797449452820817, 0.5993343841262891, 1.5129908443528073, 4.196913940733379),
    }
    for parameter, values in expected.items():
        np.testing.assert_allclose(params.loc[parameter].to_numpy(dtype=float), values, rtol=1e-5)


@pytest.mark.parametrize("method", list(ds.METHODS))
def test_psychometric_functions_start_at_chance_and_invert(method):
    m = ds.METHODS[method]
    assert ds.proportion_correct(m, 0.0) == pytest.approx(m.guess)
    for d in (0.5, 1.0, 2.0):
        assert ds.d_prime(m, ds.proportion_correct(m, d)) == pytest.approx(d, abs=1e-6)
    assert ds.d_prime(m, m.guess - 0.05) == 0.0


def test_score_responses_normalizes_codes_and_keeps_missing():
    answers = pd.Series([123, " abc", "X", None])
    correct = pd.Series(["123", "ABC", "Y", "Z"])
    scored = ds.score_responses(answers, correct)
    assert scored.tolist()[:3] == [True, True, False]
    assert pd.isna(scored.iloc[3])
