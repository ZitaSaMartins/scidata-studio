"""Tests for the pure training helpers in modules.ml_models."""

import numpy as np
import pandas as pd
import pytest

from modules.ml_models import (
    CLASSIFICATION,
    REGRESSION,
    allowed_task_types,
    build_pipeline,
    can_stratify,
    detect_task_type,
    split_feature_types,
    train_and_evaluate,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["A", "B", "A", "C"], CLASSIFICATION),
        ([True, False, True, False], CLASSIFICATION),
        ([0, 1, 1, 0, 1, 0], CLASSIFICATION),
        ([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], CLASSIFICATION),
        ([6.4, 7.25, 5.0, 8.1], REGRESSION),
        (list(range(50)), REGRESSION),
        ([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], CLASSIFICATION),
    ],
)
def test_detect_task_type(values, expected):
    assert detect_task_type(pd.Series(values)) == expected


def test_detect_task_type_few_rows_many_distinct_integers_is_regression():
    # 5 distinct integer values in 6 rows: barely any repetition, so treat as a quantity
    assert detect_task_type(pd.Series([3, 5, 7, 9, 11, 3])) == REGRESSION


def test_allowed_task_types():
    assert allowed_task_types(pd.Series(["a", "b"])) == [CLASSIFICATION]
    assert allowed_task_types(pd.Series([True, False])) == [CLASSIFICATION]
    assert allowed_task_types(pd.Series([1.5, 2.5])) == [CLASSIFICATION, REGRESSION]


def test_split_feature_types():
    X = pd.DataFrame({
        "num": [1.0, 2.0],
        "cat": ["a", "b"],
        "flag": [True, False],
        "when": pd.to_datetime(["2024-01-01", "2024-01-02"]),
    })
    assert split_feature_types(X) == (["num"], ["cat", "flag"], ["when"])


def test_pipeline_one_hot_encodes_and_scales():
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "c", "a"]})
    y = pd.Series([0, 1, 0, 1])
    pipe = build_pipeline("Logistic Regression", CLASSIFICATION, ["num"], ["cat"]).fit(X, y)
    transformed = pipe.named_steps["prep"].transform(X)
    # 1 scaled numeric column + 3 one-hot columns (not a single factorized code)
    assert transformed.shape == (4, 4)
    assert np.isclose(transformed[:, 0].mean(), 0)
    assert set(np.unique(transformed[:, 1:])) == {0.0, 1.0}


def test_random_forest_skips_scaling():
    X = pd.DataFrame({"num": [10.0, 20.0, 30.0, 40.0]})
    pipe = build_pipeline("Random Forest", REGRESSION, ["num"], []).fit(X, [1, 2, 3, 4])
    assert pipe.named_steps["prep"].transform(X)[:, 0].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_can_stratify():
    assert can_stratify(pd.Series(["a"] * 10 + ["b"] * 10), 0.2)
    assert not can_stratify(pd.Series(["a"] * 10 + ["b"]), 0.2)  # singleton class
    assert not can_stratify(pd.Series(list("abcde") * 2), 0.2)  # test set smaller than #classes


def _classification_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    product = rng.choice(["A", "B", "C"], size=n)
    sweetness = rng.normal(5, 1, size=n)
    liked = np.where((product == "A") | (sweetness > 5.5), "yes", "no")
    return pd.DataFrame({"product": product, "sweetness": sweetness}), pd.Series(liked, name="liked")


def test_train_classification_is_stratified_and_cross_validated():
    X, y = _classification_data()
    result = train_and_evaluate(X, y, "Logistic Regression", CLASSIFICATION, test_size=0.25, random_state=1, cv_folds=5)
    assert result["stratified"]
    assert len(result["cv_scores"]) == 5
    assert result["test_accuracy"] > 0.7
    assert sorted(result["labels"]) == ["no", "yes"]
    assert result["confusion_matrix"].shape == (2, 2)
    # stratification keeps class proportions close in the test set
    test_share = (pd.Series(result["y_test"]) == "yes").mean()
    assert abs(test_share - (y == "yes").mean()) < 0.05


def test_train_random_forest_importance_grouped_by_original_column():
    X, y = _classification_data()
    result = train_and_evaluate(X, y, "Random Forest", CLASSIFICATION)
    importance = result["importance"]
    assert set(importance["Feature"]) == {"product", "sweetness"}
    assert np.isclose(importance["Importance"].sum(), 1.0)


def test_train_regression():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"x": rng.normal(size=80), "group": rng.choice(["g1", "g2"], size=80)})
    y = 3 * X["x"] + (X["group"] == "g2") * 2 + rng.normal(scale=0.1, size=80)
    result = train_and_evaluate(X, y, "Linear Regression", REGRESSION, cv_folds=4)
    assert result["test_r2"] > 0.95
    assert len(result["cv_scores"]) == 4
    assert not result["stratified"]


def test_unseen_category_in_test_set_does_not_crash():
    X = pd.DataFrame({"cat": ["a"] * 10 + ["b"] * 9 + ["rare"], "num": range(20)})
    y = pd.Series([0, 1] * 10)
    result = train_and_evaluate(X, y, "Logistic Regression", CLASSIFICATION, random_state=0)
    assert 0 <= result["test_accuracy"] <= 1


def test_tiny_classes_fall_back_to_random_split_with_note():
    X = pd.DataFrame({"num": range(12)})
    y = pd.Series(["a"] * 6 + ["b"] * 5 + ["c"])
    result = train_and_evaluate(X, y, "Random Forest", CLASSIFICATION)
    assert not result["stratified"]
    assert any("stratified" in note for note in result["notes"])


def test_datetime_features_are_dropped_with_note():
    X = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=30), "num": range(30)})
    y = pd.Series(np.arange(30) * 1.5)
    result = train_and_evaluate(X, y, "Linear Regression", REGRESSION)
    assert result["numeric_features"] == ["num"]
    assert any("Date/time" in note for note in result["notes"])


def test_single_class_target_raises():
    with pytest.raises(ValueError, match="only one class"):
        train_and_evaluate(pd.DataFrame({"x": range(10)}), pd.Series(["a"] * 10), "Logistic Regression", CLASSIFICATION)
