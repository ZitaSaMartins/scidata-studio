"""Machine-learning page: model training (pure functions) and Streamlit UI.

The training core (``detect_task_type``, ``build_pipeline``,
``train_and_evaluate``) has no Streamlit calls so it can be unit tested.
``run_ml_pipeline`` is the UI; it caches training results with
``st.cache_data`` so widget interactions don't retrain the model.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLASSIFICATION = "Classification"
REGRESSION = "Regression"

MODELS = {
    CLASSIFICATION: ["Logistic Regression", "Random Forest"],
    REGRESSION: ["Linear Regression", "Random Forest"],
}

MAX_CLASSES_FOR_NUMERIC_TARGET = 10
HIGH_CARDINALITY = 50
DEFAULT_CV_FOLDS = 5


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def detect_task_type(y: pd.Series) -> str:
    """Guess whether predicting ``y`` is classification or regression.

    * Text, category and boolean targets -> classification.
    * Numbers with decimals (e.g. 6.4, 7.25) -> regression.
    * Whole numbers -> classification only when there are few distinct values
      (≤ 10) and they repeat a lot (fewer distinct values than half the rows),
      e.g. 0/1 flags or class codes. Otherwise regression (counts, scores on a
      wide scale, ...).

    The UI lets the user override this guess.
    """
    y = y.dropna()
    if y.empty:
        return REGRESSION
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return CLASSIFICATION
    values = y.to_numpy(dtype=float)
    if not np.all(np.isclose(values, np.round(values))):
        return REGRESSION
    n_unique = y.nunique()
    if n_unique <= 2:
        return CLASSIFICATION
    if n_unique <= MAX_CLASSES_FOR_NUMERIC_TARGET and n_unique < len(y) / 2:
        return CLASSIFICATION
    return REGRESSION


def allowed_task_types(y: pd.Series) -> list[str]:
    """Regression only makes sense for numeric (non-boolean) targets."""
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        return [CLASSIFICATION, REGRESSION]
    return [CLASSIFICATION]


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return ``(numeric, categorical, unsupported)`` column names.

    Booleans are treated as categorical. Date/time columns are unsupported
    (one-hot encoding every timestamp is meaningless) and get dropped.
    """
    numeric, categorical, unsupported = [], [], []
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
            unsupported.append(col)
        elif pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            categorical.append(col)
        else:
            numeric.append(col)
    return numeric, categorical, unsupported


def make_estimator(model_name: str, task_type: str, random_state: int = 42):
    if task_type == CLASSIFICATION:
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=2000)
        if model_name == "Random Forest":
            return RandomForestClassifier(n_estimators=200, random_state=random_state)
    elif task_type == REGRESSION:
        if model_name == "Linear Regression":
            return LinearRegression()
        if model_name == "Random Forest":
            return RandomForestRegressor(n_estimators=200, random_state=random_state)
    raise ValueError(f"Unknown model '{model_name}' for task '{task_type}'.")


def build_pipeline(model_name: str, task_type: str, numeric: list[str], categorical: list[str], random_state: int = 42) -> Pipeline:
    """Preprocessing + model.

    * Categorical features are one-hot encoded (unseen categories in the test
      set are ignored instead of crashing).
    * Numeric features are standardised for linear models. Tree models don't
      need scaling, so numbers are passed through unchanged for them.
    """
    scale = model_name != "Random Forest"
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler() if scale else "passthrough", numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("prep", preprocessor), ("model", make_estimator(model_name, task_type, random_state))])


def can_stratify(y: pd.Series, test_size: float) -> bool:
    """``train_test_split(stratify=y)`` needs ≥2 rows per class and room for every class in both sets."""
    counts = y.value_counts()
    n = len(y)
    n_test = math.ceil(test_size * n)
    n_train = n - n_test
    n_classes = len(counts)
    return n_classes >= 2 and counts.min() >= 2 and n_test >= n_classes and n_train >= n_classes


def _cv_splitter(task_type: str, y_train: pd.Series, folds: int, random_state: int):
    if task_type == CLASSIFICATION:
        n_splits = min(folds, int(y_train.value_counts().min()))
        if n_splits < 2:
            return None
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_splits = min(folds, len(y_train))
    if n_splits < 2:
        return None
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _grouped_importance(pipeline: Pipeline, numeric: list[str], categorical: list[str]) -> pd.DataFrame | None:
    """Feature importance summed back to the original columns (one-hot columns are added up)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    owners = list(numeric)
    if categorical:
        encoder = pipeline.named_steps["prep"].named_transformers_["cat"]
        for col, cats in zip(categorical, encoder.categories_, strict=True):
            owners.extend([col] * len(cats))
    importance = pd.DataFrame({"Feature": owners, "Importance": model.feature_importances_})
    return importance.groupby("Feature", sort=False)["Importance"].sum().reset_index().sort_values("Importance", ascending=False, ignore_index=True)


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    task_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> dict:
    """Split, fit, cross-validate and score a model. Returns a plain dict of results."""
    X = X.copy()
    X.columns = [str(c) for c in X.columns]
    notes: list[str] = []

    numeric, categorical, unsupported = split_feature_types(X)
    if unsupported:
        notes.append(f"Date/time columns were left out of the model: {', '.join(unsupported)}.")
    for col in categorical:
        n_levels = X[col].nunique()
        if n_levels > HIGH_CARDINALITY:
            notes.append(f"'{col}' has {n_levels} different values; it may be an ID column and can make the model overfit.")
    if not numeric and not categorical:
        raise ValueError("No usable predictor columns (only date/time columns were selected).")
    # Mixed-type text columns (e.g. "A", 3) would break the encoder.
    for col in categorical:
        X[col] = X[col].astype(str)

    if task_type == CLASSIFICATION:
        y = y.astype(str) if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y) else y
        if y.nunique() < 2:
            raise ValueError("The target has only one class; classification needs at least two.")

    stratified = task_type == CLASSIFICATION and can_stratify(y, test_size)
    if task_type == CLASSIFICATION and not stratified:
        notes.append("Some classes are too small to split proportionally, so the train/test split is random (not stratified).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratified else None
    )

    pipeline = build_pipeline(model_name, task_type, numeric, categorical, random_state)

    scoring = "accuracy" if task_type == CLASSIFICATION else "r2"
    cv_scores = None
    splitter = _cv_splitter(task_type, y_train, cv_folds, random_state)
    if splitter is not None and len(y_train) >= 4:
        cv_scores = cross_val_score(build_pipeline(model_name, task_type, numeric, categorical, random_state), X_train, y_train, cv=splitter, scoring=scoring)
    else:
        notes.append("Too little training data for cross-validation.")

    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    result = {
        "task_type": task_type,
        "model_name": model_name,
        "n_obs": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "stratified": stratified,
        "cv_metric": "Accuracy" if task_type == CLASSIFICATION else "R²",
        "cv_scores": None if cv_scores is None else cv_scores.tolist(),
        "y_test": np.asarray(y_test),
        "y_pred_test": np.asarray(y_pred_test),
        "importance": _grouped_importance(pipeline, numeric, categorical),
        "notes": notes,
    }

    if task_type == CLASSIFICATION:
        labels = list(pipeline.named_steps["model"].classes_)
        result.update({
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred_test, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred_test, average="weighted", zero_division=0),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_test, y_pred_test, labels=labels),
        })
    else:
        result.update({
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test) if len(y_test) >= 2 else float("nan"),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        })
    return result


@st.cache_data(show_spinner=False, max_entries=20)
def _train_cached(X: pd.DataFrame, y: pd.Series, model_name: str, task_type: str, test_size: float, random_state: int, cv_folds: int) -> dict:
    return train_and_evaluate(X, y, model_name, task_type, test_size, random_state, cv_folds)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

def run_ml_pipeline(df: pd.DataFrame, dependent_vars: list, independent_vars: list):
    """Run machine learning pipeline."""

    if not dependent_vars or not independent_vars:
        st.error("Please specify dependent and independent variables.")
        return

    if len(dependent_vars) > 1:
        target_var = st.selectbox("Target variable to predict:", dependent_vars)
    else:
        target_var = dependent_vars[0]
    features = [c for c in independent_vars if c != target_var]
    if not features:
        st.error("The target variable can't also be the only predictor.")
        return

    st.markdown(f"**Target Variable:** {target_var}")
    st.markdown(f"**Features:** {', '.join(map(str, features))}")

    # Prepare data
    X = df[features].copy()
    y = df[target_var].copy()

    valid_idx = (~X.isnull().any(axis=1)) & (~y.isnull())
    X = X[valid_idx]
    y = y[valid_idx]

    if len(X) < 5:
        st.error(f"Only {len(X)} complete rows after removing missing values; at least 5 are needed to train and test a model.")
        return

    st.info(f"Using {len(X)} observations after removing rows with missing values.")

    # Task type: detected, but the user can override it
    detected = detect_task_type(y)
    options = allowed_task_types(y)
    task_type = st.radio(
        "Task type:",
        options,
        index=options.index(detected),
        horizontal=True,
        format_func=lambda t: f"{t} (predict {'categories' if t == CLASSIFICATION else 'continuous values'})" + ("  ✓ suggested" if t == detected else ""),
        help="Suggested from the target: text or few repeated whole numbers → classification; "
             "decimals or many different values → regression.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test set size (%):", 10, 50, 20) / 100
    with col2:
        random_state = int(st.number_input("Random state (for reproducibility):", 0, 1000, 42))
    with col3:
        cv_folds = st.slider("Cross-validation folds:", 2, 10, DEFAULT_CV_FOLDS)

    selected_model = st.selectbox("Select model:", MODELS[task_type])

    train_key = (st.session_state.get("dataset_signature"), target_var, tuple(features), task_type, selected_model, test_size, random_state, cv_folds)

    if st.button("Train Model", type="primary"):
        st.session_state.ml_trained_key = train_key

    if st.session_state.get("ml_trained_key") != train_key:
        return

    try:
        with st.spinner("Training model..."):
            result = _train_cached(X, y, selected_model, task_type, test_size, random_state, cv_folds)
    except ValueError as exc:
        st.error(f"❌ {exc}")
        return

    for note in result["notes"]:
        st.warning(note)

    st.markdown("### Model Performance")
    split_text = "stratified by class" if result["stratified"] else "random"
    st.caption(
        f"{result['n_train']} training rows, {result['n_test']} test rows ({split_text} split). "
        + ("Numeric features standardised; " if selected_model != "Random Forest" and result["numeric_features"] else "")
        + ("categorical features one-hot encoded." if result["categorical_features"] else "")
    )

    show_cross_validation(result)

    if task_type == CLASSIFICATION:
        evaluate_classification(result)
    else:
        evaluate_regression(result)

    if result["importance"] is not None:
        show_feature_importance(result["importance"])


def show_cross_validation(result: dict):
    scores = result["cv_scores"]
    if not scores:
        return
    st.markdown("#### Cross-validation (training set)")
    mean, std = float(np.mean(scores)), float(np.std(scores))
    st.metric(f"Mean {result['cv_metric']} over {len(scores)} folds", f"{mean:.3f} ± {std:.3f}")
    st.caption("Each fold trains on part of the training data and scores on the rest. A large ± means results depend a lot on which rows are used.")


def evaluate_classification(result: dict):
    """Show classification metrics and confusion matrix."""

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Training Set:**")
        st.metric("Accuracy", f"{result['train_accuracy']:.3f}")
    with col2:
        st.markdown("**Test Set:**")
        st.metric("Accuracy", f"{result['test_accuracy']:.3f}")

    st.markdown("#### Classification Metrics (Test Set)")
    st.markdown(f"**Precision:** {result['precision']:.3f}")
    st.markdown(f"**Recall:** {result['recall']:.3f}")
    st.markdown(f"**F1-Score:** {result['f1']:.3f}")

    st.markdown("#### Confusion Matrix")
    labels = [str(label) for label in result["labels"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title("Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def evaluate_regression(result: dict):
    """Show regression metrics and diagnostic plots."""

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train R²", f"{result['train_r2']:.3f}")
    with col2:
        st.metric("Test R²", f"{result['test_r2']:.3f}")
    with col3:
        st.metric("Train RMSE", f"{result['train_rmse']:.3f}")
    with col4:
        st.metric("Test RMSE", f"{result['test_rmse']:.3f}")

    st.markdown("#### Actual vs Predicted Values")
    y_test, y_pred_test = result["y_test"], result["y_pred_test"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(y_test, y_pred_test, alpha=0.6, edgecolors="k")
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax1.set_xlabel("Actual", fontweight="bold")
    ax1.set_ylabel("Predicted", fontweight="bold")
    ax1.set_title("Actual vs Predicted", fontweight="bold")
    ax1.grid(alpha=0.3)

    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, alpha=0.6, edgecolors="k")
    ax2.axhline(y=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Predicted", fontweight="bold")
    ax2.set_ylabel("Residuals", fontweight="bold")
    ax2.set_title("Residual Plot", fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_feature_importance(importance: pd.DataFrame):
    """Display feature importance for tree-based models."""

    st.markdown("#### Feature Importance")
    st.caption("For categorical features, the importance of all their categories is added together.")
    st.dataframe(importance, width="stretch")

    fig, ax = plt.subplots(figsize=(10, 6))
    ordered = importance.iloc[::-1]
    ax.barh(ordered["Feature"], ordered["Importance"], color="steelblue", edgecolor="black")
    ax.set_xlabel("Importance Score", fontweight="bold")
    ax.set_title("Feature Importance", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
