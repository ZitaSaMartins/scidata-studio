"""Shared helpers: data prep, significance labelling, and common plots."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.stat_format import format_p_value


def _columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = df.select_dtypes(include=np.number).columns.astype(str).tolist()
    categorical = [str(column) for column in df.columns if str(column) not in numeric]
    return numeric, categorical


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column labels are strings before any selector runs.

    Selectors across this package stringify column labels for display and then
    index the original frame with the selected string. If the frame's actual
    labels are not already strings (for example integer headers preserved by
    ``pd.read_excel``), that indexing raises ``KeyError``. Normalizing once at
    each entry point keeps the displayed and actual labels consistent.
    """
    if all(isinstance(column, str) for column in df.columns):
        return df
    return df.rename(columns=str)


def _need(condition: bool, message: str) -> bool:
    if not condition:
        st.error(message)
    return condition


def _frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return df.loc[:, list(columns)].dropna().copy()


def _interpret_p(p_value: float, alpha: float, subject: str = "products") -> None:
    if not np.isfinite(p_value):
        st.error("The test produced an invalid (non-finite) p-value. Check the selected data.")
    elif p_value < alpha:
        st.success(f"Evidence of differences among {subject} (p={format_p_value(p_value)}).")
    else:
        st.info(f"No statistically significant difference among {subject} (p={format_p_value(p_value)}).")


def _compact_letters(
    labels: Sequence[str],
    significant_pairs: set[tuple[str, str]],
) -> dict[str, str]:
    """Assign letters so statistically different groups share no letter."""
    ordered = [str(label) for label in labels]

    def pair(first: str, second: str) -> tuple[str, str]:
        return tuple(sorted((first, second)))

    compatible = {
        label: {
            other
            for other in ordered
            if other != label and pair(label, other) not in significant_pairs
        }
        for label in ordered
    }
    cliques: list[set[str]] = []

    def find(current: set[str], candidates: set[str], excluded: set[str]) -> None:
        if not candidates and not excluded:
            cliques.append(current)
            return
        remaining = set(candidates)
        pivot = max(candidates | excluded, key=lambda label: len(candidates & compatible[label]))
        for label in candidates - compatible[pivot]:
            find(
                current | {label},
                remaining & compatible[label],
                excluded & compatible[label],
            )
            remaining.remove(label)
            excluded.add(label)

    find(set(), set(ordered), set())
    cliques.sort(key=lambda values: (-len(values), sorted(values)))
    result: dict[str, list[str]] = {label: [] for label in ordered}
    for index, clique in enumerate(cliques):
        letter = chr(65 + index) if index < 26 else f"A{index - 25}"
        for label in clique:
            result[label].append(letter)
    return {label: "".join(result[label]) or "—" for label in ordered}


def heatmap_height(n_rows: int) -> int:
    """Pixel height for a heatmap: ~30px per row plus room for title and axis, kept within one screen."""
    return int(min(650, max(260, 30 * n_rows + 140)))


def _heatmap(data: pd.DataFrame, title: str, fmt: str = ".2f") -> None:
    """Annotated heatmap that fits the page.

    Plotly with ``aspect="auto"`` stretches to the container width, and the
    height grows with the number of rows but is capped, so large panels
    (e.g. 15 assessors x 6 descriptors) stay readable without scrolling.
    """
    frame = data.copy()
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    fig = px.imshow(
        frame,
        text_auto=fmt,
        aspect="auto",
        color_continuous_scale="RdYlBu_r",
        labels={"x": frame.columns.name or "", "y": frame.index.name or "", "color": "Value"},
    )
    fig.update_xaxes(type="category", side="bottom", tickangle=-30 if frame.shape[1] > 8 else 0)
    fig.update_yaxes(type="category")
    fig.update_layout(title=title, height=heatmap_height(frame.shape[0]), margin={"l": 10, "r": 10, "t": 50, "b": 10})
    if frame.size > 300:  # too many cells to label legibly
        fig.update_traces(texttemplate=None)
    st.plotly_chart(fig, width="stretch")


def _scatter_labels(
    coordinates: np.ndarray,
    labels: Sequence[object],
    title: str,
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=55)
    for label, (x_value, y_value) in zip(labels, coordinates[:, :2], strict=False):
        ax.annotate(str(label), (x_value, y_value), xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.axvline(0, color="grey", linewidth=0.7)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

