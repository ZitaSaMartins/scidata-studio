import streamlit as st

from modules.eda import CHART_CATALOG


def init_state() -> None:
    if "df" not in st.session_state:
        st.session_state.df = None
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None
    if 'validation_report' not in st.session_state:
        st.session_state.validation_report = None
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if 'eda_category' not in st.session_state:
        st.session_state.eda_category = list(CHART_CATALOG.keys())[0]
    if 'dataset_signature' not in st.session_state:
        st.session_state.dataset_signature = None
    legacy_eda_pages = {
        "Distribution": "eda_distributions",
        "Correlation": "eda_correlations",
        "Comparison": "eda_comparisons",
        "Part-to-Whole & Hierarchical": "eda_hierarchical",
        "Temporal": "eda_temporal",
    }
    legacy_page_map = {
        "eda": legacy_eda_pages.get(st.session_state.eda_category, "eda_comparisons"),
        "hypothesis": "test_distribution",
        "sensory": "sensory_qda",
        "sensory_descriptive": "sensory_qda",
        "sensory_comparison": "sensory_product_characterization",
        "ml": "machine_learning",
        "sensory_panel_discrimination": "sensory_panel_report",
        "sensory_panel_repeatability": "sensory_panel_report",
        "sensory_panel_agreement": "sensory_panel_report",
        "sensory_panel_scale": "sensory_panel_report",
        "sensory_discrimination_power": "sensory_panel_report",
        "sensory_product_profiles": "sensory_panel_report",
        "sensory_panel": "sensory_panel_report",
        "sensory_panel_pca": "sensory_panel_report",
        "sensory_panel_assessor_comparison": "sensory_panel_report",
        "sensory_consensus_distance": "sensory_panel_report",
        "sensory_panel_clustering": "sensory_panel_report",
        "sensory_run_panel_analysis": "sensory_panel_report",
        "recommend": "recommendations",
        "help_data": "help_getting_started",
        "help_analysis": "help_analysis_types",
    }
    if st.session_state.page in legacy_page_map:
        st.session_state.page = legacy_page_map[st.session_state.page]


def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()


def store_eda_independent_selection():
    st.session_state.eda_independent_selection = list(
        st.session_state.get("_eda_independent_vars", [])
    )


def store_eda_dependent_selection():
    st.session_state.eda_dependent_selection = list(
        st.session_state.get("_eda_dependent_vars", [])
    )
