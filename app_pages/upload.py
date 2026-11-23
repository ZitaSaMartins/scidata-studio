import hashlib

import pandas as pd
import streamlit as st

from app_pages.state import set_page
from modules.data_loading import (
    SUPPORTED_EXTENSIONS,
    is_excel,
    list_excel_sheets,
    read_csv_bytes,
    read_excel_bytes,
)

_SEPARATOR_LABELS = {"Auto": None, "Comma ( , )": ",", "Semicolon ( ; )": ";", "Tab": "\t", "Pipe ( | )": "|"}
_DECIMAL_LABELS = {"Auto": None, "Point ( . )": ".", "Comma ( , )": ","}
_SEP_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}


@st.cache_data(show_spinner=False)
def _cached_sheets(raw: bytes) -> list[str]:
    return list_excel_sheets(raw)


@st.cache_data(show_spinner=False)
def _cached_excel(raw: bytes, sheet: str) -> pd.DataFrame:
    return read_excel_bytes(raw, sheet)


@st.cache_data(show_spinner=False)
def _cached_csv(raw: bytes, sep: str | None, decimal: str | None):
    return read_csv_bytes(raw, sep=sep, decimal=decimal)


def _read_upload(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Return the DataFrame and a string describing how it was read."""
    raw = uploaded_file.getvalue()
    if is_excel(uploaded_file.name):
        sheets = _cached_sheets(raw)
        sheet = sheets[0]
        if len(sheets) > 1:
            sheet = st.selectbox(
                "📑 Sheet to load:",
                sheets,
                key=f"upload_sheet_{uploaded_file.file_id}",
                help="This workbook has several sheets. Choose the one that holds your data.",
            )
        return _cached_excel(raw, sheet), f"sheet:{sheet}"

    with st.expander("⚙️ CSV options (separator and decimal mark)"):
        st.caption("Leave on **Auto** unless the preview below looks wrong, e.g. everything in one column.")
        c1, c2 = st.columns(2)
        sep_label = c1.selectbox("Column separator", list(_SEPARATOR_LABELS), key="upload_csv_sep")
        dec_label = c2.selectbox("Decimal mark", list(_DECIMAL_LABELS), key="upload_csv_decimal")
    df, fmt = _cached_csv(raw, _SEPARATOR_LABELS[sep_label], _DECIMAL_LABELS[dec_label])
    st.caption(
        f"Read as CSV with **{_SEP_NAMES.get(fmt.sep, repr(fmt.sep))}** separator "
        f"and **{'comma' if fmt.decimal == ',' else 'point'}** decimal mark."
    )
    return df, f"csv:{fmt.sep}:{fmt.decimal}"


def render() -> None:
    st.markdown("## 📤 Upload Your Data")
    st.markdown("Choose an Excel or CSV file with your data")

    if st.session_state.uploaded_filename:
        st.caption(f"📄 Currently loaded file: **{st.session_state.uploaded_filename}**")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "📎 Select file:",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
            help="Supported formats: Excel (.xlsx, .xlsm, .xls, .xlsb, .ods) or CSV/TXT. "
                 "European CSV files (; separator, decimal comma) are detected automatically."
        )

    if uploaded_file is not None:
        try:
            df, read_options = _read_upload(uploaded_file)

            signature_source = uploaded_file.getvalue() + read_options.encode()
            dataset_signature = hashlib.sha256(signature_source).hexdigest()
            dataset_changed = dataset_signature != st.session_state.dataset_signature
            if dataset_changed:
                st.session_state.distribution_test_result = None
                st.session_state.variance_test_result = None
                st.session_state.ml_trained_key = None
                st.session_state.dataset_signature = dataset_signature

            st.session_state.df = df
            st.session_state.uploaded_filename = uploaded_file.name
            st.success(f"✅ Successfully loaded: {uploaded_file.name}")

            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("📋 Total Columns", df.shape[1])
            with col3:
                st.metric("⚠️ Missing Values", int(df.isnull().sum().sum()))

            # Preview
            st.markdown("### Preview of Your Data")
            st.dataframe(df, height=450, width="stretch")

            st.info("✨ Great! Your data is uploaded. Next step: Rename variables (optional) →")

            if st.button("✏️ Rename Variables", type="primary"):
                set_page("rename")

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Try saving your file as CSV or Excel format")
    else:
        st.info("👆 Click above to select your file")

        with st.expander("ℹ️ What file should I use?"):
            st.markdown("""
            **Accepted formats:**
            - Excel: .xlsx, .xlsm, .xls, .xlsb, .ods (you can pick the sheet)
            - Text: .csv, .txt, .tsv (comma, semicolon, tab or pipe separated; decimal point or comma)

            **Requirements:**
            - First row: Column names (headers)
            - Data below: Your actual data
            - No blank rows in the middle

            **Example:**
            | Product | Score | Temperature |
            |---------|-------|-------------|
            | A | 7.5 | 23 |
            | B | 8.2 | 24 |
            """)
