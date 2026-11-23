"""Reading uploaded data files (CSV and Excel) into DataFrames.

Kept free of Streamlit so it can be unit tested. The upload page wraps these
functions with ``st.cache_data``.

Highlights:
* File extensions are matched case-insensitively (``DATA.CSV`` works).
* CSV separator (``,`` ``;`` tab ``|``) and decimal mark (``.`` or ``,``) are
  detected automatically, so European exports such as ``1,5;2,7`` load as
  numbers instead of a single text column.
* Excel files (.xlsx, .xlsm, .xls, .xlsb, .ods) are read with the ``calamine``
  engine, which also handles legacy ``.xls`` files without needing ``xlrd``.
  Hidden sheets (e.g. XLSTAT's ``*_HID`` sheets) are skipped when listing.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CSV_EXTENSIONS = {".csv", ".txt", ".tsv"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".xlsb", ".ods"}
SUPPORTED_EXTENSIONS = CSV_EXTENSIONS | EXCEL_EXTENSIONS

CANDIDATE_SEPARATORS = (",", ";", "\t", "|")
_SAMPLE_LINES = 50

_COMMA_DECIMAL = re.compile(r"^[-+]?\d+,\d+$")
_DOT_DECIMAL = re.compile(r"^[-+]?\d+\.\d+$")
_EUROPEAN_THOUSANDS = re.compile(r"^[-+]?\d{1,3}(\.\d{3})+(,\d+)?$")


@dataclass(frozen=True)
class CsvFormat:
    sep: str
    decimal: str
    thousands: str | None = None


def file_extension(filename: str) -> str:
    """Lower-cased extension including the dot (``'DATA.CSV'`` -> ``'.csv'``)."""
    return Path(filename).suffix.lower()


def is_excel(filename: str) -> bool:
    return file_extension(filename) in EXCEL_EXTENSIONS


def decode_text(raw: bytes) -> str:
    """Decode CSV bytes, trying UTF-8 (with or without BOM) then Windows-1252."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _sample_rows(text: str, sep: str) -> list[list[str]]:
    lines = [line for line in text.splitlines() if line.strip()][:_SAMPLE_LINES]
    try:
        return list(csv.reader(lines, delimiter=sep))
    except csv.Error:
        return []


def detect_separator(text: str) -> str:
    """Pick the separator that splits the sample lines most consistently.

    Each candidate is scored by how many fields it produces per line. A
    separator that gives the same number of fields (>1) on every line wins;
    ties go to the one producing more fields. Quoted fields are respected, so
    ``"1,5";"2,5"`` is correctly seen as ``;``-separated.
    """
    best_sep, best_score = ",", (False, 0)
    for sep in CANDIDATE_SEPARATORS:
        rows = _sample_rows(text, sep)
        if not rows:
            continue
        counts = [len(row) for row in rows]
        if max(counts) <= 1:
            continue
        consistent = min(counts) == max(counts)
        score = (consistent, min(counts) if consistent else sorted(counts)[len(counts) // 2])
        if score > best_score:
            best_sep, best_score = sep, score
    return best_sep


def detect_decimal(text: str, sep: str) -> tuple[str, str | None]:
    """Return ``(decimal, thousands)`` for the given separator.

    A comma can only be a decimal mark when it is not the field separator.
    We count values that look like ``1,5`` versus ``1.5`` and pick the more
    common style. ``1.234,5`` style values also turn on ``.`` as thousands.
    """
    if sep == ",":
        return ".", None
    comma = dot = thousands = 0
    for row in _sample_rows(text, sep)[1:]:  # skip header
        for value in row:
            value = value.strip()
            if _COMMA_DECIMAL.match(value):
                comma += 1
            elif _EUROPEAN_THOUSANDS.match(value) and "," in value:
                comma += 1
                thousands += 1
            elif _DOT_DECIMAL.match(value):
                dot += 1
    if comma > dot:
        return ",", ("." if thousands else None)
    return ".", None


def detect_csv_format(text: str) -> CsvFormat:
    sep = detect_separator(text)
    decimal, thousands = detect_decimal(text, sep)
    return CsvFormat(sep=sep, decimal=decimal, thousands=thousands)


def read_csv_bytes(raw: bytes, sep: str | None = None, decimal: str | None = None) -> tuple[pd.DataFrame, CsvFormat]:
    """Read CSV bytes, auto-detecting anything not given explicitly."""
    text = decode_text(raw)
    detected = detect_csv_format(text)
    use_sep = sep or detected.sep
    if decimal:
        use_decimal = decimal
        thousands = "." if decimal == "," and detected.thousands else None
    elif sep and sep != detected.sep:
        use_decimal, thousands = detect_decimal(text, use_sep)
    else:
        use_decimal, thousands = detected.decimal, detected.thousands
    fmt = CsvFormat(sep=use_sep, decimal=use_decimal, thousands=thousands)
    df = pd.read_csv(io.StringIO(text), sep=fmt.sep, decimal=fmt.decimal, thousands=fmt.thousands)
    return df, fmt


def list_excel_sheets(raw: bytes) -> list[str]:
    """Names of the visible worksheets in an Excel workbook, in workbook order."""
    from python_calamine import CalamineWorkbook

    workbook = CalamineWorkbook.from_filelike(io.BytesIO(raw))
    visible = [
        sheet.name
        for sheet in workbook.sheets_metadata
        if str(sheet.visible).endswith("Visible") and str(sheet.typ).endswith("WorkSheet")
    ]
    return visible or list(workbook.sheet_names)


def read_excel_bytes(raw: bytes, sheet: str | int = 0) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(raw), sheet_name=sheet, engine="calamine")


def load_table(filename: str, raw: bytes, sheet: str | int | None = None) -> pd.DataFrame:
    """Load a CSV or Excel upload into a DataFrame, dispatching on extension."""
    ext = file_extension(filename)
    if ext in CSV_EXTENSIONS:
        return read_csv_bytes(raw)[0]
    if ext in EXCEL_EXTENSIONS:
        if sheet is None:
            sheets = list_excel_sheets(raw)
            sheet = sheets[0] if sheets else 0
        return read_excel_bytes(raw, sheet)
    raise ValueError(f"Unsupported file type '{ext or filename}'. Use CSV or Excel ({', '.join(sorted(SUPPORTED_EXTENSIONS))}).")
