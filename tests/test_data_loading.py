"""Tests for modules.data_loading (CSV/Excel upload parsing)."""

import io
from pathlib import Path

import pandas as pd
import pytest

from modules.data_loading import (
    detect_csv_format,
    file_extension,
    is_excel,
    list_excel_sheets,
    load_table,
    read_csv_bytes,
)

REPO = Path(__file__).resolve().parent.parent


def test_extension_is_case_insensitive():
    assert file_extension("DATA.CSV") == ".csv"
    assert is_excel("Book.XLSX")
    assert not is_excel("export.Csv")


def test_uppercase_csv_extension_is_read_as_csv():
    df = load_table("RESULTS.CSV", b"a,b\n1,2\n3,4\n")
    assert list(df.columns) == ["a", "b"]
    assert df["b"].tolist() == [2, 4]


def test_standard_csv():
    df, fmt = read_csv_bytes(b"product,score\nA,7.5\nB,8.25\n")
    assert (fmt.sep, fmt.decimal) == (",", ".")
    assert df["score"].tolist() == [7.5, 8.25]


def test_european_csv_semicolon_and_decimal_comma():
    raw = "produto;nota;temperatura\nA;7,5;23\nB;8,25;24,1\n".encode()
    df, fmt = read_csv_bytes(raw)
    assert (fmt.sep, fmt.decimal) == (";", ",")
    assert pd.api.types.is_float_dtype(df["nota"])
    assert df["nota"].tolist() == [7.5, 8.25]
    assert df["temperatura"].tolist() == [23.0, 24.1]


def test_european_csv_with_thousands_separator():
    df, fmt = read_csv_bytes(b"item;valor\nA;1.234,5\nB;12,0\n")
    assert fmt.thousands == "."
    assert df["valor"].tolist() == [1234.5, 12.0]


def test_semicolon_csv_with_decimal_point_keeps_point():
    df, fmt = read_csv_bytes(b"a;b\n1.5;2\n3.5;4\n")
    assert (fmt.sep, fmt.decimal) == (";", ".")
    assert df["a"].tolist() == [1.5, 3.5]


def test_tab_separated():
    df, fmt = read_csv_bytes(b"a\tb\n1\t2\n")
    assert fmt.sep == "\t"
    assert list(df.columns) == ["a", "b"]


def test_quoted_commas_do_not_confuse_detection():
    text = 'name;score\n"Smith, J";7,5\n"Doe, A";8,0\n'
    fmt = detect_csv_format(text)
    assert (fmt.sep, fmt.decimal) == (";", ",")


def test_windows_1252_encoding_with_accents():
    raw = "produto;açúcar\nPão;3,5\n".encode("cp1252")
    df, _ = read_csv_bytes(raw)
    assert list(df.columns) == ["produto", "açúcar"]
    assert df.iloc[0, 0] == "Pão"


def test_utf8_bom_is_stripped():
    df, _ = read_csv_bytes("﻿a,b\n1,2\n".encode())
    assert list(df.columns) == ["a", "b"]


def test_manual_override_beats_detection():
    df, fmt = read_csv_bytes(b"a;b\n1,5;2\n", sep=";", decimal=".")
    assert fmt.decimal == "."
    assert df["a"].tolist() == ["1,5"]


def _xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


def test_excel_sheet_selection():
    raw = _xlsx_bytes({"First": pd.DataFrame({"x": [1, 2]}), "Second": pd.DataFrame({"y": [3, 4]})})
    assert list_excel_sheets(raw) == ["First", "Second"]
    assert list(load_table("book.xlsx", raw).columns) == ["x"]
    assert load_table("BOOK.XLSX", raw, sheet="Second")["y"].tolist() == [3, 4]


def test_hidden_excel_sheets_are_not_listed():
    from openpyxl import Workbook

    wb = Workbook()
    wb.active.title = "Data"
    wb.active.append(["a"])
    hidden = wb.create_sheet("Calc_HID")
    hidden.sheet_state = "hidden"
    buffer = io.BytesIO()
    wb.save(buffer)
    assert list_excel_sheets(buffer.getvalue()) == ["Data"]


@pytest.mark.skipif(not (REPO / "tuto_PAN_EN.xls").exists(), reason="sample .xls not present")
def test_legacy_xls_file_loads():
    raw = (REPO / "tuto_PAN_EN.xls").read_bytes()
    sheets = list_excel_sheets(raw)
    assert sheets[0] == "Data"
    assert not any(name.endswith("_HID") for name in sheets)
    df = load_table("tuto_PAN_EN.xls", raw)
    assert not df.empty


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        load_table("notes.pdf", b"%PDF")
