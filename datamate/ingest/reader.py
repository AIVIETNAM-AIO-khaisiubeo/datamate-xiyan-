"""Read raw files (xlsx / csv / sqlite) into header-detected tables.

The exports are deliberately messy: title rows above the header, two-row merged headers,
daily/monthly total rows, trailing note rows, one sheet per month, multi-file splits.
This module only finds the header and the data rows; meaning is assigned in ``roles.py``.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..textnorm import fold

_TOTAL_RE = re.compile(r"^\s*(tổng|tong|cộng|cong|total|sum)\b", re.I)
_NOTE_RE = re.compile(r"^\s*(ghi chú|ghi chu|lưu ý|luu y|note|chú thích|chu thich)\s*:", re.I)


@dataclass
class RawTable:
    file: str              # file name (no dir)
    sheet: Optional[str]   # sheet / sqlite table name
    columns: List[str]
    df: pd.DataFrame       # object dtype, raw cell values, header applied
    titles: List[str] = field(default_factory=list)   # text rows above the header
    group_header: Optional[List[str]] = None           # merged group row (e.g. Hàng / Tiền / Đơn)
    source: str = "xlsx"
    sheet_year: Optional[int] = None
    sheet_month: Optional[int] = None

    @property
    def name(self) -> str:
        return f"{self.file}::{self.sheet}" if self.sheet else self.file


def _cell_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    return str(v).strip()


def _is_texty(v) -> bool:
    s = _cell_str(v)
    if not s or not isinstance(v, str):
        return False
    return bool(re.search(r"[A-Za-zÀ-ỹđĐ]", s))


def detect_header(rows: List[list], max_scan: int = 12) -> int:
    """Index of the header row: the early row with most text cells, followed by data."""
    best, best_score = 0, -1
    n = min(len(rows), max_scan)
    for i in range(n):
        r = rows[i]
        texty = sum(1 for v in r if _is_texty(v))
        filled = sum(1 for v in r if _cell_str(v))
        if filled == 0:
            continue
        # header cells are short labels; titles are long sentences
        long_cells = sum(1 for v in r if len(_cell_str(v)) > 45)
        score = texty * 2 + filled - long_cells * 3
        # all cells distinct and texty is typical for a header
        if texty == filled and texty >= 2:
            score += 2
        if score > best_score:
            best, best_score = i, score
    return best


def sheet_period(name: str) -> (Optional[int], Optional[int]):
    """'T01_2025' / '04.2025' / 'Thang 02 2025' / 'Tháng 01-2025' / 'BanHang_T10_2025' -> (year, month)."""
    if not name:
        return None, None
    s = fold(name)
    m = re.search(r"(?:^|[^0-9])(\d{1,2})[ ._\-/](20\d{2})(?:[^0-9]|$)", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))
    m = re.search(r"(20\d{2})[ ._\-/](\d{1,2})(?:[^0-9]|$)", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(?:^|[^0-9])(20\d{2})(?:[^0-9]|$)", s)
    if m:
        return int(m.group(1)), None
    return None, None


def _rows_to_table(rows: List[list], file: str, sheet: Optional[str], source: str) -> Optional[RawTable]:
    rows = [list(r) for r in rows]
    while rows and not any(_cell_str(v) for v in rows[-1]):
        rows.pop()
    if not rows:
        return None
    h = detect_header(rows)
    header_raw = rows[h]
    titles = [" ".join(_cell_str(v) for v in r if _cell_str(v)) for r in rows[:h]]
    titles = [t for t in titles if t]
    group_header = None
    if h >= 1:
        prev = rows[h - 1]
        filled = [v for v in prev if _cell_str(v)]
        if 1 <= len(filled) < len([v for v in header_raw if _cell_str(v)]) and all(len(_cell_str(v)) < 25 for v in filled):
            group_header = [_cell_str(v) for v in prev]
    width = max(len(r) for r in rows[h:])
    cols, seen = [], {}
    for j in range(width):
        c = _cell_str(header_raw[j]) if j < len(header_raw) else ""
        if not c:
            c = f"col{j}"
        if c in seen:
            seen[c] += 1
            c = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    data = []
    header_fold = [fold(c) for c in cols]
    for r in rows[h + 1:]:
        r = list(r) + [None] * (width - len(r))
        strs = [_cell_str(v) for v in r]
        if not any(strs):
            continue
        if [fold(s) for s in strs] == header_fold:
            continue  # repeated header
        data.append(r[:width])
    df = pd.DataFrame(data, columns=cols, dtype=object)
    # drop fully empty columns that have no header
    for c in list(df.columns):
        if c.startswith("col") and re.fullmatch(r"col\d+", c) and df[c].map(_cell_str).eq("").all():
            df = df.drop(columns=[c])
    y, m = sheet_period(sheet or "")
    if y is None:
        y, m = sheet_period(os.path.splitext(file)[0])
    return RawTable(file=file, sheet=sheet, columns=list(df.columns), df=df, titles=titles,
                    group_header=group_header, source=source, sheet_year=y, sheet_month=m)


def drop_total_and_note_rows(t: RawTable, id_cols: List[str]) -> int:
    """Remove 'Tổng ngày 02/06' / 'Cộng' / 'Ghi chú: ...' rows. Returns number removed."""
    df = t.df
    if df.empty:
        return 0
    strs = df.apply(lambda col: col.map(_cell_str))
    nonempty = strs.ne("").sum(axis=1)
    is_total = pd.Series(False, index=df.index)
    for c in df.columns:
        is_total |= strs[c].str.match(_TOTAL_RE) & (strs[c].str.len() < 60)
    if id_cols:
        ids_blank = pd.Series(True, index=df.index)
        for c in id_cols:
            if c in strs:
                ids_blank &= ~strs[c].str.match(r"^\s*[A-Z]{1,4}\d{3,}\s*$")
        is_total &= ids_blank
    is_note = pd.Series(False, index=df.index)
    for c in df.columns:
        is_note |= strs[c].str.match(_NOTE_RE)
    is_note &= nonempty <= 2
    drop = is_total | is_note
    n = int(drop.sum())
    if n:
        t.df = df.loc[~drop].reset_index(drop=True)
    return n


def _xlsx_sheets(path: str):
    """Yield (sheet_name, rows); python-calamine (fast) with an openpyxl fallback."""
    try:
        from python_calamine import CalamineWorkbook
        wb = CalamineWorkbook.from_path(path)
        for sn in wb.sheet_names:
            sh = wb.get_sheet_by_name(sn)
            try:
                rows = sh.to_python(skip_empty_area=False)
            except TypeError:
                rows = sh.to_python()
            yield sn, rows
        return
    except ImportError:
        pass
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for ws in wb.worksheets:
        yield ws.title, [list(r) for r in ws.iter_rows(values_only=True)]


def read_xlsx(path: str) -> List[RawTable]:
    out = []
    for sn, rows in _xlsx_sheets(path):
        t = _rows_to_table(rows, os.path.basename(path), sn, "xlsx")
        if t is not None and len(t.df.columns) >= 2:
            out.append(t)
    return out


def read_csv(path: str) -> List[RawTable]:
    with open(path, "rb") as fh:
        raw = fh.read()
    for enc in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sample = text[:20000]
    delim = ","
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        pass
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    t = _rows_to_table(rows, os.path.basename(path), None, "csv")
    return [t] if t is not None else []


def read_sqlite(path: str) -> List[RawTable]:
    out = []
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [r[0] for r in con.execute("select name from sqlite_master where type in ('table','view')")]
        for n in names:
            df = pd.read_sql_query(f'select * from "{n}"', con)
            df = df.astype(object).where(df.notna(), None)
            out.append(RawTable(file=os.path.basename(path), sheet=n, columns=list(df.columns), df=df,
                                source="sqlite"))
    finally:
        con.close()
    return out


def _cache_path(biz_dir: str) -> Optional[str]:
    root = os.environ.get("DATAMATE_CACHE")
    if not root:
        return None
    import hashlib
    h = hashlib.md5()
    for f in sorted(os.listdir(biz_dir)):
        p = os.path.join(biz_dir, f)
        if os.path.isfile(p):
            st = os.stat(p)
            h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode())
    h.update(b"reader-v3")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"raw_{os.path.basename(os.path.normpath(biz_dir))}_{h.hexdigest()[:12]}.pkl")


def read_business_files(biz_dir: str) -> List[RawTable]:
    cp = _cache_path(biz_dir)
    if cp and os.path.exists(cp):
        import pickle
        try:
            with open(cp, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    tables = _read_business_files(biz_dir)
    if cp:
        import pickle
        try:
            with open(cp, "wb") as fh:
                pickle.dump(tables, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return tables


def _read_business_files(biz_dir: str) -> List[RawTable]:
    tables: List[RawTable] = []
    for f in sorted(os.listdir(biz_dir)):
        p = os.path.join(biz_dir, f)
        if not os.path.isfile(p):
            continue
        low = f.lower()
        try:
            if low.endswith((".xlsx", ".xlsm", ".xls")):
                tables.extend(read_xlsx(p))
            elif low.endswith((".csv", ".tsv", ".txt")):
                tables.extend(read_csv(p))
            elif low.endswith((".sqlite", ".db", ".sqlite3")):
                tables.extend(read_sqlite(p))
        except Exception as e:  # keep going; a broken file must not kill the business
            print(f"[reader] failed {p}: {e}")
    return tables
