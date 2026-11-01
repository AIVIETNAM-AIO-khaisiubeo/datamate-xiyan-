"""Parsers for the messy cell values found in Vietnamese shop exports.

Money:  '560,000'  '30.500đ'  '25k'  '1tr440'  '398,8604tr'  '45,000.0000'  '129 000'
Date:   datetime / Excel serial (45689.0 or '45689') / '01/02/2025' / '1 thg 6 2025' / '1/1' / ISO
Qty:    3.0 / '5 chai' / '1 bao 25kg' / '2 xe 4m3'
"""
from __future__ import annotations

import datetime as dt
import math
import re
from typing import Optional, Tuple

_EXCEL_EPOCH = dt.date(1899, 12, 30)

_NUM_CHARS = re.compile(r"[^0-9,.\-+]")


def is_blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() in ("", "nan", "None", "NaN", "NULL", "null"):
        return True
    try:  # pandas NA / NaT
        import pandas as pd
        if v is pd.NA or v is pd.NaT:
            return True
    except Exception:  # pragma: no cover
        pass
    return False


def _plain_number(s: str) -> Optional[float]:
    """Parse a number string that may use ',' or '.' as thousand/decimal separators."""
    s = s.strip().replace(" ", "").replace(" ", "")
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.startswith("-"):
        neg, s = True, s[1:]
    elif s.startswith("+"):
        s = s[1:]
    if not s or not re.fullmatch(r"[0-9.,]+", s):
        return None
    has_c, has_d = "," in s, "." in s
    if has_c and has_d:
        if s.rfind(",") > s.rfind("."):  # 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:  # 1,234.56
            s = s.replace(",", "")
    elif has_c:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif has_d:
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
            s = s.replace(".", "")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


def parse_money(v) -> Optional[float]:
    """Parse a VND money cell. Returns float (not rounded) or None."""
    if is_blank(v):
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    s = s.replace("vnđ", "").replace("vnd", "").replace("đồng", "").replace("đ", "").replace("₫", "")
    s = s.replace(" ", " ").strip()
    if not s:
        return None
    neg = s.startswith("-")
    if neg:
        s = s[1:].strip()
    m = re.fullmatch(r"([0-9.,]+)\s*(tr|triệu|trieu)\s*([0-9]*)", s)
    if m:
        txt = m.group(1)
        # in the 'tr' notation a single separator is always decimal: '3,255tr' = 3.255 million
        if txt.count(",") + txt.count(".") == 1:
            txt = txt.replace(",", ".")
        try:
            left = float(txt) if re.fullmatch(r"\d+(\.\d+)?", txt) else _plain_number(txt)
        except ValueError:
            left = None
        if left is None:
            return None
        right = m.group(3)
        if right:
            if len(right) > 6:
                return None
            val = left * 1_000_000 + int(right) * 10 ** (6 - len(right))
        else:
            val = left * 1_000_000
        val = float(round(val, 4))
        return -val if neg else val
    m = re.fullmatch(r"([0-9.,]+)\s*(k|nghìn|ngàn|nghin|ngan)", s)
    if m:
        txt = m.group(1)
        # '7,5k' decimal comma; '1,168k' thousands would be odd -> treat via _plain_number
        if re.fullmatch(r"\d+[,.]\d{1,2}", txt):
            base = float(txt.replace(",", "."))
        else:
            base = _plain_number(txt)
        if base is None:
            return None
        val = base * 1000
        return -val if neg else val
    x = _plain_number(s.replace(" ", ""))
    if x is None:
        return None
    return -x if neg else x


def parse_number(v) -> Optional[float]:
    """Generic numeric cell (quantities, factors, percents without '%')."""
    if is_blank(v):
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace("%", "")
    x = _plain_number(s)
    if x is not None:
        return x
    return parse_money(s)


_QTY_RE = re.compile(r"^\s*(-?[0-9]+(?:[.,][0-9]+)?)\s*(.*?)\s*$")


def parse_qty(v) -> Tuple[Optional[float], Optional[str]]:
    """'5 chai' -> (5.0, 'chai'); 3.0 -> (3.0, None); '1 bao 25kg' -> (1.0, 'bao 25kg')."""
    if is_blank(v):
        return None, None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v), None
    s = str(v).strip()
    m = _QTY_RE.match(s)
    if not m:
        return None, None
    num = m.group(1).replace(",", ".")
    try:
        q = float(num)
    except ValueError:
        return None, None
    unit = m.group(2).strip() or None
    return q, unit


_VI_MONTH = re.compile(r"^(\d{1,2})\s*thg\s*(\d{1,2})\s*(\d{4})$")


def excel_serial_to_date(x: float) -> Optional[dt.date]:
    try:
        if 20000 < x < 80000:
            return _EXCEL_EPOCH + dt.timedelta(days=int(x))
    except Exception:
        return None
    return None


def parse_datetime(v, default_year: Optional[int] = None, default_month: Optional[int] = None) -> Optional[dt.datetime]:
    """Parse a date/datetime cell. Day-first for slash formats (Vietnamese convention)."""
    if is_blank(v):
        return None
    try:
        import pandas as pd
        if isinstance(v, pd.Timestamp):
            return v.to_pydatetime()
    except Exception:  # pragma: no cover
        pass
    if isinstance(v, dt.datetime):
        return v
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day)
    if isinstance(v, dt.time):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        d = excel_serial_to_date(float(v))
        if d is None:
            return None
        frac = float(v) - int(v)
        secs = int(round(frac * 86400))
        return dt.datetime(d.year, d.month, d.day) + dt.timedelta(seconds=secs)
    s = str(v).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{5}(\.\d+)?", s):
        return parse_datetime(float(s))
    m = _VI_MONTH.match(s.lower())
    if m:
        try:
            return dt.datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?)?", s)
    if m:
        try:
            return dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                               int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0))
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?", s)
    if m:
        y = int(m.group(3))
        if y < 100:
            y += 2000
        try:
            return dt.datetime(y, int(m.group(2)), int(m.group(1)),
                               int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0))
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})", s)
    if m and default_year:
        try:
            return dt.datetime(default_year, int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        return None
    return None


def parse_date(v, default_year: Optional[int] = None) -> Optional[dt.date]:
    x = parse_datetime(v, default_year)
    return x.date() if x else None


def parse_time(v) -> Optional[dt.time]:
    if is_blank(v):
        return None
    if isinstance(v, dt.datetime):
        return v.time()
    if isinstance(v, dt.time):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        x = float(v)
        if 0 <= x < 1:
            secs = int(round(x * 86400))
            return dt.time(secs // 3600 % 24, secs // 60 % 60, secs % 60)
        d = parse_datetime(x)
        return d.time() if d else None
    s = str(v).strip()
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if h < 24 and mi < 60:
            return dt.time(h, mi, se)
    return None


def parse_bool(v) -> Optional[bool]:
    if is_blank(v):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "x", "có", "co", "yes", "y", "đúng", "dung", "kê đơn", "ke don", "rx"):
        return True
    if s in ("false", "0", "không", "khong", "no", "n", "sai", ""):
        return False
    return None


def round_half_up(x: float, ndigits: int = 0) -> float:
    """Round half away from zero (rulebook: 'nửa đồng làm tròn lên')."""
    if x is None:
        return None
    q = 10 ** ndigits
    y = abs(x) * q
    r = math.floor(y + 0.5 + 1e-9)
    r = r / q
    return -r if x < 0 else r
