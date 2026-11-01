"""Deterministic resolution of Vietnamese time expressions against the as-of date.

Implements the rulebook §2 conventions: weeks start Monday, 'tuần trước' = whole previous
week, 'tháng này' = month-to-date, 'N ngày gần nhất' includes today, month/quarter without a
year = this year unless that is after today, Tết window from the solar–lunar calendar, etc.
All functions work on the *normalised* question (see textnorm.normalize_question).
"""
from __future__ import annotations

import calendar
import datetime as dt
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

D = dt.date


@dataclass
class Period:
    start: D
    end: D
    label: str = ""
    kind: str = "range"     # day / week / month / quarter / year / range / tet / mtd
    full_end: Optional[D] = None  # calendar end of the period (month end for 'tháng này')

    def days(self) -> int:
        return (self.end - self.start).days + 1

    def __repr__(self):
        return f"Period({self.start}..{self.end},{self.kind},{self.label})"


def month_end(y: int, m: int) -> D:
    return D(y, m, calendar.monthrange(y, m)[1])


def month_period(y: int, m: int, label: str = "") -> Period:
    return Period(D(y, m, 1), month_end(y, m), label or f"{m}/{y}", "month", month_end(y, m))


def prev_month(y: int, m: int) -> Tuple[int, int]:
    return (y - 1, 12) if m == 1 else (y, m - 1)


def add_months(y: int, m: int, k: int) -> Tuple[int, int]:
    t = y * 12 + (m - 1) + k
    return t // 12, t % 12 + 1


def week_monday(d: D) -> D:
    return d - dt.timedelta(days=d.weekday())


# ------------------------------------------------------------------ lunar calendar / Tết
class LunarCalendar:
    def __init__(self, path: Optional[str] = None):
        self.rows = {}
        self.by_lunar = {}
        if path and os.path.exists(path):
            import csv
            with open(path, encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    try:
                        d = D.fromisoformat(r["date"])
                    except Exception:
                        continue
                    ld, lm, ly = int(r["lunar_day"]), int(r["lunar_month"]), int(r["lunar_year"])
                    leap = str(r.get("lunar_leap", "")).strip().lower() in ("true", "1")
                    self.rows[d] = (ld, lm, ly, leap, r.get("holiday") or "")
                    if not leap:
                        self.by_lunar[(ly, lm, ld)] = d

    def solar(self, ly: int, lm: int, ld: int) -> Optional[D]:
        return self.by_lunar.get((ly, lm, ld))

    def tet_window(self, year: int) -> Optional[Tuple[D, D]]:
        """23 tháng Chạp (lunar year Y-1) .. mùng 10 tháng Giêng (lunar year Y)."""
        a = self.solar(year - 1, 12, 23)
        b = self.solar(year, 1, 10)
        if a is None:
            # month 12 may have 29 days; 23 always exists, so missing means out of calendar range
            return None
        if b is None:
            return None
        return a, b

    def lunar_year_of(self, d: D) -> Optional[int]:
        r = self.rows.get(d)
        return r[2] if r else None


# ------------------------------------------------------------------ parsing helpers
_MONTH_RE = re.compile(r"\bthang (\d{1,2})/(\d{4})\b")
_MONTH_NOYEAR_RE = re.compile(r"\bthang (\d{1,2})\b(?!\s*/)(?!\s*nam)")
_QUARTER_RE = re.compile(r"\bquy (\d)\s*/\s*(\d{4})\b")
_QUARTER_NOYEAR_RE = re.compile(r"\bquy (\d)\b(?!\s*/)")
_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_DM_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b(?!/)")


def _mk_date(dd, mm, yy) -> Optional[D]:
    try:
        return D(int(yy), int(mm), int(dd))
    except ValueError:
        return None


def resolve_month_noyear(m: int, today: D) -> Tuple[int, int]:
    y = today.year
    if D(y, m, 1) > today:
        y -= 1
    return y, m


def resolve_quarter_noyear(q: int, today: D) -> int:
    y = today.year
    if D(y, 3 * (q - 1) + 1, 1) > today:
        y -= 1
    return y


def quarter_period(y: int, q: int) -> Period:
    m1 = 3 * (q - 1) + 1
    return Period(D(y, m1, 1), month_end(y, m1 + 2), f"Q{q}/{y}", "quarter", month_end(y, m1 + 2))


@dataclass
class TimeInfo:
    periods: List[Period] = field(default_factory=list)   # in order of mention
    snapshot: Optional[D] = None                          # "đến hết ngày X" / "cuối ngày hôm nay"
    mtd_compare: bool = False                             # 'cùng kỳ tháng trước'
    yoy: bool = False                                     # 'cùng kỳ năm trước'
    prev_period: bool = False                             # 'so với tháng/kỳ trước'
    tet: List[Tuple[int, str]] = field(default_factory=list)
    before_tet_days: Optional[int] = None
    window_days: Optional[int] = None                     # 'N ngày kể từ ngày D'
    window_start: Optional[D] = None
    explicit_dates: List[D] = field(default_factory=list)
    text_wo_time: str = ""


def parse_time(q: str, today: D, cal: Optional[LunarCalendar] = None) -> TimeInfo:
    """Extract periods from a normalised question."""
    ti = TimeInfo()
    s = " " + q + " "
    spans: List[Tuple[int, int, Period]] = []

    def add(m, p: Period):
        spans.append((m.start(), m.end(), p))

    # snapshot expressions
    m = re.search(r"(?:tinh |tinh toi |tinh den |toi |den |cuoi |vao cuoi |het |theo ton kho )?(?:het |cuoi )?ngay (\d{1,2})/(\d{1,2})/(\d{4})", s)
    snap_kw = re.search(r"(den het ngay|toi het ngay|tinh den het|tinh toi het|cuoi ngay|het ngay|den cuoi ngay|theo ton kho ngay|tai ngay|den ngay)\s+(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if snap_kw:
        ti.snapshot = _mk_date(snap_kw.group(2), snap_kw.group(3), snap_kw.group(4))
    elif re.search(r"cuoi ngay hom nay|het hom nay|cuoi ngay|het ngay hom nay|ton cuoi|hom nay kho|kho con|con ton|dang ton|hien (?:van )?con|dang con", s):
        ti.snapshot = today

    # Tết
    for mt in re.finditer(r"\btet(?: am lich| nguyen dan)?(?: nam)? ?(nay|ngoai|truoc|roi|\d{4})?", s):
        w = mt.group(1)
        if w is None or w == "nay":
            y = today.year
            # 'Tết năm nay' = Tết of the current solar year
            ti.tet.append((y, "this"))
        elif w in ("ngoai", "truoc", "roi"):
            ti.tet.append((today.year - 1, "last"))
        else:
            ti.tet.append((int(w), "year"))
    mb = re.search(r"(\d+) ngay (?:ngay )?truoc tet", s)
    if mb:
        ti.before_tet_days = int(mb.group(1))
    elif re.search(r"truoc tet", s):
        ti.before_tet_days = -1  # same length as the Tết window

    # price-effect windows: 'N ngay ke tu ngay D'
    mw = re.search(r"(\d+) ngay (?:dau|ke tu|tinh tu|sau)", s)
    if mw:
        ti.window_days = int(mw.group(1))

    # explicit ranges 'tu D1 den (het) D2'
    for mr in re.finditer(r"tu (?:ngay )?(\d{1,2})/(\d{1,2})/(\d{4}) (?:den|toi|-)(?: het)?(?: ngay)? (\d{1,2})/(\d{1,2})/(\d{4})", s):
        a = _mk_date(mr.group(1), mr.group(2), mr.group(3))
        b = _mk_date(mr.group(4), mr.group(5), mr.group(6))
        if a and b:
            add(mr, Period(a, b, f"{a}..{b}", "range"))
    # 'tu thang A/Y den thang B/Y' month ranges -> list of months
    mr = re.search(r"(?<![/\d])(?:tu )?(?:thang )?(\d{1,2})/(\d{4}) (?:den|toi|qua|-)(?: het)?(?: thang)? (\d{1,2})/(\d{4})\b", s)
    if mr and (int(mr.group(4)), int(mr.group(3))) <= (int(mr.group(2)), int(mr.group(1))):
        mr = None
    month_range = None
    if mr:
        y1, m1, y2, m2 = int(mr.group(2)), int(mr.group(1)), int(mr.group(4)), int(mr.group(3))
        month_range = (y1, m1, y2, m2)
        add(mr, Period(D(y1, m1, 1), month_end(y2, m2), "months", "month_range"))
    # 'N thang lien nhau ket thuc o thang M/Y'
    mk = re.search(r"(\d+) thang lien (?:nhau|tiep|ke) (?:ket thuc (?:o|vao|tai)|tinh (?:toi|den)|den|toi|cho toi)(?: het)? thang (\d{1,2})/(\d{4})", s)
    if mk:
        n, m2, y2 = int(mk.group(1)), int(mk.group(2)), int(mk.group(3))
        y1, m1 = add_months(y2, m2, -(n - 1))
        add(mk, Period(D(y1, m1, 1), month_end(y2, m2), "months", "month_range"))

    # month lists sharing a trailing year: 'thang 9 thang 10 thang 11/2025', 'thang 5 6 7/2025', 'thang 1, 2 va 3/2026'
    for ml in re.finditer(r"\bthang (\d{1,2})((?:\s+(?:va\s+)?(?:thang\s+)?\d{1,2}){1,5})\s*(?:/|\s+nam\s+)\s*(\d{4})\b", s):
        yr = int(ml.group(3))
        mons = [int(ml.group(1))] + [int(x) for x in re.findall(r"\d{1,2}", ml.group(2))]
        if all(1 <= m_ <= 12 for m_ in mons) and mons == sorted(mons):
            for m_ in mons:
                spans.append((ml.start(), ml.end(), month_period(yr, m_)))
    # single dates
    for md in _DATE_RE.finditer(s):
        d = _mk_date(md.group(1), md.group(2), md.group(3))
        if d:
            ti.explicit_dates.append(d)
    # months with year (also bare "M/YYYY" listed after a first 'thang')
    for mm in _MONTH_RE.finditer(s):
        mo, yr = int(mm.group(1)), int(mm.group(2))
        if 1 <= mo <= 12:
            add(mm, month_period(yr, mo))
    for mm in re.finditer(r"(?<![/\d])(\d{1,2})/(\d{4})\b", s):
        if s[max(0, mm.start() - 6):mm.start()].endswith("thang ") or s[max(0, mm.start() - 4):mm.start()].endswith("quy "):
            continue
        # skip parts of dd/mm/yyyy
        if re.match(r"\d{1,2}/\d{1,2}/\d{4}", s[max(0, mm.start() - 3):mm.end()]):
            continue
        if s[max(0, mm.start() - 1):mm.start()] == "/":
            continue
        mo, yr = int(mm.group(1)), int(mm.group(2))
        if 1 <= mo <= 12:
            add(mm, month_period(yr, mo))
    for mq in _QUARTER_RE.finditer(s):
        q_, y_ = int(mq.group(1)), int(mq.group(2))
        if 1 <= q_ <= 4:
            add(mq, quarter_period(y_, q_))
    for mq in re.finditer(r"\bquy (\d)\b(?!\s*/)", s):
        q_ = int(mq.group(1))
        if 1 <= q_ <= 4 and not any(a <= mq.start() < b for a, b, _ in spans):
            y_ = resolve_quarter_noyear(q_, today)
            add(mq, quarter_period(y_, q_))
    for mm in _MONTH_NOYEAR_RE.finditer(s):
        mo = int(mm.group(1))
        if not 1 <= mo <= 12 or any(a <= mm.start() < b for a, b, _ in spans):
            continue
        # "3 thang" (count) is not a month name ('thang 4 thang 5' is a list)
        if re.search(r"\d\s*$", s[:mm.start()]) and not re.search(r"\bthang \d{1,2}\s*$", s[:mm.start()]):
            continue
        y_, m_ = resolve_month_noyear(mo, today)
        add(mm, month_period(y_, m_))
    # 'thang M nam roi / ngoai / truoc' = month M of last year
    for mm in re.finditer(r"\bthang (\d{1,2}) (?:cua )?nam (?:roi|ngoai|truoc)\b", s):
        mo = int(mm.group(1))
        if 1 <= mo <= 12 and not any(a <= mm.start() < b for a, b, _ in spans):
            add(mm, month_period(today.year - 1, mo))
    # relative expressions
    rel = [
        (r"\btu (?:dau thang|mung 1|ngay 1|ngay dau thang)(?: nay)? (?:den|toi) (?:het )?(?:hom nay|bua nay|nay|gio)\b",
         lambda: Period(today.replace(day=1), today, "this_month", "mtd", month_end(today.year, today.month))),
        (r"\b(?:ngay )?hom nay\b|\bbua nay\b|\btrong ngay hom nay\b", lambda: Period(today, today, "today", "day")),
        (r"\bhom qua\b|\bbua qua\b|\bngay hom qua\b", lambda: Period(today - dt.timedelta(1), today - dt.timedelta(1), "yesterday", "day")),
        (r"\btuan nay\b", lambda: Period(week_monday(today), today, "this_week", "week")),
        (r"\btuan (?:truoc|roi|vua roi|vua qua|qua)\b", lambda: Period(week_monday(today) - dt.timedelta(7), week_monday(today) - dt.timedelta(1), "last_week", "week")),
        (r"\bthang nay\b", lambda: Period(today.replace(day=1), today, "this_month", "mtd", month_end(today.year, today.month))),
        (r"\b(?:tu )?dau nam(?: nay)?(?: (?:toi|den) (?:nay|het hom nay|hom nay))?\b|\btu (?:ngay )?1/1(?:/\d{4})? (?:den|toi) het (?:hom nay|bua nay)|\bnam nay\b",
         lambda: Period(D(today.year, 1, 1), today, "ytd", "ytd")),
    ]
    for rx, fn in rel:
        for mm in re.finditer(rx, s):
            if any(a <= mm.start() < b for a, b, _ in spans):
                continue
            add(mm, fn())
    for mm in re.finditer(r"\b(\d+) ngay gan (?:nhat|day)\b|\b(\d+) ngay qua\b|\b(\d+) ngay vua qua\b", s):
        n = int(mm.group(1) or mm.group(2) or mm.group(3))
        add(mm, Period(today - dt.timedelta(n - 1), today, f"last{n}d", "range"))
    for mm in re.finditer(r"\bthang (?:truoc|roi|vua roi|vua qua)\b", s):
        if re.search(r"(so voi|so|vs|hon|kem|cung ky|ngay dau|ngay cua|so ngay)\s*(?:cung ky\s*)?$", s[:mm.start()]):
            continue
        y_, m_ = prev_month(today.year, today.month)
        add(mm, month_period(y_, m_, "last_month"))
    for mm in re.finditer(r"\bnam (?:truoc|ngoai|roi)\b", s):
        if re.search(r"(cung ky|so voi|so|vs|cung thang)\s*$", s[:mm.start()]):
            continue
        add(mm, Period(D(today.year - 1, 1, 1), D(today.year - 1, 12, 31), "last_year", "year"))
    for mm in re.finditer(r"\bnam (\d{4})\b", s):
        if any(a <= mm.start() < b for a, b, _ in spans):
            continue
        if re.search(r"(thang \d{1,2}|quy \d|tet)\s*$", s[:mm.start()]):
            continue
        y_ = int(mm.group(1))
        add(mm, Period(D(y_, 1, 1), D(y_, 12, 31), f"year{y_}", "year"))
    # comparisons
    if re.search(r"cung (?:ky|so ngay)(?: dau)?(?: cua)? (?:thang|th) (?:truoc|roi|lien truoc)", s):
        ti.mtd_compare = True
    if re.search(r"cung ky nam (?:truoc|ngoai|roi)|cung thang (?:cua )?nam (?:truoc|ngoai|roi)|nam ngoai cung thang", s):
        ti.yoy = True
    if re.search(r"(so voi|so|vs|hon|kem|chenh)[^.]*?(thang|ky|quy|tuan) (lien )?(truoc|roi)\b|thang lien truoc|thang truoc do|ky truoc", s):
        ti.prev_period = True
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    # de-overlap (a month list contributes several periods with the same span)
    out, last_end, last_span = [], -1, None
    for a, b, p in spans:
        if a < last_end and (a, b) != last_span:
            continue
        out.append(p)
        last_end, last_span = max(last_end, b), (a, b)
    # 'tính luôn hôm nay, 7 ngày gần nhất' / 'tháng này tính đến hôm nay': 'hôm nay' only bounds a longer period
    bound = re.search(r"\b(?:toi|den|tinh den|tinh toi|tinh luon|tinh ca|ke ca|tinh) (?:het )?(?:ngay )?(?:hom nay|bua nay|bua ni|hom ni)\b", s)
    if len(out) > 1 and (bound or any(p.kind in ("range", "mtd", "ytd", "week") and p.end == today for p in out)):
        out = [p for p in out if not (p.kind == "day" and p.start == today)] or out
    ti.periods = out
    # text without time
    t = s
    for a, b, _ in sorted(spans, key=lambda x: -x[0]):
        t = t[:a] + " " + t[b:]
    ti.text_wo_time = re.sub(r"\s+", " ", t).strip()
    return ti


def tet_period(cal: LunarCalendar, year: int) -> Optional[Period]:
    w = cal.tet_window(year)
    if not w:
        return None
    return Period(w[0], w[1], f"tet{year}", "tet")
