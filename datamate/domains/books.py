"""Hand-kept credit books stored beside the sales book, one sheet each.

* debt book ('Sổ nợ'): first row of an order = purchase date, customer, amount still owed after paying at
  purchase; repayments ('Đã trả', 'Ngày trả') on that row and on following rows of the same order.
* instalment book ('Sổ góp' / 'Trả góp'): first row of a contract = purchase date, customer, price, down
  payment, number of instalments, amount per instalment, due day of month; each instalment paid is a row.
  Instalment i is due on that day of the i-th month after the purchase month (month end if shorter);
  the last instalment takes the rounding remainder.

Recognised from their headers (never from a business id).  Both are attached to the Biz lazily and
removed from the 'unmodelled tables' list once understood.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..textnorm import fold, fold_words
from ..values import is_blank, parse_datetime, parse_money


def _d(v) -> Optional[dt.date]:
    if is_blank(v):
        return None
    x = parse_datetime(v)
    if x is None:
        return None
    return x.date() if isinstance(x, dt.datetime) else x


def _m(v) -> float:
    if is_blank(v):
        return 0.0
    x = parse_money(v)
    return float(x) if x is not None else 0.0


def _col(cols: Dict[str, str], *pats) -> Optional[str]:
    for fc, c in cols.items():
        if any(re.search(p, fc) for p in pats):
            return c
    return None


@dataclass
class Contract:
    date: dt.date
    key: str
    customer: str
    price: float
    prepaid: float
    n: int
    per: float
    due_day: int
    pays: List[Tuple[dt.date, float]] = field(default_factory=list)

    def schedule(self) -> List[Tuple[dt.date, float]]:
        out, rest = [], self.price - self.prepaid
        y, m = self.date.year, self.date.month
        for i in range(1, self.n + 1):
            m += 1
            if m > 12:
                y, m = y + 1, 1
            last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)).day
            amt = self.per if i < self.n else rest - self.per * (self.n - 1)
            out.append((dt.date(y, m, min(self.due_day, last)), amt))
        return out

    def paid(self, x: dt.date) -> float:
        return sum(a for d, a in self.pays if d is not None and d <= x)

    def outstanding(self, x: dt.date) -> float:
        return max(0.0, self.price - self.prepaid - self.paid(x))

    def due_count(self, x: dt.date, inclusive: bool = True) -> int:
        return sum(1 for d, _ in self.schedule() if (d <= x if inclusive else d < x))

    def overdue(self, x: dt.date) -> float:
        due = sum(a for d, a in self.schedule() if d < x)
        return max(0.0, due - self.paid(x))


@dataclass
class DebtOrder:
    date: dt.date
    key: str
    customer: str
    owed: float
    pays: List[Tuple[dt.date, float]] = field(default_factory=list)

    def remaining(self, x: dt.date) -> float:
        return self.owed - sum(a for d, a in self.pays if d is not None and d <= x)


def attach(b) -> None:
    if getattr(b, "_books_done", False):
        return
    b._books_done = True
    b._instal, b._debtbook = None, None
    keep = []
    for t in getattr(b, "other", []) or []:
        cols = {fold(str(c)): c for c in t.df.columns}
        kn = " ".join(cols)
        if re.search(r"so ky", kn) and re.search(r"gop", kn):
            b._instal = _parse_instal(t.df, cols)
            if b._instal:
                continue
        if re.search(r"\bno\b", kn) and re.search(r"da tra", kn) and re.search(r"ngay tra", kn):
            b._debtbook = _parse_debts(t.df, cols)
            if b._debtbook:
                continue
        keep.append(t)
    b.other = keep


def _parse_instal(df, cols) -> List[Contract]:
    c_date, c_stt, c_cus = _col(cols, r"^ngay$", r"ngay mua"), _col(cols, r"^stt"), _col(cols, r"^khach")
    c_price, c_pre = _col(cols, r"^gia$", r"gia ban", r"tong"), _col(cols, r"tra truoc")
    c_n, c_per, c_day = _col(cols, r"so ky"), _col(cols, r"gop/thang|gop thang|moi ky"), _col(cols, r"han gop")
    c_pd, c_pa = _col(cols, r"ngay gop"), _col(cols, r"da gop")
    if not all([c_date, c_cus, c_price, c_n, c_per, c_day, c_pd, c_pa]):
        return []
    out: List[Contract] = []
    by_key: Dict[str, Contract] = {}
    cur = None
    for _, r in df.iterrows():
        key = None if c_stt is None or is_blank(r[c_stt]) else str(r[c_stt]).strip()
        if not is_blank(r[c_cus]) and _d(r[c_date]):
            try:
                cur = Contract(_d(r[c_date]), key or f"row{len(out)}", str(r[c_cus]).strip(), _m(r[c_price]),
                               _m(r[c_pre]) if c_pre else 0.0, int(float(r[c_n])), _m(r[c_per]), int(float(r[c_day])))
            except (TypeError, ValueError):
                cur = None
                continue
            out.append(cur)
            by_key[cur.key] = cur
        tgt = by_key.get(key) if key else cur
        if tgt is not None and not is_blank(r[c_pa]):
            tgt.pays.append((_d(r[c_pd]), _m(r[c_pa])))
    return out


def _parse_debts(df, cols) -> List[DebtOrder]:
    c_date, c_cus, c_owed = _col(cols, r"^ngay$", r"ngay mua"), _col(cols, r"^khach"), _col(cols, r"^no$")
    c_pa, c_pd, c_note = _col(cols, r"da tra"), _col(cols, r"ngay tra"), _col(cols, r"ghi chu|stt")
    if not all([c_date, c_cus, c_owed, c_pa, c_pd]):
        return []
    out: List[DebtOrder] = []
    by_key: Dict[str, DebtOrder] = {}
    cur = None
    for _, r in df.iterrows():
        key = None if c_note is None or is_blank(r[c_note]) else str(r[c_note]).strip()
        if not is_blank(r[c_cus]) and _d(r[c_date]):
            cur = DebtOrder(_d(r[c_date]), key or f"row{len(out)}", str(r[c_cus]).strip(), _m(r[c_owed]))
            out.append(cur)
            if key:
                by_key[key] = cur
        tgt = by_key.get(key) if key else cur
        if tgt is not None and not is_blank(r[c_pa]):
            tgt.pays.append((_d(r[c_pd]), _m(r[c_pa])))
    return out


# ---------------------------------------------------------------------------- parsing
def _customer(contracts, s: str) -> Optional[str]:
    best = None
    for c in contracts:
        fn = fold_words(c.customer)
        if (" " + fn + " ") in (" " + s + " ") and (best is None or len(fn) > len(fold_words(best))):
            best = c.customer
    return best


def parse(rp, s: str, q: str, ti, today: dt.date):
    from ..nlu.frame import Frame
    b = rp.b
    attach(b)
    if not b._instal or not re.search(r"\bgop\b|tra gop|hop dong", s):
        return None
    snap = ti.snapshot or (ti.explicit_dates[-1] if ti.explicit_dates else today)
    cus = _customer(b._instal, s)
    f = {"customer": cus} if cus else {}
    if re.search(r"(?:may|bao nhieu|bn|so) ky", s) and re.search(r"toi han|den han|qua han", s):
        mode = "overdue_n" if re.search(r"qua han", s) and not re.search(r"toi han|den han", s) else "due_n"
        return Frame("instal", snapshot=snap, filters=f, params={"what": mode}, answer_type="number")
    if re.search(r"(?:may|bao nhieu|bn|so) (?:khach|nguoi)", s) and re.search(r"qua han|cham|tre", s):
        return Frame("instal", snapshot=snap, filters=f, params={"what": "overdue_customers"}, answer_type="number")
    if re.search(r"qua han|cham gop|tre han|cham dong|tre gop", s) and re.search(r"tien|bao nhieu|bn|tong", s) and \
            not re.search(r"ke ca|tinh ca|chua toi han", s):
        return Frame("instal", snapshot=snap, filters=f, params={"what": "overdue_amount"}, answer_type="number")
    if re.search(r"con lai|chua gop|chua tra|con thieu|con no|chua dong", s):
        return Frame("instal", snapshot=snap, filters=f, params={"what": "outstanding"}, answer_type="number")
    return Frame("unsupported", reason="câu hỏi sổ góp chưa được mô hình hoá")


def run_instal(fe, fr):
    from ..engine.answer import Answer
    from ..engine.executor import money
    b = fe.ex.b
    x = fr.snapshot or fe.ex.today
    cs = [c for c in b._instal if c.date <= x]
    if fr.filters.get("customer"):
        cs = [c for c in cs if c.customer == fr.filters["customer"]]
        if not cs:
            return Answer("abstain", reason="khách không có hợp đồng góp")
    w = fr.params["what"]
    if w == "due_n":
        return Answer("number", sum(c.due_count(x, True) for c in cs))
    if w == "overdue_n":
        return Answer("number", sum(c.due_count(x, False) for c in cs))
    if w == "overdue_customers":
        return Answer("number", len({fold_words(c.customer) for c in cs if c.overdue(x) > 0.5}))
    if w == "overdue_amount":
        return Answer("number", money(sum(c.overdue(x) for c in cs)))
    if w == "outstanding":
        return Answer("number", money(sum(c.outstanding(x) for c in cs)))
    raise KeyError(w)


def run_debt(fe, fr):
    """Debt intents answered from the debt book (orders still owed after purchase)."""
    from ..engine.answer import Answer
    from ..engine.executor import money
    b = fe.ex.b
    x = fr.snapshot or fe.ex.today
    rows = [(o, o.remaining(x)) for o in b._debtbook if o.date <= x]
    rows = [(o, r) for o, r in rows if r > 0.5]
    it = fr.intent
    if it == "debt_total":
        return Answer("number", money(sum(r for _, r in rows)))
    if it == "debtor_count":
        return Answer("number", len({fold_words(o.customer) for o, _ in rows}))
    if it == "debt_aging":
        out = []
        for lo, hi in fr.params["buckets"]:
            out.append(money(sum(r for o, r in rows if (x - o.date).days >= lo and (hi is None or (x - o.date).days <= hi))))
        return Answer("list", out)
    if it == "debt_overdue":
        n = fr.params["days"]
        return Answer("number", money(sum(r for o, r in rows if (x - o.date).days > n)))
    if it == "top_debtors":
        agg: Dict[str, float] = {}
        for o, r in rows:
            agg[o.customer] = agg.get(o.customer, 0.0) + r
        items = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
        k = fr.k or 3
        if len(items) > k and abs(items[k - 1][1] - items[k][1]) < 0.5:
            return Answer("abstain", reason="tie at boundary")
        return Answer("table", {"columns": ["khach_hang", "so_tien_no"], "rows": [[n_, money(v)] for n_, v in items[:k]]})
    return None
