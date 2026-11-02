"""Controlled modifications of a correct SQL ("near-miss" negatives, XiYan-SQL §III-D / Appendix II.C).

Each mutation introduces one subtle, realistic error that a generator could make on this task; the result
is re-executed and kept as a negative only if its answer differs from the correct one.  Used for
self-refine SFT data, the selection model's listwise data, and to measure the SQL checks
(``scripts/eval_sql_checks.py``).
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Callable, Dict, List, Optional


def drop_is_valid(sql: str) -> Optional[str]:
    out = re.sub(r"(?i)\b(?:\w+\.)?is_valid\s*=\s*1\s+and\s+", "", sql)
    out = re.sub(r"(?i)\s+and\s+(?:\w+\.)?is_valid\s*=\s*1\b", "", out)
    out = re.sub(r"(?i)where\s+(?:\w+\.)?is_valid\s*=\s*1\b", "WHERE 1 = 1", out)
    return out if out != sql else None


def shift_month(sql: str) -> Optional[str]:
    """Period off by one month (a classic relative-time error)."""
    lits = re.findall(r"'(\d{4})-(\d{2})-(\d{2})'", sql)
    if not lits:
        return None

    def mv(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        mo -= 1
        if mo == 0:
            y, mo = y - 1, 12
        last = (dt.date(y + (mo == 12), mo % 12 + 1, 1) - dt.timedelta(1)).day
        return f"'{y:04d}-{mo:02d}-{min(d, last):02d}'"
    return re.sub(r"'(\d{4})-(\d{2})-(\d{2})'", mv, sql)


def end_exclusive(sql: str) -> Optional[str]:
    """Inclusive upper bound written as exclusive ('<' instead of '<=' / BETWEEN)."""
    m = re.search(r"(?i)(\S+)\s+between\s+date\s+'([\d-]+)'\s+and\s+date\s+'([\d-]+)'", sql)
    if not m:
        return None
    return sql[:m.start()] + f"{m.group(1)} >= DATE '{m.group(2)}' AND {m.group(1)} < DATE '{m.group(3)}'" + sql[m.end():]


def paid_instead_of_revenue(sql: str) -> Optional[str]:
    out = re.sub(r"(?i)\b((?:\w+\.)?)revenue\b", r"\1total_due", sql)
    return out if out != sql else None


def drop_kind(sql: str) -> Optional[str]:
    out = re.sub(r"(?i)\s+and\s+(?:\w+\.)?kind\s*=\s*'item'", "", sql)
    return out if out != sql else None


def limit_off(sql: str) -> Optional[str]:
    m = re.search(r"(?i)\blimit\s+(\d+)", sql)
    if not m:
        return None
    k = int(m.group(1))
    return sql[:m.start(1)] + str(k + 1 if k == 1 else k - 1) + sql[m.end(1):]


def fanout_join(sql: str) -> Optional[str]:
    """Sum an order-level measure after joining the order lines (duplicates orders)."""
    m = re.match(r"(?is)^select\s+(sum|count)\s*\(\s*(\*|(?:orders\.)?revenue)\s*\)\s+from\s+orders\s+where\s+(.+)$", sql.strip())
    if not m:
        return None
    agg, col, cond = m.group(1), m.group(2), m.group(3)
    cond = re.sub(r"(?<![.\w])(is_valid|order_date|status|channel|payment_method|order_hour|customer_group|staff|customer_key)\b",
                  r"orders.\1", cond)
    target = "*" if col == "*" else "orders.revenue"
    return f"SELECT {agg.upper()}({target}) FROM orders JOIN sales_lines ON sales_lines.order_id = orders.order_id WHERE {cond}"


MUTATIONS: Dict[str, Callable[[str], Optional[str]]] = {
    "drop_is_valid": drop_is_valid, "shift_month": shift_month, "end_exclusive": end_exclusive,
    "paid_instead_of_revenue": paid_instead_of_revenue, "drop_kind": drop_kind, "limit_off": limit_off,
    "fanout_join": fanout_join,
}


def mutate_all(sql: str) -> List[tuple]:
    out = []
    for name, fn in MUTATIONS.items():
        try:
            m = fn(sql)
        except Exception:  # noqa: BLE001
            m = None
        if m and m != sql:
            out.append((name, m))
    return out
