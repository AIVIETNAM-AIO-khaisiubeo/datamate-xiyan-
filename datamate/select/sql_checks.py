"""Content checks of a candidate SQL against the question, the schema metadata and the rulebook semantics.

The selector of XiYan-SQL reads the SQL (selection model); here a deterministic critic does the part that
can be checked exactly, before any model is asked.  Each check returns a flag with a severity in (0, 1];
the selector lowers the candidate's reliability accordingly (severity >= 0.8: practically rejected).

  missing_is_valid     revenue / order-count style question, SQL on orders / sales_lines never filters is_valid
  valid_on_cancel      cancellation question filtered on is_valid = 1 (cancelled orders are not valid)
  no_date_filter       the question names a period, the SQL never uses the table's date column
  period_mismatch      the SQL's date literals match none of the parsed period's bounds
  after_today          the SQL reads dates after as_of_date
  join_fanout          orders joined with a line-level table while summing / counting an order-level column
  paid_not_revenue     'doanh thu / bán được bao nhiêu tiền' answered from payments or total_due
  missing_kind_item    product ranking on sales_lines without kind = 'item' (service fee lines are not items)
  limit_mismatch       top-k question with a different LIMIT
  shape                scalar answer expected, query returns several rows (first row would be taken)
"""
from __future__ import annotations

import datetime as dt
import re
from typing import List, Optional, Sequence, Tuple

LINE_TABLES = ("sales_lines", "payments", "deliveries")
DATE_COL = {"orders": "order_date", "sales_lines": "order_date", "payments": "pay_date", "purchases": "receipt_date",
            "purchase_lines": "receipt_date", "expenses": "expense_date", "deliveries": "delivery_date"}

_REV = re.compile(r"doanh thu|tien ban|ban duoc .*tien|thu ve|bao nhieu tien|tien hang|doanh so")
_COUNT = re.compile(r"(?:so|bao nhieu|may|dem) (?:don|hoa don|bill)")
_CANCEL = re.compile(r"\bhuy\b|tra hang|tra lai")
_PAID = re.compile(r"khach phai tra|tra no|thu no|da tra|thanh toan")
_RANK = re.compile(r"\b(?:mat hang|san pham|mon|hang|sp)\b.*\b(?:nao|gi|chi|mo)\b|ban chay|\btop\s*\d|\bliet ke\b")


def _tables(sql: str) -> List[str]:
    low = sql.lower()
    return [t for t in DATE_COL if re.search(r"\b" + t + r"\b", low)]


def _strip_strings(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def lint(sql: str, nq: str, periods: Sequence = (), today: Optional[dt.date] = None, answer_type: str = "number",
         k: Optional[int] = None, n_rows: Optional[int] = None) -> List[Tuple[str, float]]:
    s = " " + nq + " "
    low = sql.lower()
    code = _strip_strings(low)
    tabs = _tables(low)
    flags: List[Tuple[str, float]] = []
    cancel_q = bool(_CANCEL.search(s))
    rev_q = bool(_REV.search(s) or _COUNT.search(s)) and not cancel_q
    if rev_q and any(t in tabs for t in ("orders", "sales_lines")) and "is_valid" not in code and "status" not in code:
        flags.append(("missing_is_valid", 0.9))
    if cancel_q and re.search(r"is_valid\s*=\s*(?:1|true)", code) and "cancel" in low:
        flags.append(("valid_on_cancel", 0.9))
    if periods:
        if tabs and not any(DATE_COL[t] in code for t in tabs):
            flags.append(("no_date_filter", 0.8))
        lits = [dt.date.fromisoformat(x) for x in re.findall(r"'(\d{4}-\d{2}-\d{2})'", low)]
        if lits:
            starts, ends = set(), set()
            for p in periods:
                starts |= {p.start, p.start - dt.timedelta(1)}                    # '>= start' / '> start-1'
                e = {p.end, getattr(p, "full_end", None) or p.end}
                if today:
                    e.add(min(p.end, today))
                ends |= e | {d + dt.timedelta(1) for d in e}                    # '<= end' / '< end+1'
            if min(lits) not in starts or max(lits) not in ends:
                flags.append(("period_mismatch", 0.8))
            # strict '<' on the inclusive last day drops that day ('đến hết ngày X' includes X)
            for p in periods:
                last = {p.end} | ({min(p.end, today)} if today else set())
                for d in last:
                    if re.search(r"<\s*(?:date\s*)?'" + d.isoformat() + "'", low):
                        flags.append(("end_exclusive", 0.8))
                        break
            if today and any(d > today + dt.timedelta(1) for d in lits):
                flags.append(("after_today", 0.5))
    if "orders" in tabs and any(t in tabs for t in LINE_TABLES) and re.search(r"\bjoin\b", code):
        agg_order = re.search(r"sum\s*\(\s*(?:\w+\.)?(?:revenue|total_due)\s*\)", code) or \
            re.search(r"count\s*\(\s*\*\s*\)", code) or re.search(r"count\s*\(\s*(?:\w+\.)?order_id\s*\)", code)
        if agg_order and "distinct" not in code:
            flags.append(("join_fanout", 0.9))
    if rev_q and not _PAID.search(s) and (re.search(r"\bpayments\b", code) or re.search(r"\btotal_due\b", code)):
        flags.append(("paid_not_revenue", 0.9))
    if _RANK.search(s) and "sales_lines" in tabs and "kind" not in code and answer_type in ("string", "list"):
        flags.append(("missing_kind_item", 0.5))
    if k:
        m = re.search(r"\blimit\s+(\d+)", code)
        if m and int(m.group(1)) != k:
            flags.append(("limit_mismatch", 0.8))
    if n_rows is not None and n_rows > 1 and answer_type in ("number", "percent", "date", "boolean", "string"):
        if answer_type != "string" or not re.search(r"\blimit\s+1\b", code):
            flags.append(("shape", 0.7))
    return flags


def penalty_factor(flags: Sequence[Tuple[str, float]]) -> float:
    """Multiplier on a candidate's reliability (1.0 = no flag)."""
    f = 1.0
    for _, sev in flags:
        f *= (1.0 - 0.9 * sev)
    return f
