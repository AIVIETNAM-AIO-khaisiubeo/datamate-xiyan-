"""Frame -> SQL over the canonical views ("silver" SQL for the SQL generators).

The training split has answers but no SQL.  For every train question whose rule frame reproduces the gold
answer, this compiler writes the equivalent DuckDB SQL over the views of ``mschema.build_views``; a query
is kept only if executing it gives the same answer (``scripts/build_sql_silver.py``).  The verified queries
feed the ICL few-shot bank of G4, the multi-task / multi-format SFT data and the selection-model data
(XiYan-SQL §III-C/D), and give the Schema Filter a measurable target (columns a correct query uses).

Three output formats per frame (XiYan "multi-format" SQL):
  base      plain SELECT, upper-case keywords, full table names
  cte       structural variant: the filtered rows in a CTE, aggregate outside
  alias     stylistic variant: lower-case keywords, short table aliases
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..nlu.frame import Frame
from ..nlu.timeparse import Period

FORMATS = ("base", "cte", "alias")
SHIFT_HOURS = {"sang": (6, 13), "chieu": (14, 21)}
LINE_FILTERS = ("product", "group", "ingredient", "rx", "size", "brand", "base_unit")
ORDER_FILTERS = ("shift", "pay", "channel", "cust_group", "staff")


def _q(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _between(col: str, p: Period) -> str:
    return f"{col} BETWEEN DATE {_q(p.start)} AND DATE {_q(p.end)}"


class _Ctx:
    def __init__(self, fmt: str):
        self.fmt = fmt
        self.o = "o" if fmt == "alias" else "orders"
        self.l = "l" if fmt == "alias" else "sales_lines"

    def kw(self, sql: str) -> str:
        if self.fmt != "alias":
            return sql
        return re.sub(r"\b(SELECT|FROM|WHERE|AND|OR|GROUP BY|ORDER BY|LIMIT|JOIN|ON|AS|SUM|COUNT|DISTINCT|ROUND|CASE|WHEN|"
                      r"THEN|ELSE|END|BETWEEN|DATE|DESC|ASC|IN|IS|NOT|NULL|WITH|LOWER|UPPER|MAX|STRFTIME|NULLIF)\b",
                      lambda m: m.group(1).lower(), sql)

    def tab(self, name: str) -> str:
        if self.fmt != "alias":
            return name
        return f"{name} {self.o if name == 'orders' else self.l}"


def _order_conds(c: _Ctx, f: dict, ex) -> Optional[List[str]]:
    o = c.o
    out = []
    if f.get("shift"):
        a, b = SHIFT_HOURS[f["shift"]]
        out.append(f"{o}.order_hour BETWEEN {a} AND {b}")
    if f.get("pay"):
        out.append(f"{o}.payment_method = {_q(f['pay'])}")
    if f.get("channel"):
        out.append(f"{o}.channel = {_q(f['channel'])}")
    if f.get("cust_group"):
        out.append(f"LOWER({o}.customer_group) = LOWER({_q(f['cust_group'])})")
    if f.get("staff"):
        out.append(f"UPPER({o}.staff) = UPPER({_q(f['staff'])})")
    return out


def _line_conds(c: _Ctx, f: dict, ex) -> Optional[List[str]]:
    l = c.l
    out = []
    if f.get("product"):
        keys = f["product"] if isinstance(f["product"], (list, set, tuple)) else [f["product"]]
        out.append(f"{l}.product_key IN ({', '.join(_q(k) for k in keys)})" if len(keys) > 1 else f"{l}.product_key = {_q(keys[0])}")
    if f.get("group"):
        out.append(f"LOWER({l}.product_group) = LOWER({_q(f['group'])})")
    for k in ("ingredient", "rx", "size", "brand", "base_unit"):
        if f.get(k) is not None:
            return None          # catalogue attributes not exposed on the line view
    return out


def compile_frame(fr: Frame, ex, fmt: str = "base") -> Optional[Tuple[str, str]]:
    """(sql, answer_type) or None when the intent / filters are outside the compiled subset."""
    if fr.intent in ("unanswerable", "unsupported") or not fr.periods:
        return None
    try:
        P = [ex.clip(p) for p in fr.periods]
    except Exception:
        return None
    c = _Ctx(fmt)
    f = dict(fr.filters or {})
    if any(k not in LINE_FILTERS + ORDER_FILTERS for k, v in f.items() if v is not None):
        return None
    oc = _order_conds(c, f, ex)
    lc = _line_conds(c, f, ex)
    if lc is None:
        return None
    it = fr.intent
    p = P[0]
    o, l = c.o, c.l
    has_orders = getattr(ex, "has_orders", False)

    def where(conds):
        return " AND ".join(conds)

    def lines_query(select: str, extra: List[str], tail: str = "") -> str:
        conds = [f"{l}.is_valid = 1", f"{l}.kind = 'item'", _between(f"{l}.order_date", p)] + lc + extra
        src = c.tab("sales_lines")
        if oc:
            src += f" JOIN {c.tab('orders')} ON {o}.order_id = {l}.order_id"
            conds += oc
        if fmt == "cte":
            inner = f"SELECT {l}.* FROM {src} WHERE {where(conds)}"
            sel = select.replace(f"{l}.", "")
            return f"WITH item_lines AS ({inner}) SELECT {sel} FROM item_lines{(' ' + tail.replace(l + '.', '')) if tail else ''}"
        return f"SELECT {select} FROM {src} WHERE {where(conds)}{(' ' + tail) if tail else ''}"

    def orders_query(select: str, extra: List[str], tail: str = "", valid: bool = True) -> str:
        conds = ([f"{o}.is_valid = 1"] if valid else []) + [_between(f"{o}.order_date", p)] + oc + extra
        if fmt == "cte":
            inner = f"SELECT * FROM {c.tab('orders')} WHERE {where(conds)}"
            sel = select.replace(f"{o}.", "")
            return f"WITH sel_orders AS ({inner}) SELECT {sel} FROM sel_orders{(' ' + tail.replace(o + '.', '')) if tail else ''}"
        return f"SELECT {select} FROM {c.tab('orders')} WHERE {where(conds)}{(' ' + tail) if tail else ''}"

    sql, at = None, None
    if it == "revenue":
        if lc:
            sql, at = lines_query(f"SUM({l}.amount)", []), "number"
        elif has_orders:
            sql, at = orders_query(f"SUM({o}.revenue)", []), "number"
    elif it == "order_count" and has_orders:
        if lc:
            conds = [f"{o}.is_valid = 1", _between(f"{o}.order_date", p)] + oc + \
                    [f"{o}.order_id IN (SELECT {l}.order_id FROM {c.tab('sales_lines')} WHERE {l}.is_valid = 1 AND {l}.kind = 'item' AND {where(lc)})"]
            sql, at = f"SELECT COUNT(*) FROM {c.tab('orders')} WHERE {where(conds)}", "number"
        else:
            sql, at = orders_query("COUNT(*)", []), "number"
    elif it == "cancel_count" and has_orders and not lc:
        sql, at = orders_query("COUNT(*)", [f"{o}.status = 'cancelled'"], valid=False), "number"
    elif it == "cancel_rate" and has_orders and not lc:
        sql = orders_query(f"ROUND(100.0 * SUM(CASE WHEN {o}.status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*), 2)", [], valid=False)
        at = "percent"
    elif it == "aov" and has_orders and not lc:
        sql, at = orders_query(f"ROUND(SUM({o}.revenue) / COUNT(*))", []), "number"
    elif it == "units" and lc and not fr.unit:
        sql, at = lines_query(f"SUM({l}.base_qty)", []), "number"
    elif it == "top_products" and fr.by in ("qty", "revenue"):
        k = fr.k or 1
        m = f"SUM({l}.base_qty)" if fr.by == "qty" else f"SUM({l}.amount)"
        sql = lines_query(f"{l}.product_name", [], f"GROUP BY {l}.product_key, {l}.product_name ORDER BY {m} DESC LIMIT {k}")
        at = "string" if (fr.answer_type == "string" or k == 1) else "list"
    elif it == "top_customers" and has_orders and fr.by in ("revenue", "orders") and not lc:
        k = fr.k or 1
        m = f"SUM({o}.revenue)" if fr.by == "revenue" else "COUNT(*)"
        sql = orders_query(f"MAX({o}.customer_name)", [f"{o}.customer_key IS NOT NULL"],
                           f"GROUP BY {o}.customer_key ORDER BY {m} DESC LIMIT {k}")
        at = "string" if k == 1 else "list"
    elif it == "best_day" and has_orders and not lc and fr.by in (None, "revenue"):
        sql, at = orders_query(f"{o}.order_date", [], f"GROUP BY {o}.order_date ORDER BY SUM({o}.revenue) DESC LIMIT 1"), "date"
    elif it == "share" and has_orders and lc and not oc:
        part = lines_query(f"SUM({l}.amount)", [])
        tot = f"SELECT SUM(revenue) FROM orders WHERE is_valid = 1 AND {_between('order_date', p)}"
        sql, at = f"SELECT ROUND(100.0 * ({part}) / NULLIF(({tot}), 0), 2)", "percent"
    elif it == "purchase_total" and not f:
        if fmt == "cte":
            sql = f"WITH r AS (SELECT * FROM purchases WHERE cancelled = 0 AND {_between('receipt_date', p)}) SELECT SUM(total) FROM r"
        else:
            sql = f"SELECT SUM(total) FROM purchases WHERE cancelled = 0 AND {_between('receipt_date', p)}"
        at = "number"
    elif it == "monthly_list" and has_orders and not f and len(P) >= 2:
        parts = [f"(SELECT SUM(revenue) FROM orders WHERE is_valid = 1 AND {_between('order_date', q)})" for q in P]
        sql, at = "SELECT " + ", ".join(parts), "list"
    if sql is None:
        return None
    return c.kw(sql), at


def tables_columns(sql: str, views) -> List[str]:
    """'table.column' names an SQL text refers to (lexical; good enough for schema-filter recall)."""
    low = sql.lower()
    out = []
    for t, df in views.items():
        if not re.search(r"\b" + t + r"\b", low):
            continue
        for col in df.columns:
            if re.search(r"\b" + re.escape(col.lower()) + r"\b", low):
                out.append(f"{t}.{col}")
    return out
