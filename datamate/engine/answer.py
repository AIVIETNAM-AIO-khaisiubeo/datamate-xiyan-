"""Frame -> typed answer, using the rulebook-faithful Executor."""
from __future__ import annotations

import calendar
import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..nlu.frame import Frame
from ..nlu.timeparse import LunarCalendar, Period, add_months, month_end, month_period, prev_month, tet_period
from ..textnorm import fold_key
from .executor import Executor, NotSupported, Unanswerable, money, pct_change, r2, _num


@dataclass
class Answer:
    type: str
    value: Any = None
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_submission(self, qid: str) -> dict:
        if self.type == "abstain":
            return {"id": qid, "type": "abstain"}
        if self.type == "unanswerable":
            d = {"id": qid, "type": "unanswerable"}
            if self.reason:
                d["reason"] = str(self.reason)
            return d
        return {"id": qid, "type": self.type, "answer": self.value}

    def key(self) -> str:
        """Canonical key for consistency voting."""
        import json
        return f"{self.type}:{json.dumps(self.value, ensure_ascii=False, sort_keys=True, default=str)}"


def _prev_period(p: Period) -> Period:
    if p.kind in ("month",):
        y, m = prev_month(p.start.year, p.start.month)
        return month_period(y, m)
    if p.kind == "quarter":
        y, m = add_months(p.start.year, p.start.month, -3)
        return Period(dt.date(y, m, 1), month_end(*add_months(y, m, 2)), "prevq", "quarter")
    if p.kind == "week":
        return Period(p.start - dt.timedelta(7), p.start - dt.timedelta(1), "prevw", "week")
    if p.kind == "day":
        return Period(p.start - dt.timedelta(1), p.start - dt.timedelta(1), "prevd", "day")
    n = p.days()
    return Period(p.start - dt.timedelta(n), p.start - dt.timedelta(1), "prev", "range")


def _mtd_prev(p: Period) -> Period:
    """Same number of days of the previous month (rulebook 'Tháng này' row)."""
    y, m = prev_month(p.start.year, p.start.month)
    last = calendar.monthrange(y, m)[1]
    day = min(p.end.day, last)
    return Period(dt.date(y, m, 1), dt.date(y, m, day), "mtd_prev", "mtd")


def _yoy(p: Period) -> Period:
    def sh(d):
        try:
            return d.replace(year=d.year - 1)
        except ValueError:
            return d.replace(year=d.year - 1, day=28)
    if p.kind == "month":
        return month_period(p.start.year - 1, p.start.month)
    return Period(sh(p.start), sh(p.end), "yoy", p.kind)


class FrameExecutor:
    def __init__(self, ex: Executor, cal: Optional[LunarCalendar] = None):
        self.ex = ex
        self.cal = cal

    def run(self, fr: Frame) -> Answer:
        try:
            return self._run(fr)
        except Unanswerable as e:
            msg = str(e)
            # 'the data does not record X' is only safe when every table of the shop was understood
            unmodelled = [t for t in (getattr(self.ex.b, "other", None) or []) if len(t.df) > 5]
            if unmodelled and "ngoài khoảng dữ liệu" not in msg:
                return Answer("abstain", reason=f"possibly in an unmodelled table: {msg}")
            return Answer("unanswerable", reason=msg, meta={"unans": True})
        except (NotSupported, KeyError, ValueError, TypeError, ZeroDivisionError, IndexError, AttributeError) as e:
            return Answer("abstain", reason=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    def _run(self, fr: Frame) -> Answer:
        ex, it = self.ex, fr.intent
        P = fr.periods[0] if fr.periods else None
        f = fr.filters
        from ..domains import books
        books.attach(ex.b)
        if it == "instal":
            return books.run_instal(self, fr)
        if it in ("debt_total", "debtor_count", "debt_aging", "debt_overdue", "top_debtors") and getattr(ex.b, "_debtbook", None):
            a = books.run_debt(self, fr)
            if a is not None:
                return a
        if it.startswith("phone_"):
            from ..domains import phone
            return phone.run(self, fr)
        if it.startswith("repair_"):
            from ..domains import repair
            return repair.run(self, fr)
        if it == "unanswerable":
            return Answer("unanswerable", reason=fr.reason)
        if it == "unsupported":
            return Answer("abstain", reason=fr.reason)
        if it == "revenue":
            return Answer("number", ex.revenue(P, f))
        if it == "revenue_multi":
            return Answer("list", [ex.revenue(P, dict(f, **part)) for part in fr.params["parts"]])
        if it == "order_count":
            return Answer("number", ex.order_count(P, f))
        if it == "cancel_count":
            return Answer("number", ex.order_count(P, f, status="cancelled"))
        if it == "cancel_rate":
            return Answer("percent", ex.cancel_rate(P, f))
        if it == "aov":
            return Answer("number", ex.aov(P, f))
        if it == "units":
            return Answer("number", ex.units(P, f, fr.unit))
        if it in ("delivery_orders_min_trips", "delivered_qty"):
            D = ex.b.deliveries
            if D.empty:
                raise Unanswerable("không có sổ giao hàng")
            Pc = ex.clip(P)
            O = ex.orders(Pc, None)
            O = O[O["status_final"] != "cancelled"]
            oids = set(O["order_id"])
            Dd = D[D["order_id"].isin(oids) & D["date"].map(lambda d: d is not None and d <= ex.today)]
            if it == "delivery_orders_min_trips":
                c = Dd.groupby("order_id")["trip_no"].nunique()
                return Answer("number", int((c >= fr.params["n"]).sum()))
            k = f["product"]
            col = "base_qty" if "base_qty" in Dd.columns else "qty_base"
            q = Dd.loc[Dd["pkey"] == k, col].sum()
            if fr.unit:
                fac = ex.unit_factor(k, fr.unit)
                if fac and fac != 1:
                    q = q / fac
            return Answer("number", _num(q))
        if it == "credit_due":
            T = ex.b.debt_terms
            if T.empty or "due_date" not in T.columns:
                raise Unanswerable("không có sổ hạn nợ")
            O = ex.O
            od = dict(zip(O["order_id"], O["date"]))
            st = dict(zip(O["order_id"], O["status_final"]))
            rows = [(o, d) for o, d in zip(T["order_id"], T["due_date"])
                    if d is not None and od.get(o) is not None and od[o] <= fr.snapshot and st.get(o) != "cancelled"]
            if fr.params["mode"] == "after_today":
                return Answer("number", sum(1 for _, d in rows if d > fr.snapshot))
            return Answer("number", sum(1 for _, d in rows if P.start <= d <= P.end))
        if it == "top_brands":
            Pb = ex.b.products
            if Pb.empty or "brand" not in Pb.columns or Pb["brand"].isna().all():
                raise Unanswerable("danh mục không ghi hãng")
            bm = dict(zip(Pb["key"], Pb["brand"]))
            L = ex.lines(ex.clip(P), f)
            s = L.groupby(L["pkey"].map(bm))["base_qty"].sum()
            s = s[s.index.notna()]
            if s.empty:
                raise Unanswerable("không có hàng bán")
            top = s.sort_values(ascending=False)
            if len(top) > 1 and abs(top.iloc[0] - top.iloc[1]) < 1e-9:
                return Answer("abstain", reason="tie")
            return Answer("string", str(top.index[0]))
        if it == "units_share":
            Pc = ex.clip(P)
            base = {"group": fr.params["base_group"]}
            tot = ex.lines(Pc, base)["base_qty"].sum()
            if tot == 0:
                raise Unanswerable("không có hàng của nhóm trong kỳ")
            part = ex.lines(Pc, dict(base, **f))["base_qty"].sum()
            return Answer("percent", r2(part / tot * 100))
        if it == "share":
            Pc = ex.clip(P)
            part = ex.revenue(Pc, f)
            tot = ex.revenue(Pc, {})
            if tot == 0:
                raise Unanswerable("tổng doanh thu bằng 0")
            return Answer("percent", r2(part / tot * 100))
        if it == "growth":
            return Answer("percent", self._growth(fr))
        if it in ("top_products", "top_customers", "top_staff", "top_suppliers"):
            return self._rank(fr)
        if it == "best_day":
            return Answer("date", ex.best_day(P, fr.by or "revenue"))
        if it == "best_month":
            vals = []
            for p in fr.periods:
                vals.append((ex.revenue(p, {}), p))
            mx = max(v for v, _ in vals)
            p = [p for v, p in vals if v == mx][0]
            return Answer("date", f"{p.start.year:04d}-{p.start.month:02d}")
        if it == "monthly_list":
            return Answer("list", [ex.revenue(p, f) for p in fr.periods])
        if it == "increasing":
            v = [ex.revenue(p, f) for p in fr.periods]
            return Answer("boolean", all(v[i] < v[i + 1] for i in range(len(v) - 1)))
        if it == "price_on":
            return Answer("number", ex.price_on(f["product"], fr.params["date"]))
        if it == "price_effect":
            return Answer("percent", self._price_effect(fr))
        if it == "purchase_total":
            return Answer("number", ex.purchase_total(ex.clip_purchase(P) if hasattr(ex, "clip_purchase") else P))
        if it == "avg_purchase_price":
            return Answer("number", ex.avg_purchase_price(f["product"], P, fr.unit))
        if it == "purchase_minus_sales":
            return Answer("number", self._purchase_minus_sales(fr))
        if it == "purchased_qty":
            key = f["product"]
            PL = ex.purchase_lines(P)
            v = PL.loc[PL["pkey"] == key, "base_qty"].sum()
            if fr.unit:
                fac = ex.unit_factor(key, fr.unit)
                if fac and fac != 1:
                    v = v / fac
            return Answer("number", _num(v))
        if it == "stock":
            k = f["product"]
            v = ex.stock_at(k, fr.snapshot)
            if fr.unit:
                fac = ex.unit_factor(k, fr.unit)
                if fac and fac != 1:
                    v = v / fac
            return Answer("number", _num(v))
        if it == "stock_split":
            k = f["product"]
            v = ex.stock_at(k, fr.snapshot)
            fac = None
            for u in (fr.params.get("unit_cands") or []) + ([fr.unit] if fr.unit else []):
                ff = ex.unit_factor(k, u)
                if ff and ff > 1:
                    fac = ff
                    break
            if not fac:
                raise NotSupported("unknown big unit")
            big = int(v // fac)
            return Answer("list", [big, _num(v - big * fac)])
        if it == "stock_value":
            return Answer("number", self._stock_value(fr.snapshot))
        if it == "out_of_stock":
            return Answer("number", self._out_of_stock(fr.snapshot))
        if it == "debt_total":
            return Answer("number", ex.debt_total(fr.snapshot))
        if it == "debtor_count":
            return Answer("number", len(ex.debtors(fr.snapshot)))
        if it == "top_debtors":
            items = ex.debtors(fr.snapshot)[: fr.k or 3]
            return Answer("table", {"columns": ["khach_hang", "so_tien_no"],
                                    "rows": [[ex.cust_name(k), money(v)] for k, v in items]})
        if it == "debt_aging":
            return Answer("list", ex.debt_aging(fr.snapshot, fr.params["buckets"]))
        if it == "debt_overdue":
            n = fr.params["days"]
            return Answer("list" if False else "number", ex.debt_aging(fr.snapshot, [(n + 1, None)])[0])
        if it == "debt_flow":
            a, b = ex.debt_flow(P, f.get("customer"))
            return Answer("list", [a, b])
        if it == "repayments":
            a, b = ex.debt_flow(P, None)
            return Answer("number", b)
        if it in ("cust_min_orders", "cust_min_orders_pct"):
            items = ex.rank_customers(P, "orders")
            op, n = fr.params["op"], fr.params["n"]
            c = sum(1 for _, v in items if (v >= n if op == ">=" else v > n))
            if it == "cust_min_orders":
                return Answer("number", c)
            if not items:
                raise Unanswerable("không có khách")
            return Answer("percent", r2(c / len(items) * 100))
        if it == "products_min_orders":
            items = ex.rank_products(P, "orders")
            op, n = fr.params["op"], fr.params["n"]
            return Answer("number", sum(1 for _, v in items if (v >= n if op == ">=" else v > n)))
        if it == "unsold_products":
            return Answer("number", self._unsold(P))
        if it == "days_revenue":
            op, x = fr.params["thr"]
            return Answer("number", ex.days_with_revenue(P, op, x))
        if it == "orders_min_amount":
            ff = dict(f)
            ff["min_amount"] = fr.params["thr"]
            return Answer("number", ex.order_count(P, ff))
        if it == "gross_profit":
            return Answer("number", ex.gross_profit(P, f))
        if it == "net_profit":
            return Answer("number", self._net_profit(P))
        if it == "delivery_trips":
            return Answer("number", self._delivery_trips(P))
        if it == "peak_hour":
            O = ex.orders(P, True)
            if "hour" not in O.columns or O["hour"].isna().all():
                raise Unanswerable("không có giờ tạo đơn")
            s = O.groupby("hour").size()
            mx = s.max()
            return Answer("number", int(min(h for h, v in s.items() if v == mx)))
        if it == "avg_weekday":
            Pc = ex.clip(P)
            wd = fr.params["weekday"]
            ndays = sum(1 for i in range((Pc.end - Pc.start).days + 1) if (Pc.start + dt.timedelta(i)).weekday() == wd)
            if ndays == 0:
                raise Unanswerable("kỳ không có ngày thứ đó")
            if fr.params.get("metric") == "order_count":
                O = ex.orders(Pc, True)
                tot = int(sum(1 for d in O["date"] if d.weekday() == wd))
                return Answer("number", r2(tot / ndays))
            s = ex.daily_revenue(Pc)
            tot = sum(v for d, v in s.items() if d.weekday() == wd)
            return Answer("number", money(tot / ndays))
        if it == "special_lines":
            return Answer("number", self._special_lines(P, fr.params["what"]))
        if it == "new_members":
            C = ex.b.customers
            if C.empty or "registered" not in C.columns or C["registered"].isna().all():
                raise Unanswerable("không có ngày đăng ký thành viên")
            return Answer("number", int(sum(1 for d in C["registered"] if d is not None and P.start <= d <= P.end)))
        if it == "customers_on_street":
            C = ex.b.customers
            if C.empty or "address" not in C.columns or C["address"].isna().all():
                raise Unanswerable("không có địa chỉ khách")
            st = fold_key(fr.params["street"])
            ids = set()
            for i, a in zip(C.get("customer_id", C.index), C["address"]):
                if isinstance(a, str) and st and st in fold_key(a):
                    ids.add(i)
            return Answer("number", len(ids))
        if it == "expiring":
            return self._expiring(fr)
        if it == "nearest_expiry":
            return Answer("date", self._nearest_expiry(f["product"], fr.snapshot))
        raise NotSupported(it)

    # ------------------------------------------------------------------ helpers
    def _metric(self, metric: str, p: Period, f: dict) -> float:
        if metric == "order_count":
            return self.ex.order_count(p, f)
        return self.ex.revenue(p, f)

    def _growth(self, fr: Frame) -> float:
        ex = self.ex
        m = fr.params.get("metric", "revenue")
        mode = fr.params.get("mode")
        f = dict(fr.filters)
        f.pop("shift", None)
        if mode == "shift":
            P = ex.clip(fr.periods[0])
            a = self._metric(m, P, dict(f, shift="chieu"))
            b = self._metric(m, P, dict(f, shift="sang"))
            return pct_change(a, b)
        if mode == "tet_yoy":
            y = fr.params["year"]
            p1, p0 = tet_period(self.cal, y), tet_period(self.cal, y - 1)
            if p1 is None or p0 is None:
                raise Unanswerable("không có lịch Tết")
            return pct_change(self._metric(m, ex.clip(p1), f), self._metric(m, ex.clip(p0), f))
        if mode == "before_tet":
            y = fr.params["year"]
            p1 = tet_period(self.cal, y)
            n = fr.params.get("days") or -1
            if n < 0:
                n = p1.days()
            p0 = Period(p1.start - dt.timedelta(n), p1.start - dt.timedelta(1), "pretet", "range")
            return pct_change(self._metric(m, ex.clip(p1), f), self._metric(m, ex.clip(p0), f))
        P = fr.periods[0]
        if mode == "mtd":
            Pc = Period(P.start, min(P.end, ex.today), P.label, "mtd")
            base = _mtd_prev(Pc)
        elif mode == "yoy":
            Pc = P
            base = _yoy(P)
        elif mode == "explicit":
            Pc, base = fr.periods[0], fr.periods[1]
        else:
            Pc = P
            base = _prev_period(P)
        cur = self._metric(m, ex.clip(Pc), f)
        prv = self._metric(m, ex.clip(base), f)
        return pct_change(cur, prv)

    def _rank(self, fr: Frame) -> Answer:
        ex = self.ex
        P = fr.periods[0]
        k = fr.k or 1
        if fr.intent == "top_products":
            items = ex.rank_products(P, fr.by or "qty", {kk: v for kk, v in fr.filters.items() if kk in ("group", "rx", "ingredient", "size")})
            names = [ex.pname(key) for key, _ in items]
        elif fr.intent == "top_customers":
            items = ex.rank_customers(P, fr.by or "revenue")
            names = [ex.cust_name(key) for key, _ in items]
        elif fr.intent == "top_staff":
            items = ex.rank_staff(P, fr.by or "revenue")
            names = [key for key, _ in items]
        else:
            items = ex.rank_suppliers(P, fr.by or "amount")
            names = [key for key, _ in items]
        if not items:
            raise Unanswerable("không có dữ liệu xếp hạng")
        if fr.answer_type == "string" or (k == 1 and fr.answer_type != "list"):
            return Answer("string", names[0], meta={"tie": len(items) > 1 and abs(items[0][1] - items[1][1]) < 1e-9})
        return Answer("list", names[:k], meta={"tie": len(items) > k and abs(items[k - 1][1] - items[k][1]) < 1e-9})

    def _special_lines(self, P: Period, what: str) -> int:
        """Voucher / loyalty-point / shipping amounts: positive, by invoice date, valid invoices only."""
        ex = self.ex
        Pc = ex.clip(P)
        O = ex.orders(Pc, True)
        oids = set(O["order_id"])
        L = ex.L
        kind = {"voucher": "voucher", "points": "points", "shipping": "shipping"}[what]
        sub = L[(L["kind"] == kind) & L["order_id"].isin(oids)]
        if not sub.empty:
            return money(abs(sub["amount"].fillna(0)).sum())
        if what == "shipping" and "shipping_fee" in O.columns and O["shipping_fee"].notna().any():
            return money(O["shipping_fee"].fillna(0).sum())
        if what == "voucher" and "order_discount" in O.columns and O["order_discount"].notna().any():
            return money(O["order_discount"].fillna(0).abs().sum())
        raise Unanswerable("dữ liệu không ghi khoản này")

    def _price_effect(self, fr: Frame) -> float:
        ex = self.ex
        key = fr.filters["product"]
        d = fr.params.get("date")
        n = fr.params["n"]
        if d is None:
            ch = ex.price_change_dates(key)
            newp = fr.params.get("new_price")
            cands = [c for c in ch if c[2] is not None and c[1] > c[2] and (newp is None or abs(c[1] - newp) < 1)]
            if not cands:
                raise NotSupported("price change date unknown")
            d = cands[0][0]
        after = ex.qty_window(key, d, d + dt.timedelta(n - 1))
        before = ex.qty_window(key, d - dt.timedelta(n), d - dt.timedelta(1))
        return pct_change(after, before)

    def _purchase_minus_sales(self, fr: Frame) -> float:
        ex = self.ex
        key = fr.filters["product"]
        P = fr.periods[0]
        PL = ex.purchase_lines(P)
        inb = PL.loc[PL["pkey"] == key, "base_qty"].sum()
        sold = ex.lines(P, {"product": key})["base_qty"].sum()
        v = inb - sold
        if fr.unit:
            fac = ex.unit_factor(key, fr.unit)
            if fac and fac != 1:
                v = v / fac
        return _num(v)

    def _stock_value(self, day: dt.date) -> int:
        ex = self.ex
        P = ex.b.products
        if P.empty or "cost" not in P.columns or P["cost"].isna().all():
            raise Unanswerable("danh mục không có giá vốn")
        tot = 0.0
        for k, c in zip(P["key"], P["cost"]):
            if c is None or (isinstance(c, float) and math.isnan(c)):
                continue
            try:
                s = ex.stock_at(k, day)
            except Unanswerable:
                continue
            if s > 0:
                tot += s * c
        return money(tot)

    def _out_of_stock(self, day: dt.date) -> int:
        ex = self.ex
        keys = set(ex.b.stock_end["pkey"].dropna()) if not ex.b.stock_end.empty else set()
        if not keys and not ex.b.stock_card.empty:
            keys = set(ex.b.stock_card["product_code"].dropna())
        if not keys:
            raise Unanswerable("không có tồn kho")
        return sum(1 for k in keys if abs(ex.stock_at(k, day)) < 1e-9)

    def _unsold(self, P: Period) -> int:
        ex = self.ex
        Pc = ex.clip(P)
        L = ex.L
        valid = L[L["valid"] & (~L["neg"]) & (L["kind"] == "item")]
        before = set(valid.loc[valid["date"] < Pc.start, "pkey"].dropna())
        during = set(valid.loc[(valid["date"] >= Pc.start) & (valid["date"] <= Pc.end), "pkey"].dropna())
        return len(before - during)

    def _net_profit(self, P: Period) -> int:
        ex = self.ex
        gp = ex.gross_profit(P)
        E = ex.b.expenses
        if E.empty:
            raise Unanswerable("không có sổ chi phí")
        Pc = ex.clip(P)
        sub = E[E["date"].map(lambda d: d is not None and Pc.start <= d <= P.end)]
        return money(gp - sub["amount"].sum())

    def _delivery_trips(self, P: Period) -> int:
        ex = self.ex
        D = ex.b.deliveries
        if D.empty:
            raise Unanswerable("không có sổ giao hàng")
        Pc = Period(P.start, min(P.end, ex.today), P.label, P.kind)
        sub = D[D["date"].map(lambda d: d is not None and Pc.start <= d <= Pc.end)]
        return int(sub[["order_id", "trip_no"]].drop_duplicates().shape[0])

    def _lot_remaining(self, day: dt.date) -> "pd.DataFrame":
        import pandas as pd
        ex = self.ex
        Lt = ex.b.lots
        LS = ex.b.lot_sales
        if Lt.empty:
            raise Unanswerable("không có dữ liệu lô")
        O = ex.O
        od = dict(zip(O["order_id"], O["date"])) if not O.empty else {}
        ost = dict(zip(O["order_id"], O["status_final"])) if not O.empty else {}
        sold = {}
        if not LS.empty:
            for o, lot, q, r in zip(LS["order_id"], LS["lot"], LS["qty_base"], LS.get("qty_returned", [0] * len(LS))):
                d = od.get(o)
                if d is None or d > day or ost.get(o) == "cancelled":
                    continue
                qv = (q or 0) - ((r or 0) if not (isinstance(r, float) and math.isnan(r)) else 0)
                sold[lot] = sold.get(lot, 0) + qv
        rows = []
        H = ex.b.purchases
        rdate = dict(zip(H["receipt_id"], H["date"])) if not H.empty else {}
        for _, r in Lt.iterrows():
            rd = rdate.get(r.get("receipt_id"))
            if rd is not None and rd > day:
                continue
            rem = (r.get("qty_base") or 0) - sold.get(r.get("lot"), 0)
            rows.append({"lot": r.get("lot"), "pkey": r.get("pkey") or r.get("product_code"), "expiry": r.get("expiry"),
                         "remaining": rem, "unit_cost": r.get("unit_cost")})
        return pd.DataFrame(rows)

    def _expiring(self, fr: Frame) -> Answer:
        ex = self.ex
        day = fr.snapshot
        end = fr.params["end"]
        if not self.ex.has_lot_stock():
            raise Unanswerable("không có dữ liệu lô")
        R = self.ex.lot_remaining(day)
        R = R[(R["remaining"] > 0) & R["expiry"].map(lambda d: d is not None and day <= d <= end)]
        what = fr.params["what"]
        if what == "lot_count":
            return Answer("number", int(len(R)))
        if what == "product_count":
            return Answer("number", int(R["pkey"].nunique()))
        if what == "value":
            if R["unit_cost"].isna().all():
                raise Unanswerable("không có giá nhập lô")
            return Answer("number", money((R["remaining"] * R["unit_cost"]).sum()))
        g = R.groupby("pkey")["expiry"].min().reset_index()
        g["code"] = g["pkey"].map(ex.code_of)
        g = g.sort_values(["expiry", "code"])
        return Answer("list", [ex.pname(k) for k in g["pkey"]])

    def _nearest_expiry(self, key: str, day: dt.date) -> str:
        if not self.ex.has_lot_stock():
            raise Unanswerable("không có dữ liệu lô")
        R = self.ex.lot_remaining(day)
        R = R[(R["pkey"] == key) & (R["remaining"] > 0) & R["expiry"].map(lambda d: d is not None and d >= day)]
        if R.empty:
            raise Unanswerable("không có lô còn tồn")
        return min(R["expiry"]).isoformat()
