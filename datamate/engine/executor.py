"""Rulebook-faithful metric executor over a canonical ``Biz``.

Every public method returns plain Python values and raises ``Unanswerable`` when the data
cannot support the question (rulebook §7) or ``NotSupported`` when the executor lacks the
capability (the agent then abstains rather than guessing).
"""
from __future__ import annotations

import datetime as dt
import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..ingest.builder import Biz, norm_payment
from ..nlu.timeparse import Period
from ..textnorm import fold, fold_key
from ..values import round_half_up


class Unanswerable(Exception):
    pass


class NotSupported(Exception):
    pass


def r2(x: float) -> float:
    return float(round_half_up(x, 2))


def money(x: float) -> int:
    return int(round_half_up(x, 0))


def pct_change(cur: float, base: float) -> float:
    if base == 0:
        raise Unanswerable("kỳ so sánh bằng 0")
    return r2((cur - base) / base * 100.0)


SHIFTS = {"sang": (6, 13), "chieu": (14, 21)}


class Executor:
    def __init__(self, biz: Biz, today: dt.date):
        self.b = biz
        self.today = today
        self.O = biz.orders
        self.L = biz.lines
        self.has_orders = not self.O.empty
        self.has_lines = not self.L.empty
        lo = biz.data_start
        hi = min(biz.data_end or today, today) if biz.data_end else today
        self.data_lo, self.data_hi = lo, hi

    # ------------------------------------------------------------------ period helpers
    def clip(self, p: Period, allow_partial: bool = True) -> Period:
        """Clip to [data_start, min(data_end, today)]; raise if fully outside."""
        lo, hi = self.data_lo, self.data_hi
        s, e = p.start, min(p.end, self.today)
        if lo is None or hi is None:
            raise Unanswerable("không có dữ liệu ngày")
        if e < lo or s > hi or s > e:
            raise Unanswerable(f"kỳ {p.start}..{p.end} nằm ngoài khoảng dữ liệu {lo}..{hi}")
        return Period(max(s, lo), min(e, hi), p.label, p.kind, p.full_end)

    def _in(self, dates: pd.Series, p: Period) -> pd.Series:
        return dates.map(lambda d: d is not None and isinstance(d, dt.date) and p.start <= d <= p.end)

    # ------------------------------------------------------------------ order selection
    def orders(self, p: Optional[Period], valid: Optional[bool] = True, filt: Optional[dict] = None,
               clip: bool = True) -> pd.DataFrame:
        if not self.has_orders:
            raise NotSupported("no order table")
        O = self.O
        if p is not None:
            if clip:
                p = self.clip(p)
            m = (O["date"] >= p.start) & (O["date"] <= p.end) if O["date"].map(lambda x: isinstance(x, dt.date)).all() \
                else self._in(O["date"], p)
            O = O[m]
        if valid is True:
            O = O[O["valid"]]
        if filt:
            O = self._order_filters(O, filt)
        return O

    def _order_filters(self, O: pd.DataFrame, f: dict) -> pd.DataFrame:
        if f.get("shift"):
            if "datetime" not in O.columns or O["datetime"].isna().all():
                raise Unanswerable("không có giờ tạo đơn")
            a, b = SHIFTS[f["shift"]]
            O = O[O["hour"].map(lambda h: h is not None and not (isinstance(h, float) and math.isnan(h)) and a <= h <= b)]
        if f.get("pay"):
            if "pay_method_n" not in O.columns or O["pay_method_n"].isna().all():
                raise Unanswerable("không có hình thức thanh toán")
            O = O[O["pay_method_n"] == f["pay"]]
        if f.get("street"):
            C = self.b.customers
            if C.empty or "address" not in C.columns or C["address"].isna().all():
                raise Unanswerable("không có địa chỉ khách")
            st = fold_key(f["street"])
            keys = set()
            ids = C["customer_id"] if "customer_id" in C.columns else [None] * len(C)
            names = C["customer_name"] if "customer_name" in C.columns else [None] * len(C)
            for cid, nm, ad in zip(ids, names, C["address"]):
                if isinstance(ad, str) and st and st in fold_key(ad):
                    if isinstance(cid, str) and cid:
                        keys.add(cid)
                    if isinstance(nm, str):
                        keys.add("C:" + fold_key(nm))
            O = O[O["cust_key"].isin(keys)]
        if f.get("channel"):
            if "channel_n" not in O.columns or O["channel_n"].isna().all():
                raise Unanswerable("không có kênh bán")
            O = O[O["channel_n"] == f["channel"]]
        if f.get("cust_group"):
            if "cust_group" not in O.columns or O["cust_group"].isna().all():
                raise Unanswerable("không có nhóm khách")
            g = fold_key(f["cust_group"])
            O = O[O["cust_group"].map(lambda x: x is not None and not (isinstance(x, float)) and fold_key(str(x)) == g)]
        if f.get("staff"):
            if "staff" not in O.columns or O["staff"].isna().all():
                raise Unanswerable("không có nhân viên bán")
            O = O[O["staff"].map(lambda x: x is not None and str(x).strip().upper() == f["staff"].upper())]
        if f.get("customer"):
            O = O[O["cust_key"] == f["customer"]]
        if f.get("min_amount") is not None:
            op, v = f["min_amount"]
            col = O["revenue"]
            O = O[col >= v] if op == ">=" else O[col > v] if op == ">" else O[col <= v] if op == "<=" else O[col < v]
        return O

    # ------------------------------------------------------------------ line selection
    def lines(self, p: Optional[Period], filt: Optional[dict] = None, valid: bool = True, items_only: bool = True,
              clip: bool = True) -> pd.DataFrame:
        if not self.has_lines:
            raise NotSupported("no line table")
        L = self.L
        if p is not None:
            if clip:
                p = self.clip(p)
            L = L[(L["date"] >= p.start) & (L["date"] <= p.end)] if L["date"].map(lambda x: isinstance(x, dt.date)).all() \
                else L[self._in(L["date"], p)]
        if valid:
            L = L[L["valid"]]
        L = L[~L["neg"]]
        if items_only:
            L = L[L["kind"] == "item"]
        if filt:
            order_f = {k: v for k, v in filt.items() if k in ("shift", "pay", "channel", "cust_group", "staff", "customer",
                                                              "min_amount", "street")}
            if order_f:
                if not self.has_orders:
                    raise Unanswerable("không có thông tin đơn")
                oids = set(self._order_filters(self.O[self.O["order_id"].isin(set(L["order_id"].dropna()))], order_f)["order_id"])
                L = L[L["order_id"].isin(oids)]
            L = self._line_filters(L, filt)
        return L

    def _line_filters(self, L: pd.DataFrame, f: dict) -> pd.DataFrame:
        if f.get("product"):
            keys = f["product"] if isinstance(f["product"], (list, set, tuple)) else [f["product"]]
            L = L[L["pkey"].isin(set(keys))]
        if f.get("group"):
            g = fold_key(f["group"])
            gm = self.product_groups()
            if not gm:
                raise Unanswerable("không có nhóm hàng")
            keys = {k for k, v in gm.items() if fold_key(v) == g}
            L = L[L["pkey"].isin(keys)]
        if f.get("ingredient"):
            ing = fold_key(f["ingredient"])
            dm = self.drug_info()
            if not dm:
                raise Unanswerable("không có hoạt chất")
            keys = {k for k, (a, rx) in dm.items() if a and fold_key(a) == ing}
            L = L[L["pkey"].isin(keys)]
        if f.get("rx") is not None:
            dm = self.drug_info()
            if not dm:
                raise Unanswerable("không có thông tin kê đơn")
            rxk = {k for k, (a, rx) in dm.items() if rx is True}
            L = L[L["pkey"].isin(rxk)] if f["rx"] else L[~L["pkey"].isin(rxk)]
        if f.get("size"):
            sz = str(f["size"]).lower()
            P = self.b.products
            if not P.empty and "size" in P.columns and P["size"].notna().any():
                # rulebook (mẹ & bé): size / milk number comes from the catalogue column only
                def _sz(v):
                    return re.sub(r"^(size|so|co)", "", fold_key(str(v)))
                keys = {k for k, v in zip(P["key"], P["size"]) if isinstance(v, str) and _sz(v) == _sz(sz)}
                keys |= {k for k, v in zip(P["key"], P["size"]) if isinstance(v, (int, float)) and str(v).rstrip("0").rstrip(".") == sz}
                L = L[L["pkey"].isin(keys)]
            else:
                L = L[L["pkey"].map(lambda k: bool(re.search(r"\(\s*%s\s*\)\s*$" % re.escape(sz), self.b.product_name(k).lower())))]
        if f.get("brand"):
            P = self.b.products
            if P.empty or "brand" not in P.columns:
                raise Unanswerable("không có hãng trong danh mục")
            keys = {k for k, v in zip(P["key"], P["brand"]) if isinstance(v, str) and fold_key(v) == fold_key(f["brand"])}
            L = L[L["pkey"].isin(keys)]
        if f.get("base_unit"):
            u = fold_key(f["base_unit"])
            L = L[L["pkey"].map(lambda k: fold_key(self.base_unit_of(k) or "") == u)]
        return L

    # ------------------------------------------------------------------ catalogue helpers
    def product_groups(self) -> Dict[str, str]:
        if hasattr(self, "_pg"):
            return self._pg
        gm = {}
        P = self.b.products
        if not P.empty and "group" in P.columns:
            for k, g in zip(P["key"], P["group"]):
                if g:
                    gm[k] = g
        if not gm and self.has_lines and "product_group" in self.L.columns:
            for k, g in zip(self.L["pkey"], self.L["product_group"]):
                if k and isinstance(g, str) and g.strip():
                    gm.setdefault(k, g.strip())
        self._pg = gm
        return gm

    def drug_info(self) -> Dict[str, Tuple[Optional[str], Optional[bool]]]:
        d = getattr(self.b, "_drug", {}) or {}
        return d

    def base_unit_of(self, key: str) -> Optional[str]:
        bu = self.b.base_unit(key)
        if bu:
            return bu
        if not hasattr(self, "_bu_guess"):
            g = {}
            if self.has_lines:
                sub = self.L[self.L["kind"] == "item"]
                for k, u, q, bq in zip(sub["pkey"], sub["unit_eff"], sub["qty"], sub["base_qty"]):
                    if k and isinstance(u, str) and q and bq and abs(q - bq) < 1e-9:
                        g.setdefault(k, {}).setdefault(u, 0)
                        g[k][u] += 1
            self._bu_guess = {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in g.items()}
        return self._bu_guess.get(key)

    def unit_factor(self, key: str, unit: str) -> Optional[float]:
        bu = self.base_unit_of(key)
        if bu and fold_key(bu) == fold_key(unit):
            return 1.0
        return self.b.unit_factor.get((key, fold_key(unit)))

    def pname(self, k: str) -> str:
        return self.b.product_name(k)

    def code_of(self, k: str) -> str:
        c = self.b.product_code(k)
        return c if c else self.pname(k)

    # ------------------------------------------------------------------ revenue & orders
    def revenue(self, p: Period, filt: Optional[dict] = None) -> int:
        f = filt or {}
        line_f = any(f.get(k) is not None for k in ("product", "group", "ingredient", "rx", "size", "base_unit", "brand"))
        if line_f:
            L = self.lines(p, f)
            return money(L["amount"].sum())
        if self.has_orders:
            O = self.orders(p, True, f)
            return money(O["revenue"].sum())
        L = self.lines(p, f, items_only=False)
        L = L[L["kind"].isin(["item", "service_fee"])]
        return money(L["amount"].sum())

    def order_count(self, p: Period, filt: Optional[dict] = None, status: str = "valid") -> int:
        if not self.has_orders:
            raise Unanswerable("sổ không ghi mã đơn nên không đếm được số đơn")
        f = filt or {}
        line_f = {k: v for k, v in f.items() if k in ("product", "group", "ingredient", "rx", "size", "brand")}
        if status == "valid":
            O = self.orders(p, True, {k: v for k, v in f.items() if k not in line_f})
        elif status == "cancelled":
            O = self.orders(p, None, {k: v for k, v in f.items() if k not in line_f})
            O = O[O["status_final"] == "cancelled"]
        elif status == "returned":
            O = self.orders(p, None, f)
            O = O[O["status_final"] == "returned"]
        else:
            O = self.orders(p, None, f)
        if line_f:
            L = self.lines(p, line_f)
            O = O[O["order_id"].isin(set(L["order_id"]))]
        return int(len(O))

    def cancel_rate(self, p: Period, filt: Optional[dict] = None) -> float:
        O = self.orders(p, None, filt)
        if len(O) == 0:
            raise Unanswerable("không có đơn")
        return r2((O["status_final"] == "cancelled").sum() / len(O) * 100)

    def aov(self, p: Period, filt: Optional[dict] = None) -> int:
        O = self.orders(p, True, filt)
        if len(O) == 0:
            raise Unanswerable("không có đơn")
        return money(O["revenue"].sum() / len(O))

    def units(self, p: Period, filt: Optional[dict] = None, unit: Optional[str] = None) -> float:
        L = self.lines(p, filt)
        if unit and filt and filt.get("product"):
            keys = filt["product"] if isinstance(filt["product"], (list, set, tuple)) else [filt["product"]]
            tot = 0.0
            for k in keys:
                f = self.unit_factor(k, unit)
                q = L.loc[L["pkey"] == k, "base_qty"].sum()
                if f is None:
                    f = 1.0
                tot += q / f
            return _num(tot)
        return _num(L["base_qty"].sum())

    def daily_revenue(self, p: Period, filt=None) -> pd.Series:
        if self.has_orders and not filt:
            O = self.orders(p, True)
            return O.groupby("date")["revenue"].sum()
        L = self.lines(p, filt, items_only=False)
        L = L[L["kind"].isin(["item", "service_fee"])]
        return L.groupby("date")["amount"].sum()

    def best_day(self, p: Period, by: str = "revenue") -> str:
        if by == "orders":
            O = self.orders(p, True)
            s = O.groupby("date").size()
        else:
            s = self.daily_revenue(p)
        if s.empty:
            raise Unanswerable("không có doanh thu")
        mx = s.max()
        d = min(k for k, v in s.items() if abs(v - mx) < 1e-6)
        return d.isoformat()

    def days_with_revenue(self, p: Period, op: str, x: float) -> int:
        s = self.daily_revenue(p)
        s = s.round(0)
        return int((s >= x).sum() if op == ">=" else (s > x).sum() if op == ">" else (s <= x).sum() if op == "<=" else (s < x).sum())

    # ------------------------------------------------------------------ rankings
    def rank_products(self, p: Period, by: str = "qty", filt: Optional[dict] = None) -> List[Tuple[str, float]]:
        L = self.lines(p, filt)
        L = L[L["pkey"].notna()]
        if by == "qty":
            s = L.groupby("pkey")["base_qty"].sum()
        elif by == "revenue":
            s = L.groupby("pkey")["amount"].sum()
        elif by == "profit":
            if L["cost_unit"].isna().all():
                raise Unanswerable("không có giá vốn")
            s = (L["amount"] - L["qty"] * L["cost_unit"]).groupby(L["pkey"]).sum()
        elif by == "orders":
            s = L.groupby("pkey")["order_id"].nunique()
        else:
            raise NotSupported(by)
        items = [(k, float(v)) for k, v in s.items()]
        items.sort(key=lambda kv: (-round(kv[1], 6), self.code_of(kv[0])))
        return items

    def rank_customers(self, p: Period, by: str = "revenue", min_orders: int = 0) -> List[Tuple[str, float]]:
        if by == "cancels":
            O = self.orders(p, None)
            O = O[(O["status_final"] == "cancelled") & O["cust_key"].notna()]
            s = O.groupby("cust_key").size()
        else:
            O = self.orders(p, True)
            O = O[O["cust_key"].notna()]
            if by == "revenue":
                s = O.groupby("cust_key")["revenue"].sum()
            elif by == "orders":
                s = O.groupby("cust_key").size()
            elif by == "days":
                s = O.groupby("cust_key")["date"].nunique()
            elif by == "aov":
                g = O.groupby("cust_key")["revenue"]
                cnt = g.size()
                s = (g.sum() / cnt)[cnt >= max(min_orders, 3)].map(lambda x: round_half_up(x, 0))
            else:
                raise NotSupported(by)
        if self.O["cust_key"].isna().all():
            raise Unanswerable("dữ liệu không ghi khách nên không xác định được khách")
        items = [(k, float(v)) for k, v in s.items()]
        items.sort(key=lambda kv: (-round(kv[1], 6), kv[0]))
        return items

    def cust_name(self, k: str) -> str:
        return getattr(self.b, '_cust_name', {}).get(k, k)

    def rank_staff(self, p: Period, by: str = "revenue") -> List[Tuple[str, float]]:
        O = self.orders(p, True)
        if "staff" not in O.columns or O["staff"].isna().all():
            raise Unanswerable("không có nhân viên bán")
        O = O[O["staff"].notna()]
        if by == "revenue":
            s = O.groupby("staff")["revenue"].sum()
        elif by == "tickets":
            s = O.groupby("staff").size()
        else:
            s = O.groupby("staff")["date"].nunique()
        items = [(k, float(v)) for k, v in s.items()]
        items.sort(key=lambda kv: (-round(kv[1], 6), kv[0]))
        return items

    # ------------------------------------------------------------------ prices
    def price_on(self, key: str, day: dt.date) -> int:
        L = self.L
        sub = L[(L["pkey"] == key) & (L["date"] == day) & (L["kind"] == "item") & (~L["neg"])]
        if sub.empty:
            raise Unanswerable("không có dòng bán ngày đó")
        vals = []
        for u, pr in zip(sub["unit_eff"], sub["unit_price"]):
            if pr is None or (isinstance(pr, float) and math.isnan(pr)):
                continue
            f = self.unit_factor(key, u) if isinstance(u, str) else 1.0
            vals.append(pr / (f or 1.0))
        if not vals:
            raise Unanswerable("không có đơn giá")
        s = pd.Series(vals).round(2).value_counts()
        return money(s.index[0])

    def price_change_dates(self, key: str) -> List[Tuple[dt.date, float, Optional[float]]]:
        P = self.b.prices
        out = []
        if not P.empty and "pkey" in P.columns:
            sub = P[P["pkey"] == key].copy()
            sub = sub[sub["date"].notna()].sort_values("date")
            prev = None
            for d, np_, op in zip(sub["date"], sub["new_price"], sub.get("old_price", [None] * len(sub))):
                out.append((d, np_, prev if prev is not None else op))
                prev = np_
        return out

    def qty_window(self, key: str, start: dt.date, end: dt.date) -> float:
        L = self.L
        sub = L[(L["pkey"] == key) & (L["date"] >= start) & (L["date"] <= end) & L["valid"] & (~L["neg"]) & (L["kind"] == "item")]
        return float(sub["base_qty"].sum())

    # ------------------------------------------------------------------ purchases
    def purchases_df(self, p: Period) -> pd.DataFrame:
        H = self.b.purchases
        if H.empty or "date" not in H.columns:
            raise Unanswerable("không có sổ nhập hàng")
        H = H[H["date"].notna()]
        lo = H["date"].min()
        if p.end < lo or p.start > self.today:
            raise Unanswerable("kỳ ngoài khoảng dữ liệu nhập")
        H = H[(H["date"] >= p.start) & (H["date"] <= min(p.end, self.today))]
        return H[~H["cancelled"]]

    def purchase_total(self, p: Period) -> int:
        H = self.purchases_df(p)
        if "total" not in H.columns or H["total"].isna().all():
            raise Unanswerable("không có tiền nhập")
        return money(H["total"].sum())

    def purchase_lines(self, p: Period) -> pd.DataFrame:
        PL = self.b.purchase_lines
        if PL.empty:
            raise Unanswerable("không có chi tiết nhập")
        H = self.purchases_df(p)
        return PL[PL["receipt_id"].isin(set(H["receipt_id"]))]

    def avg_purchase_price(self, key: str, p: Period, unit: Optional[str]) -> int:
        PL = self.purchase_lines(p)
        sub = PL[PL["pkey"] == key]
        if sub.empty or "unit_cost" not in sub.columns or sub["unit_cost"].isna().all():
            raise Unanswerable("không có phiếu nhập mặt hàng trong kỳ")
        amt = (sub["qty"] * sub["unit_cost"]).sum() if "amount" not in sub.columns or sub["amount"].isna().any() else sub["amount"].sum()
        # quantity expressed in the requested unit
        q = 0.0
        for qq, u in zip(sub["qty"], sub.get("unit", [None] * len(sub))):
            if unit and isinstance(u, str) and fold_key(u) != fold_key(unit):
                fu = self.unit_factor(key, u) or 1.0
                ft = self.unit_factor(key, unit) or 1.0
                q += qq * fu / ft
            else:
                q += qq
        if q == 0:
            raise Unanswerable("số lượng nhập bằng 0")
        return money(amt / q)

    def rank_suppliers(self, p: Period, by: str = "amount") -> List[Tuple[str, float]]:
        H = self.purchases_df(p)
        col = "supplier_name" if "supplier_name" in H.columns and H["supplier_name"].notna().any() else "supplier_id"
        if col not in H.columns:
            raise Unanswerable("không có nhà cung cấp")
        H = H[H[col].notna()]
        s = H.groupby(col)["total"].sum() if by == "amount" else H.groupby(col)["receipt_id"].nunique()
        items = [(k, float(v)) for k, v in s.items()]
        items.sort(key=lambda kv: (-round(kv[1], 6), kv[0]))
        return items

    # ------------------------------------------------------------------ inventory
    def _stock_series(self) -> Dict[str, float]:
        S = self.b.stock_end
        if S.empty or "stock_qty" not in S.columns:
            raise Unanswerable("không có số tồn kho")
        return {k: float(v) for k, v in zip(S["pkey"], S["stock_qty"]) if k and v is not None and not math.isnan(v)}

    def _lot_net(self):
        """Per-lot (qty in, received date) and per-order-dated net quantities sold from lots."""
        if hasattr(self, "_lotnet"):
            return self._lotnet
        Lt, LS = self.b.lots, self.b.lot_sales
        H = self.b.purchases
        rdate = dict(zip(H["receipt_id"], H["date"])) if not H.empty and "date" in H.columns else {}
        pk = Lt["pkey"] if "pkey" in Lt.columns else Lt["product_code"]
        lots = pd.DataFrame({"lot": Lt["lot"], "pkey": pk, "qty": Lt["qty_base"].astype(float),
                             "rdate": Lt["receipt_id"].map(rdate) if "receipt_id" in Lt.columns else None,
                             "expiry": Lt["expiry"] if "expiry" in Lt.columns else None,
                             "unit_cost": Lt["unit_cost"] if "unit_cost" in Lt.columns else np.nan})
        O = self.O
        od = dict(zip(O["order_id"], O["date"])) if not O.empty else {}
        sales = pd.DataFrame({"lot": LS["lot"], "order_id": LS["order_id"],
                              "net": LS["qty_base"].fillna(0).astype(float) -
                                     (LS["qty_returned"].fillna(0).astype(float) if "qty_returned" in LS.columns else 0.0)})
        sales["date"] = sales["order_id"].map(od)
        sales["pkey"] = sales["lot"].map(dict(zip(lots["lot"], lots["pkey"])))
        self._lotnet = (lots, sales)
        return self._lotnet

    def has_lot_stock(self) -> bool:
        return not self.b.lots.empty and not self.b.lot_sales.empty and "qty_base" in self.b.lots.columns

    def lot_remaining(self, day: dt.date) -> pd.DataFrame:
        lots, sales = self._lot_net()
        L = lots[lots["rdate"].map(lambda d: d is None or (isinstance(d, float)) or d <= day)].copy()
        s = sales[sales["date"].map(lambda d: d is not None and not isinstance(d, float) and d <= day)]
        sold = s.groupby("lot")["net"].sum()
        L["remaining"] = L["qty"] - L["lot"].map(sold).fillna(0.0)
        return L

    def stock_at(self, key: str, day: dt.date) -> float:
        sc = self.b.stock_card
        if not sc.empty:
            sub = sc[sc["product_code"] == key]
            sub = sub[sub["move_datetime"].map(lambda x: x is not None and x.date() <= day)]
            if sub.empty:
                return 0.0
            return float(sub.sort_values("move_datetime")["balance"].iloc[-1])
        if self.has_lot_stock():
            R = self.lot_remaining(day)
            return float(R.loc[R["pkey"] == key, "remaining"].sum())
        if self.b.stock_end.empty and not self.b.purchase_lines.empty:
            PL = self.b.purchase_lines
            canc = set(self.b.purchases.loc[self.b.purchases["cancelled"], "receipt_id"]) if not self.b.purchases.empty else set()
            pin = PL[(PL["pkey"] == key) & PL["date"].map(lambda d: d is not None and not isinstance(d, float) and d <= day)
                     & ~PL["receipt_id"].isin(canc)]["base_qty"].sum()
            L = self.L
            sub = L[(L["pkey"] == key) & (L["kind"] == "item") & L["date"].map(lambda d: d is not None and d <= day)]
            sold = sub[sub["status_final"] != "cancelled"]["base_qty"].sum()
            return float(pin - sold)
        end_stock = self._stock_series()
        if key not in end_stock:
            raise Unanswerable("mặt hàng không có trong báo cáo tồn")
        s = end_stock[key]
        end = self.b.data_end
        if day >= end:
            return s
        # roll back movements after `day`
        PL = self.b.purchase_lines
        if not PL.empty:
            sub = PL[(PL["pkey"] == key) & PL["date"].map(lambda d: d is not None and d > day)]
            if "cancelled" not in PL.columns and not self.b.purchases.empty:
                canc = set(self.b.purchases.loc[self.b.purchases["cancelled"], "receipt_id"])
                sub = sub[~sub["receipt_id"].isin(canc)]
            s -= float(sub["base_qty"].sum())
        L = self.L
        sub = L[(L["pkey"] == key) & L["date"].map(lambda d: d is not None and d > day) & (L["kind"] == "item")]
        st = sub["status_final"] if "status_final" in sub.columns else pd.Series(["completed"] * len(sub), index=sub.index)
        sold = sub[(st != "cancelled") & (~sub["neg"])]
        s += float(sold["base_qty"].sum())
        # returned goods came back into stock
        ret = sub[sub["neg"]]
        s -= float((-ret["base_qty"]).sum())
        return s

    # ------------------------------------------------------------------ debt
    def debt_orders(self) -> pd.DataFrame:
        O = self.O
        if O.empty:
            raise Unanswerable("không có đơn")
        O = O[O["status_final"] != "cancelled"]
        if "pay_method_n" in O.columns and O["pay_method_n"].notna().any():
            O = O[O["pay_method_n"] != "cod"]
        if "channel_n" in O.columns:
            O = O[O["channel_n"] != "ecommerce"]
        return O

    def paid_until(self, day: dt.date) -> Dict[str, float]:
        P = self.b.payments
        if P.empty:
            return {}
        sub = P[P["pay_date"].map(lambda d: d is not None and d <= day)]
        return sub.groupby("order_id")["pay_amount"].sum().to_dict()

    def outstanding(self, day: dt.date) -> pd.DataFrame:
        """Per-order outstanding amount at the end of `day` (only > 0)."""
        O = self.debt_orders()
        O = O[O["date"].map(lambda d: d is not None and d <= day)]
        if not self.b.payments.empty:
            paid = self.paid_until(day)
            rem = O["due"] - O["order_id"].map(paid).fillna(0.0)
        elif "remaining" in O.columns and O["remaining"].notna().any():
            if day < (self.b.data_end or day):
                raise Unanswerable("chỉ có số còn nợ tại cuối dữ liệu")
            rem = O["remaining"].fillna(0.0)
        elif "paid" in O.columns and O["paid"].notna().any():
            if day < (self.b.data_end or day):
                raise Unanswerable("chỉ có số đã trả tại cuối dữ liệu")
            rem = O["due"] - O["paid"].fillna(0.0)
        else:
            raise Unanswerable("không có dữ liệu công nợ")
        out = O.assign(outstanding=rem.round(2))
        return out[out["outstanding"] > 0.5]

    def debt_total(self, day: dt.date) -> int:
        return money(self.outstanding(day)["outstanding"].sum())

    def debtors(self, day: dt.date) -> List[Tuple[str, float]]:
        o = self.outstanding(day)
        o = o[o["cust_key"].notna()]
        s = o.groupby("cust_key")["outstanding"].sum()
        items = [(k, float(v)) for k, v in s.items()]
        items.sort(key=lambda kv: (-round(kv[1], 4), kv[0]))
        return items

    def debt_aging(self, day: dt.date, buckets: List[Tuple[int, Optional[int]]]) -> List[int]:
        o = self.outstanding(day)
        age = o["date"].map(lambda d: (day - d).days)
        out = []
        for lo, hi in buckets:
            m = (age >= lo) & ((age <= hi) if hi is not None else True)
            out.append(money(o.loc[m, "outstanding"].sum()))
        return out

    def debt_flow(self, p: Period, customer: Optional[str] = None) -> Tuple[int, int]:
        """(new debt in period, repayments in period) for credit orders."""
        p = self.clip(p)
        O = self.O[self.O["status_final"] != "cancelled"]
        if "pay_method_n" in O.columns and O["pay_method_n"].notna().any():
            O = O[O["pay_method_n"] == "debt"]
        if customer:
            O = O[O["cust_key"] == customer]
        P = self.b.payments
        if P.empty:
            raise Unanswerable("không có lịch sử trả tiền")
        Pm = P[P["order_id"].isin(set(O["order_id"]))]
        od = dict(zip(O["order_id"], O["date"]))
        same_day = Pm[[od.get(o) == d for o, d in zip(Pm["order_id"], Pm["pay_date"])]]
        paid_at_purchase = same_day.groupby("order_id")["pay_amount"].sum().to_dict()
        new = O[(O["date"] >= p.start) & (O["date"] <= p.end)]
        new_debt = sum(max(0.0, due - paid_at_purchase.get(o, 0.0)) for o, due in zip(new["order_id"], new["due"]))
        later = Pm[[(d is not None and od.get(o) is not None and d > od.get(o) and p.start <= d <= p.end)
                    for o, d in zip(Pm["order_id"], Pm["pay_date"])]]
        return money(new_debt), money(later["pay_amount"].sum())

    # ------------------------------------------------------------------ profit
    def gross_profit(self, p: Period, filt=None) -> int:
        L = self.lines(p, filt)
        if L["cost_unit"].isna().all():
            raise Unanswerable("không có giá vốn trên dòng bán")
        rev_all = self.revenue(p, filt) if not filt else money(L["amount"].sum())
        cost = (L["qty"] * L["cost_unit"]).sum()
        return money(rev_all - cost)


def _num(x: float):
    if abs(x - round(x)) < 1e-6:
        return int(round(x))
    return r2(x)
