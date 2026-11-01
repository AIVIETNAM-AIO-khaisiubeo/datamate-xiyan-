"""Assemble a canonical, rule-ready data model (``Biz``) from heterogeneous shop files.

Canonical frames (all optional except ``lines``/``orders``):
  lines           one row per sold line (item / service fee / voucher / points / shipping)
  orders          one row per order, with normalised status, customer, payment, totals
  payments        one row per payment received for an order
  products        catalogue (key, code, name, group, base unit, price, cost, drug info, ...)
  purchases       one row per purchase receipt, purchase_lines one row per received line
  stock_end       stock at the end of the data (base units)
  lots / lot_sales / deliveries / debt_terms / prices / expenses / customers / returns / ...
"""
from __future__ import annotations

import datetime as dt
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..textnorm import fold, fold_key
from ..values import (is_blank, parse_bool, parse_date, parse_datetime, parse_money, parse_number, parse_qty,
                      parse_time)
from .readme import ReadmeInfo, parse_readme
from .reader import RawTable, drop_total_and_note_rows, read_business_files
from .roles import assign_roles, classify

MONEY = {"unit_price", "amount", "line_discount", "cost_unit", "goods_total", "order_discount", "shipping_fee", "vat",
         "order_total", "paid", "remaining", "pay_amount", "price", "cost", "total", "unit_cost", "new_price",
         "old_price", "refund", "trade_in", "cash_in", "cash_refund", "cash_net"}
NUMS = {"base_qty", "stock_qty", "qty_base", "qty_returned", "trip_no", "days", "guests", "line_no", "balance",
        "points", "commission", "vat_rate"}
DATES = {"date", "due_date", "expiry", "created", "registered", "date_from", "date_to", "start"}
DATETIMES = {"datetime", "pay_datetime", "ret_datetime", "move_datetime"}
IDS = {"order_id", "receipt_id", "payment_id", "product_code", "customer_id", "supplier_id", "staff", "lot",
       "return_id", "doc_id"}
ORDER_LEVEL = {"order_id", "date", "datetime", "status", "customer_name", "customer_id", "customer_group", "phone",
               "address", "channel", "payment_method", "staff", "goods_total", "order_discount", "shipping_fee", "vat",
               "order_total", "paid", "remaining", "note", "table_no", "area", "guests", "trade_in"}

WALKIN = {"", "khach le", "le", "kl", "khach vang lai", "vang lai", "khach", "khach le.", "none", "nan", "-",
          "khachle", "guest", "walk in"}


def norm_status(v) -> Optional[str]:
    if is_blank(v):
        return None
    s = fold(str(v))
    if re.search(r"\bhuy\b|cancel|khong giao duoc|giao that bai|that bai|\bbom\b|\bboom\b|khong nhan|tu choi nhan|failed|\bfail\b|khong thanh cong", s):
        return "cancelled"
    if re.search(r"tra hang|tra ve|doi tra|hoan tra|return|tra lai|hoan hang", s):
        return "returned"
    if re.search(r"dang giao|shipping|van chuyen|cho giao|dang ship|delivering|dang gui|cho lay|cho xu ly|dang xu ly|pending|chua giao", s):
        return "shipping"
    return "completed"


def norm_payment(v) -> Optional[str]:
    if is_blank(v):
        return None
    s = fold(str(v))
    if re.search(r"\bcod\b|thu ho|giao hang thu tien", s):
        return "cod"
    if re.search(r"ghi no|\bdebt\b|cong no|tra sau|\bno\b|tra gop|installment", s):
        return "debt"
    if re.search(r"chuyen khoan|transfer|\bck\b|bank|ngan hang|qr", s):
        return "transfer"
    if re.search(r"vi dien tu|ewallet|e-wallet|momo|zalopay|zalo pay|vnpay|shopeepay|^vi$", s):
        return "ewallet"
    if re.search(r"tien mat|cash|\btm\b", s):
        return "cash"
    if re.search(r"\bthe\b|card|pos|quet the", s):
        return "card"
    return s


def norm_channel(v) -> Optional[str]:
    if is_blank(v):
        return None
    s = fold(str(v))
    if re.search(r"zalo", s):
        return "zalo"
    if re.search(r"\bsan\b|tmdt|shopee|lazada|tiki|tiktok|ecommerce|e-commerce|marketplace", s):
        return "ecommerce"
    if re.search(r"dien thoai|phone|goi", s):
        return "phone"
    if re.search(r"facebook|\bfb\b|messenger", s):
        return "facebook"
    if re.search(r"grab|shopeefood|baemin|gojek|befood|app giao", s):
        return "delivery_app"
    if re.search(r"web|online", s):
        return "online"
    if re.search(r"store|tai quay|tai tiem|tai cua hang|tai quan|truc tiep|offline|quay", s):
        return "store"
    return s


def customer_is_walkin(name) -> bool:
    if is_blank(name):
        return True
    return fold(str(name)).strip(" .") in WALKIN


def strip_phone(name: str) -> str:
    s = re.sub(r"[\s\-–(]*\(?\+?\d[\d .]{7,}\d\)?\s*$", "", str(name)).strip()
    return s or str(name)


@dataclass
class Biz:
    biz_id: str
    path: str
    readme: ReadmeInfo
    lines: pd.DataFrame
    orders: pd.DataFrame
    payments: pd.DataFrame
    products: pd.DataFrame
    unit_factor: Dict[Tuple[str, str], float]
    purchases: pd.DataFrame
    purchase_lines: pd.DataFrame
    stock_end: pd.DataFrame
    lots: pd.DataFrame
    lot_sales: pd.DataFrame
    deliveries: pd.DataFrame
    debt_terms: pd.DataFrame
    prices: pd.DataFrame
    expenses: pd.DataFrame
    customers: pd.DataFrame
    returns: pd.DataFrame
    tables_: pd.DataFrame
    stock_card: pd.DataFrame
    staff: pd.DataFrame
    suppliers: pd.DataFrame
    shifts: pd.DataFrame
    other: List[RawTable] = field(default_factory=list)
    raw: List[Tuple[str, Dict[str, str], RawTable]] = field(default_factory=list)
    data_start: Optional[dt.date] = None
    data_end: Optional[dt.date] = None
    flags: Dict[str, object] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    # --------------------------------------------------------------- helpers
    def product_name(self, key: str) -> str:
        r = self._pname.get(key)
        return r if r else key

    def product_code(self, key: str) -> Optional[str]:
        return self._pcode.get(key)

    def factor(self, pkey: str, unit: Optional[str]) -> Optional[float]:
        if unit is None:
            return 1.0
        return self.unit_factor.get((pkey, fold_key(unit)))

    def base_unit(self, pkey: str) -> Optional[str]:
        return self._pbase.get(pkey)


# ============================================================================ conversion helpers
def _to_canon(t: RawTable, roles: Dict[str, str]) -> pd.DataFrame:
    cols = {}
    for c, r in roles.items():
        if r == "empty":
            continue
        cols[r] = t.df[c].values
    df = pd.DataFrame(cols, dtype=object)
    df["_src"] = t.name
    df["_row"] = np.arange(len(df))
    return df


def _ffill_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Blank order-level cells inherit the value of the order's first row (rulebook §8)."""
    if "order_id" not in df.columns:
        return df
    oid = df["order_id"].map(lambda v: None if is_blank(v) else str(v).strip())
    df["order_id"] = oid.ffill()
    grp = (df["order_id"] != df["order_id"].shift()).cumsum()
    for c in df.columns:
        if c in ORDER_LEVEL and c != "order_id":
            col = df[c].map(lambda v: None if is_blank(v) else v)
            df[c] = col.groupby(grp).transform("first")
    return df


def _parse_cols(df: pd.DataFrame, default_year: Optional[int] = None) -> pd.DataFrame:
    for c in list(df.columns):
        if c in MONEY:
            df[c] = pd.to_numeric(df[c].map(parse_money), errors="coerce")
        elif c in NUMS:
            df[c] = pd.to_numeric(df[c].map(parse_number), errors="coerce")
        elif c in DATES:
            df[c] = df[c].map(lambda v: parse_date(v, default_year))
        elif c in DATETIMES:
            df[c] = df[c].map(lambda v: parse_datetime(v, default_year))
        elif c in IDS:
            df[c] = df[c].map(lambda v: None if is_blank(v) else re.sub(r"\.0$", "", str(v).strip()))
        elif c == "time":
            df[c] = df[c].map(parse_time)
        elif c == "qty":
            pq = df[c].map(parse_qty)
            df["qty"] = pd.to_numeric(pq.map(lambda x: x[0]), errors="coerce")
            df["qty_unit"] = pq.map(lambda x: x[1])
        elif c == "factor":
            pq = df[c].map(parse_qty)
            df["factor"] = pd.to_numeric(pq.map(lambda x: x[0]), errors="coerce")
            df["factor_unit"] = pq.map(lambda x: x[1])
        elif c in ("rx", "active"):
            df[c] = df[c].map(parse_bool)
    return df


def _str_or_none(v):
    return None if is_blank(v) else str(v).strip()


def _readme_cols_for(readme: ReadmeInfo, t: RawTable) -> Dict[str, str]:
    """README column dictionary for a table; section keys may be patterns like
    `BanHang_T<tháng>_<năm>.xlsx`, sheet `DonHang_ChiTiet`."""
    cols = readme.columns
    for key in (t.file, f"{t.file}::{t.sheet}"):
        if key in cols:
            return cols[key]
    best = None
    for k, v in cols.items():
        fpat, _, spat = k.partition("::")
        if "<" not in fpat:
            continue
        rx = "^" + re.sub(r"<[^>]*>", ".+", re.escape(fpat)) + "$"
        try:
            ok = re.match(rx, t.file, re.I) is not None
        except re.error:
            ok = False
        if not ok:
            continue
        if spat and t.sheet and fold_key(spat) != fold_key(t.sheet):
            continue
        best = v
    if best is None and t.sheet:
        for k, v in cols.items():
            if k.endswith("::" + t.sheet):
                best = v
    return best or {}


# ============================================================================ builder
def build_biz(biz_dir: str, biz_id: Optional[str] = None) -> Biz:
    biz_id = biz_id or os.path.basename(os.path.normpath(biz_dir))
    rp = os.path.join(biz_dir, "README.md")
    readme = parse_readme(open(rp, encoding="utf-8").read() if os.path.exists(rp) else "")
    raws = read_business_files(biz_dir)
    log: List[str] = []
    by_kind: Dict[str, List[Tuple[RawTable, Dict[str, str], pd.DataFrame]]] = defaultdict(list)
    raw_typed = []
    order_key_header = None
    m = re.search(r"các dòng cùng \*\*(.+?)\*\* là một (?:đơn|hoá đơn|hóa đơn|phiếu)", readme.text)
    if m:
        order_key_header = fold_key(m.group(1))
    row_is_order = bool(re.search(r"mỗi dòng (?:là )?một (?:lượt|phiếu|đơn|hoá đơn|hóa đơn|lần)", readme.text, re.I)) and \
        not re.search(r"mỗi dòng (?:là )?một (?:mặt hàng|dòng hàng|sản phẩm)", readme.text, re.I)
    for t in raws:
        rc = _readme_cols_for(readme, t)
        kind = classify(t, rc)
        roles = assign_roles(t, kind, rc)
        if order_key_header and kind == "sales":
            for c in t.columns:
                if fold_key(c) == order_key_header:
                    roles = {k: v for k, v in roles.items() if v != "order_id"}
                    roles[c] = "order_id"
        ids = [c for c, r in roles.items() if r in ("order_id", "receipt_id", "product_code", "payment_id")]
        n = drop_total_and_note_rows(t, ids)
        if n:
            log.append(f"{t.name}: dropped {n} total/note rows")
        raw_typed.append((kind, roles, t))
        df = _to_canon(t, roles)
        df["_sheet_year"] = t.sheet_year
        df["_sheet_month"] = t.sheet_month
        if kind == "sales" and "order_id" not in df.columns and row_is_order:
            df["order_id"] = [f"R{len(by_kind['sales'])}_{i:06d}" for i in range(len(df))]
        if kind in ("sales",):
            df = _ffill_groups(df)
        df = _parse_cols(df, t.sheet_year)
        by_kind[kind].append((t, roles, df))
        log.append(f"{t.name}: kind={kind} rows={len(df)} roles={sorted(set(roles.values()))}")

    def cat(kind: str) -> pd.DataFrame:
        parts = [d for _, _, d in by_kind.get(kind, [])]
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True, sort=False)

    lines = cat("sales")
    orders = cat("orders")
    payments = cat("payments")
    products = cat("products")
    units = cat("units")
    stock = cat("stock")
    purchases = cat("purchases")
    plines = cat("purchase_lines")
    lots = cat("lots")
    lot_sales = cat("lot_sales")
    deliveries = cat("deliveries")
    debt_terms = cat("debt_terms")
    prices = cat("price_history")
    expenses = cat("expenses")
    customers = cat("customers")
    drugs = cat("drugs")
    returns = cat("returns")
    tables_ = cat("tables")
    stock_card = cat("stock_card")
    staff = cat("staff")
    suppliers = cat("suppliers")
    shifts = cat("shift_report")

    biz = Biz(biz_id=biz_id, path=biz_dir, readme=readme, lines=lines, orders=orders, payments=payments,
              products=products, unit_factor={}, purchases=purchases, purchase_lines=plines, stock_end=stock,
              lots=lots, lot_sales=lot_sales, deliveries=deliveries, debt_terms=debt_terms, prices=prices,
              expenses=expenses, customers=customers, returns=returns, tables_=tables_, stock_card=stock_card,
              staff=staff, suppliers=suppliers, shifts=shifts, other=[t for t, _, _ in by_kind.get("other", [])],
              raw=raw_typed, log=log)
    biz._cust_name, biz._pname, biz._pcode, biz._pbase, biz._drug = {}, {}, {}, {}, {}
    biz._aliases = {}
    biz._tradein = cat("tradein")
    al = cat("aliases")
    if not al.empty and "alias" in al.columns and "product_name" in al.columns:
        for a_, n_, g_ in zip(al["alias"], al["product_name"], al.get("product_group", [None] * len(al))):
            if is_blank(n_):
                continue
            g_ = None if is_blank(g_) else str(g_).strip()
            biz._aliases[fold_key(str(n_))] = (str(n_).strip(), g_)
            if not is_blank(a_):
                biz._aliases[fold_key(str(a_))] = (str(n_).strip(), g_)
        if biz.products.empty:
            uniq = {}
            for n_, g_ in biz._aliases.values():
                uniq.setdefault(n_, g_)
            biz.products = pd.DataFrame({"product_name": list(uniq), "product_group": list(uniq.values())})
    _build_products(biz, drugs, units)
    _build_lines_orders(biz)
    _build_payments(biz)
    _build_purchases(biz)
    _build_misc(biz)
    return biz


# ---------------------------------------------------------------------------- products & units
def _build_products(biz: Biz, drugs: pd.DataFrame, units: pd.DataFrame):
    names_by_key: Dict[str, Counter] = defaultdict(Counter)
    code_by_namekey: Dict[str, str] = {}
    prod = biz.products
    rows = []
    if not prod.empty:
        for _, r in prod.iterrows():
            code = _str_or_none(r.get("product_code"))
            name = _str_or_none(r.get("product_name"))
            if not code and not name:
                continue
            key = code if code else "N:" + fold_key(name)
            if name:
                names_by_key[key][name] += 3
                code_by_namekey.setdefault(fold_key(name), key)
            rows.append({"key": key, "code": code, "name": name,
                         "group": _str_or_none(r.get("product_group")),
                         "base_unit": _str_or_none(r.get("base_unit")),
                         "price": r.get("price"), "cost": r.get("cost"),
                         "stock_qty": r.get("stock_qty"), "brand": _str_or_none(r.get("brand")),
                         "size": _str_or_none(r.get("size")), "aliases": _str_or_none(r.get("aliases")),
                         "active": r.get("active"), "conversion": _str_or_none(r.get("conversion")),
                         "product_type": _str_or_none(r.get("product_type"))})
    cat = pd.DataFrame(rows)
    if not cat.empty:
        cat = cat.drop_duplicates("key", keep="first")
    biz._code_by_namekey = code_by_namekey
    biz._names_by_key = names_by_key
    # drugs
    biz._drug = {}
    if not drugs.empty:
        for _, r in drugs.iterrows():
            code = _str_or_none(r.get("product_code"))
            if code:
                biz._drug[code] = (_str_or_none(r.get("active_ingredient")), r.get("rx"))
    biz.products = cat
    biz._units_raw = units


def _resolve_pkey(biz: Biz, code, name) -> Optional[str]:
    code = _str_or_none(code)
    if code:
        return code
    name = _str_or_none(name)
    if not name:
        return None
    k = fold_key(name)
    return biz._code_by_namekey.get(k) or ("N:" + k)


def _build_unit_factors(biz: Biz):
    uf: Dict[Tuple[str, str], float] = {}
    chain: Dict[Tuple[str, str], Tuple[float, str]] = {}
    base_of: Dict[str, str] = {}
    cat = biz.products
    if not cat.empty:
        for _, r in cat.iterrows():
            if r.get("base_unit"):
                base_of[r["key"]] = r["base_unit"]
                uf[(r["key"], fold_key(r["base_unit"]))] = 1.0
            conv = r.get("conversion")
            if isinstance(conv, str) and conv:
                for m in re.finditer(r"1\s+([^=,;]+?)\s*=\s*([0-9.,]+)\s*([^,;]+)", conv):
                    try:
                        f = float(m.group(2).replace(",", "."))
                    except ValueError:
                        continue
                    chain[(r["key"], fold_key(m.group(1)))] = (f, fold_key(m.group(3)))
    u = getattr(biz, "_units_raw", pd.DataFrame())
    if not u.empty:
        for _, r in u.iterrows():
            key = _resolve_pkey(biz, r.get("product_code"), r.get("product_name"))
            big = _str_or_none(r.get("big_unit"))
            f = r.get("factor")
            if not key or not big or f is None or (isinstance(f, float) and np.isnan(f)):
                continue
            fu = r.get("factor_unit")
            bu = _str_or_none(r.get("base_unit"))
            if bu:
                base_of.setdefault(key, bu)
                uf[(key, fold_key(bu))] = 1.0
            if fu and not is_blank(fu):
                chain[(key, fold_key(big))] = (float(f), fold_key(fu))
            elif biz.readme.factor_is_chain:
                chain[(key, fold_key(big))] = (float(f), None)  # relative to the unit just below; resolved later
            else:
                uf[(key, fold_key(big))] = float(f)
    for (pname, big, f, bu) in biz.readme.conversions:
        key = _resolve_pkey(biz, None, pname)
        uf[(key, fold_key(big))] = f
        uf.setdefault((key, fold_key(bu)), 1.0)
        base_of.setdefault(key, bu)
    for (pname, big, f, small) in biz.readme.bundle_rules:
        key = _resolve_pkey(biz, None, pname)
        chain[(key, fold_key(big))] = (f, fold_key(small))
    # resolve chains: factor(big) = f * factor(small)
    for _ in range(4):
        for (key, big), (f, small) in list(chain.items()):
            if small is None:
                # 'ngay duoi': the largest known factor below
                below = [v for (k, uu), v in uf.items() if k == key]
                if below:
                    uf[(key, big)] = f * max(below)
                    chain.pop((key, big))
                continue
            sf = uf.get((key, small))
            if sf is None and base_of.get(key) and fold_key(base_of[key]) == small:
                sf = 1.0
            if sf is not None:
                uf[(key, big)] = f * sf
                chain.pop((key, big))
    for (key, big), (f, small) in chain.items():
        uf.setdefault((key, big), f)
    biz.unit_factor = uf
    biz._pbase = base_of


# ---------------------------------------------------------------------------- lines & orders
_SPECIAL_LINE = [
    (re.compile(r"^phi phuc vu"), "service_fee"),
    (re.compile(r"^voucher|^ma giam|^giam gia hoa don"), "voucher"),
    (re.compile(r"^dung\s*\d*\s*diem|^tru diem|^doi diem"), "points"),
    (re.compile(r"^phi ship|^phi giao|^ship\b|^phi van chuyen"), "shipping"),
]


def _line_kind(name) -> str:
    if is_blank(name):
        return "item"
    f = fold(str(name))
    for rx, k in _SPECIAL_LINE:
        if rx.search(f):
            return k
    return "item"


_TOP_RE = re.compile(r"^\s*(?:(\d+)\s+)?(.+?)\s*\(\s*([0-9.,]+\s*(?:đ|d|k|vnd|dong|đồng)?)\s*\)\s*$", re.I)


def _split_toppings(biz: Biz, L: pd.DataFrame) -> pd.DataFrame:
    known = {}
    for k, cnt in biz._names_by_key.items():
        for n in cnt:
            known[fold_key(n)] = n
    rows, extra = [], []
    idx = L.index[L["product_name"].map(lambda v: isinstance(v, str) and " + " in v)]
    for i in idx:
        name = L.at[i, "product_name"]
        parts = [p.strip() for p in name.split(" + ")]
        base, tops = parts[0], []
        ok = True
        for p in parts[1:]:
            m = _TOP_RE.match(p)
            if not m:
                ok = False
                break
            n = int(m.group(1)) if m.group(1) else 1
            tname = m.group(2).strip()
            money = parse_money(m.group(3))
            if money is None:
                ok = False
                break
            tops.append((n, tname, money))
        if not ok or not tops:
            continue
        L.at[i, "product_name"] = base
        tot = sum(t[2] for t in tops)
        amt = L.at[i, "amount"] if "amount" in L.columns else None
        if amt is not None and not (isinstance(amt, float) and np.isnan(amt)):
            L.at[i, "amount"] = float(amt) - tot
        for n, tname, money in tops:
            full = known.get(fold_key("Topping " + tname)) or known.get(fold_key(tname)) or ("Topping " + tname)
            r = L.loc[i].copy()
            r["product_name"] = full
            r["product_code"] = None if "product_code" in L.columns else None
            r["qty"] = float(n)
            r["qty_unit"] = None
            r["base_qty"] = float(n)
            r["unit"] = "phần"
            r["unit_price"] = money / n
            r["amount"] = money
            r["line_discount"] = 0.0 if "line_discount" in L.columns else None
            r["_topping"] = True
            extra.append(r)
    if extra:
        L = pd.concat([L, pd.DataFrame(extra)], ignore_index=True, sort=False)
        biz.log.append(f"split {len(extra)} toppings from combined cells")
    return L


def _build_lines_orders(biz: Biz):
    L = biz.lines.copy() if not biz.lines.empty else pd.DataFrame()
    O = biz.orders.copy() if not biz.orders.empty else pd.DataFrame()
    has_lines = not L.empty
    # ---------------------------------------------------------------- orders table normalise
    if not O.empty:
        O["order_id"] = O["order_id"].map(_str_or_none)
        O = O[O["order_id"].notna()].copy()
        if "date" not in O.columns and "datetime" in O.columns:
            O["date"] = O["datetime"].map(lambda x: x.date() if x else None)
        O = O.drop_duplicates("order_id", keep="first")
    if has_lines:
        if "order_id" in L.columns:
            L["order_id"] = L["order_id"].map(_str_or_none)
        else:
            L["order_id"] = None
        # dates: line date, else sheet date, else order table
        if "date" not in L.columns:
            L["date"] = None
        if "datetime" in L.columns:
            L["date"] = [d if d is not None else (x.date() if x is not None else None) for d, x in zip(L["date"], L["datetime"])]
        if not O.empty and L["date"].isna().any():
            od = dict(zip(O["order_id"], O["date"]))
            L["date"] = [d if d is not None else od.get(o) for d, o in zip(L["date"], L["order_id"])]
        # abbreviations sheet ("Ghi tắt": short name -> full name, type)
        amap = getattr(biz, "_aliases", {}) or {}
        if amap and "product_name" in L.columns:
            full = [amap.get(fold_key(str(n)), (None, None)) if not is_blank(n) else (None, None) for n in L["product_name"]]
            L["product_name"] = [f[0] if f[0] else n for f, n in zip(full, L["product_name"])]
            grp = [f[1] for f in full]
            if "product_group" not in L.columns:
                L["product_group"] = grp
            else:
                L["product_group"] = [g if not is_blank(g) else x for g, x in zip(L["product_group"], grp)]
        # toppings written in the same cell as the drink: "Latte (M) + 2 Thạch cà phê (14.000đ)"
        if "product_name" in L.columns and L["product_name"].map(lambda v: isinstance(v, str) and " + " in v).any():
            L = _split_toppings(biz, L)
        # service-ticket books (vehicle / work / mechanic columns)
        if "vehicle" in L.columns:
            from ..domains import repair
            L = repair.enrich(biz, L)
        # line kind
        L["kind"] = L["product_name"].map(_line_kind) if "product_name" in L.columns else "item"
        # product key
        codes = L["product_code"] if "product_code" in L.columns else pd.Series([None] * len(L))
        names = L["product_name"] if "product_name" in L.columns else pd.Series([None] * len(L))
        L["pkey"] = [(_resolve_pkey(biz, c, n) if k == "item" or k == "service_fee" else None)
                     for c, n, k in zip(codes, names, L["kind"])]
        for k, n in zip(L["pkey"], names):
            if k and not is_blank(n):
                biz._names_by_key[k][str(n).strip()] += 1
        # lines that carry only a code: classify by the catalogue name ('Phí phục vụ' is not an item)
        special = {k for k, cnt in biz._names_by_key.items() if any(_line_kind(n) != "item" for n in cnt)}
        if special:
            kinds = []
            for k, kd in zip(L["pkey"], L["kind"]):
                if kd == "item" and k in special:
                    nm = max(biz._names_by_key[k].items(), key=lambda kv: kv[1])[0]
                    kinds.append(_line_kind(nm))
                else:
                    kinds.append(kd)
            L["kind"] = kinds
    _build_unit_factors(biz)
    if has_lines:
        # quantities
        if "qty" not in L.columns:
            L["qty"] = np.nan
            L["qty_unit"] = None
        unit_col = L["unit"] if "unit" in L.columns else pd.Series([None] * len(L))
        L["unit_eff"] = [(_str_or_none(qu) or _str_or_none(u)) for qu, u in zip(L.get("qty_unit", [None] * len(L)), unit_col)]
        if "base_qty" not in L.columns:
            L["base_qty"] = np.nan
        # infer factors from lines having both qty and base qty
        inferred = defaultdict(Counter)
        for k, u, q, b in zip(L["pkey"], L["unit_eff"], L["qty"], L["base_qty"]):
            if k and u and q and b and not np.isnan(q) and not np.isnan(b) and q != 0:
                inferred[(k, fold_key(u))][round(b / q, 6)] += 1
        for kk, cnt in inferred.items():
            f, c = cnt.most_common(1)[0]
            if kk not in biz.unit_factor and c >= 2:
                biz.unit_factor[kk] = f
        bq = []
        for k, u, q, b in zip(L["pkey"], L["unit_eff"], L["qty"], L["base_qty"]):
            if b is not None and not (isinstance(b, float) and np.isnan(b)):
                bq.append(float(b))
                continue
            if q is None or (isinstance(q, float) and np.isnan(q)):
                bq.append(np.nan)
                continue
            f = biz.unit_factor.get((k, fold_key(u))) if (k and u) else None
            bq.append(float(q) * (f if f else 1.0))
        L["base_qty"] = bq
        # money
        for c in ("amount", "unit_price", "line_discount", "cost_unit", "qty", "base_qty"):
            if c not in L.columns:
                L[c] = np.nan
            L[c] = pd.to_numeric(L[c], errors="coerce").astype(float)
        miss = (L["amount"].isna() & L["qty"].notna() & L["unit_price"].notna()).to_numpy()
        if miss.any():
            amt = L["amount"].to_numpy(copy=True)
            amt[miss] = (L["qty"].to_numpy()[miss] * L["unit_price"].to_numpy()[miss]
                         - np.nan_to_num(L["line_discount"].to_numpy()[miss]))
            L["amount"] = amt
        # time of order
        if "datetime" not in L.columns:
            L["datetime"] = None
        if "time" in L.columns:
            L["datetime"] = [x if x is not None else (dt.datetime.combine(d, t) if (d is not None and t is not None) else None)
                             for x, d, t in zip(L["datetime"], L["date"], L["time"])]
        L["status_n"] = L["status"].map(norm_status) if "status" in L.columns else None
        L["neg"] = (L["qty"] < 0) | (L["amount"] < 0) & (L["kind"] == "item")
        # duplicated export rows of the same order (rulebook §8): count once
        if L["order_id"].notna().any():
            keycols = [c for c in ("order_id", "line_no", "pkey", "product_name", "qty", "unit_eff", "unit_price", "amount",
                                   "line_discount", "base_qty", "date", "status", "datetime") if c in L.columns]
            sub = L[keycols].astype(str)
            dup = sub.duplicated(keep="first") & L["order_id"].notna()
            if dup.any():
                biz.log.append(f"dropped {int(dup.sum())} duplicated line rows")
                L = L.loc[~dup].copy()
        L = L.reset_index(drop=True)

    # ---------------------------------------------------------------- derive / merge orders
    if O.empty and has_lines and L["order_id"].notna().any():
        agg_cols = [c for c in ORDER_LEVEL if c in L.columns and c != "order_id"]
        agg_cols += [c for c in ("plate", "model_key", "model_raw", "warranty", "work") if c in L.columns]
        first = L[L["order_id"].notna()].groupby("order_id", sort=False)[agg_cols + ["datetime"] if "datetime" not in agg_cols else agg_cols].first()
        O = first.reset_index()
        O["_derived"] = True
    elif not O.empty and has_lines:
        # bring line-level order fields that the order table lacks (e.g. customer name in the line export)
        extra = [c for c in ORDER_LEVEL if c in L.columns and c not in O.columns and c != "order_id"]
        if extra:
            f = L[L["order_id"].notna()].groupby("order_id", sort=False)[extra].first().reset_index()
            O = O.merge(f, on="order_id", how="left")
    if not O.empty:
        if "datetime" not in O.columns:
            O["datetime"] = None
        if "time" in O.columns:
            O["datetime"] = [x if x is not None else (dt.datetime.combine(d, t) if (d is not None and t is not None) else None)
                             for x, d, t in zip(O["datetime"], O["date"], O["time"])]
        # status
        O["status_n"] = O["status"].map(norm_status) if "status" in O.columns else None
        O["has_return"] = False
        if has_lines:
            Lk = L[L["order_id"].notna()]
            g = Lk.groupby("order_id")
            neg = g["neg"].any()
            O["has_return"] = O["order_id"].map(neg).fillna(False).astype(bool)
            if "status_n" in L.columns:
                flags = pd.DataFrame({"order_id": Lk["order_id"],
                                      "c": Lk["status_n"].eq("cancelled"), "r": Lk["status_n"].eq("returned"),
                                      "s": Lk["status_n"].eq("shipping"), "a": Lk["status_n"].notna()})
                fg = flags.groupby("order_id")[["c", "r", "s", "a"]].any()
                fc, fr, fs, fa = (fg[c].to_dict() for c in ("c", "r", "s", "a"))
                derived = "_derived" in O.columns
                def comb(oid, st):
                    if st and not derived:
                        return st
                    if fc.get(oid):
                        return "cancelled"
                    if fr.get(oid):
                        return "returned"
                    if fs.get(oid):
                        return "shipping"
                    return "completed" if fa.get(oid) else None
                O["status_n"] = [comb(o, s) for o, s in zip(O["order_id"], O["status_n"])]
                ret = {o for o, v in fr.items() if v}
                O.loc[O["order_id"].isin(ret), "has_return"] = True
        # status written in the note column (shops without a status column)
        if O["status_n"].isna().all() and "note" in O.columns:
            O["status_n"] = O["note"].map(lambda v: norm_status(v) if (not is_blank(v) and re.search(
                r"huy|tra|dang giao|cancel|return|ship", fold(str(v)))) else "completed")
        O["status_n"] = O["status_n"].fillna("completed")
        if not biz.lot_sales.empty and "qty_returned" in biz.lot_sales.columns:
            ls = biz.lot_sales
            r = set(ls.loc[ls["qty_returned"].fillna(0) > 0, "order_id"].dropna())
            O.loc[O["order_id"].isin(r), "has_return"] = True
        if not biz.returns.empty and "order_id" in biz.returns.columns:
            r = set(biz.returns["order_id"].dropna())
            O.loc[O["order_id"].isin(r), "has_return"] = True
        O.loc[O["status_n"] == "returned", "has_return"] = True
        st = []
        for s, hr in zip(O["status_n"], O["has_return"]):
            if s == "cancelled":
                st.append("cancelled")
            elif hr or s == "returned":
                st.append("returned")
            else:
                st.append(s)
        O["status_final"] = st
        O["valid"] = O["status_final"].isin(["completed", "shipping"])
        # customers
        _customers(biz, O)
        if "payment_method" in O.columns:
            O["pay_method_n"] = O["payment_method"].map(norm_payment)
        else:
            O["pay_method_n"] = None
        O["channel_n"] = O["channel"].map(norm_channel) if "channel" in O.columns else None
        if "date" in O.columns:
            O["date"] = O["date"].map(lambda x: x if (x is None or isinstance(x, dt.date)) else None)
    if has_lines and not O.empty:
        m = O.set_index("order_id")
        for c in ("status_final", "valid", "cust_key", "date"):
            if c == "date":
                continue
            L[c] = L["order_id"].map(m[c]) if c in m.columns else None
        # lines whose order is missing from the order table
        L["valid"] = L["valid"].fillna(False).astype(bool)
    elif has_lines:
        # no order ids at all: every row stands alone ("bảng không ghi theo đơn")
        L["status_final"] = L["status_n"].fillna("completed") if "status_n" in L.columns else "completed"
        L.loc[L["neg"], "status_final"] = "returned"
        L["valid"] = L["status_final"].isin(["completed", "shipping"])
        L["cust_key"] = None
    # revenue per order from lines
    if not O.empty:
        if has_lines and L["order_id"].notna().any():
            rv = L[(L["kind"].isin(["item", "service_fee"])) & (~L["neg"])].groupby("order_id")["amount"].sum()
            O["revenue"] = O["order_id"].map(rv).fillna(0.0)
            O["has_lines"] = O["order_id"].isin(set(L["order_id"].dropna()))
            sp = L[L["kind"] == "shipping"].groupby("order_id")["amount"].sum()
            vc = L[L["kind"].isin(["voucher", "points"])].groupby("order_id")["amount"].sum()
            if "shipping_fee" not in O.columns and len(sp):
                O["shipping_fee"] = O["order_id"].map(sp).fillna(0.0)
            if "order_discount" not in O.columns and len(vc):
                O["order_discount"] = -O["order_id"].map(vc).fillna(0.0)
        else:
            O["revenue"] = O["goods_total"] if "goods_total" in O.columns else np.nan
            O["has_lines"] = False
        if has_lines and "product_group" in L.columns:
            from ..domains import phone
            if phone.is_phone_shop(biz, L):
                biz.flags["phone"] = True
                O = phone.enrich_orders(biz, O, getattr(biz, "_tradein", None))
        # amount the customer must pay for the whole order (debt base)
        due = None
        if "order_total" in O.columns and O["order_total"].notna().any():
            due = O["order_total"]
        else:
            base = O["goods_total"] if "goods_total" in O.columns and O["goods_total"].notna().any() else O["revenue"]
            due = base.fillna(0) - (O["order_discount"].fillna(0) if "order_discount" in O.columns else 0) + \
                (O["shipping_fee"].fillna(0) if "shipping_fee" in O.columns else 0) + \
                (O["vat"].fillna(0) if "vat" in O.columns else 0) - \
                (O["trade_in"].fillna(0) if "trade_in" in O.columns else 0)
        O["due"] = due
        if "datetime" in O.columns:
            O["hour"] = O["datetime"].map(lambda x: x.hour if x is not None and not (isinstance(x, float)) else None)
    biz.lines = L
    biz.orders = O
    # names
    biz._pname = {}
    for k, cnt in biz._names_by_key.items():
        # prefer a properly accented spelling, then the most frequent one
        best = sorted(cnt.items(), key=lambda kv: (-(fold(kv[0]) != kv[0].lower()), -(kv[0] != kv[0].upper()), -kv[1]))
        biz._pname[k] = best[0][0]
    biz._pcode = {}
    if not biz.products.empty:
        for k, c in zip(biz.products["key"], biz.products["code"]):
            if c:
                biz._pcode[k] = c
    for k in list(biz._pname):
        if not k.startswith("N:"):
            biz._pcode.setdefault(k, k)
    # dates range
    ds = []
    if not O.empty and "date" in O.columns:
        ds = [d for d in O["date"] if isinstance(d, dt.date)]
    elif has_lines:
        ds = [d for d in L["date"] if isinstance(d, dt.date)]
    biz.data_start = min(ds) if ds else None
    biz.data_end = biz.readme.data_end or (max(ds) if ds else None)


def _customers(biz: Biz, O: pd.DataFrame):
    has_id = "customer_id" in O.columns and O["customer_id"].notna().any()
    biz.flags["customer_ids"] = bool(has_id)
    cust_tab = biz.customers
    id2name, id2group = {}, {}
    if not cust_tab.empty and "customer_id" in cust_tab.columns:
        for _, r in cust_tab.iterrows():
            cid = _str_or_none(r.get("customer_id"))
            if not cid:
                continue
            if not is_blank(r.get("customer_name")):
                id2name[cid] = str(r.get("customer_name")).strip()
            if not is_blank(r.get("customer_group")):
                id2group[cid] = str(r.get("customer_group")).strip()
    biz._cust_name = {}
    if has_id:
        keys = O["customer_id"].map(_str_or_none)
        O["cust_key"] = keys
        names = O["customer_name"] if "customer_name" in O.columns else pd.Series([None] * len(O))
        for k, n in zip(keys, names):
            if k and k not in biz._cust_name:
                biz._cust_name[k] = id2name.get(k) or (str(n).strip() if not is_blank(n) else k)
        for k, n in id2name.items():
            biz._cust_name.setdefault(k, n)
        if "customer_group" in O.columns and O["customer_group"].notna().any():
            O["cust_group"] = O["customer_group"]
        else:
            O["cust_group"] = keys.map(lambda k: id2group.get(k) if k else None)
    elif "customer_name" in O.columns:
        def key(n):
            if customer_is_walkin(n):
                return None
            return "C:" + fold_key(str(n))
        O["cust_key"] = O["customer_name"].map(key)
        cnt = defaultdict(Counter)
        for k, n in zip(O["cust_key"], O["customer_name"]):
            if k:
                cnt[k][str(n).strip()] += 1
        for k, c in cnt.items():
            best = sorted(c.items(), key=lambda kv: (-(fold(kv[0]) != kv[0].lower()), -kv[1]))
            biz._cust_name[k] = best[0][0]
        # group by name from the customer table
        n2g = {}
        if not cust_tab.empty and "customer_name" in cust_tab.columns and "customer_group" in cust_tab.columns:
            for n, g in zip(cust_tab["customer_name"], cust_tab["customer_group"]):
                if not is_blank(n) and not is_blank(g):
                    n2g["C:" + fold_key(str(n))] = str(g).strip()
        if "customer_group" in O.columns and O["customer_group"].notna().any():
            O["cust_group"] = O["customer_group"]
        else:
            O["cust_group"] = O["cust_key"].map(lambda k: n2g.get(k) if k else None)
    else:
        O["cust_key"] = None
        O["cust_group"] = None
    O.loc[O["cust_key"].isna(), "cust_group"] = None


# ---------------------------------------------------------------------------- payments
def _build_payments(biz: Biz):
    P = biz.payments
    if P.empty:
        return
    P = P.copy()
    if "pay_datetime" not in P.columns:
        for c in ("datetime", "date"):
            if c in P.columns:
                P["pay_datetime"] = P[c].map(lambda x: x if isinstance(x, dt.datetime) else
                                             (dt.datetime.combine(x, dt.time()) if isinstance(x, dt.date) else None))
                break
    P["pay_date"] = P["pay_datetime"].map(lambda x: x.date() if x is not None else None)
    if "pay_amount" not in P.columns:
        for c in ("amount", "paid", "total"):
            if c in P.columns:
                P["pay_amount"] = P[c]
                break
    P["method_n"] = P["payment_method"].map(norm_payment) if "payment_method" in P.columns else None
    P = P[P["order_id"].notna()]
    biz.payments = P.reset_index(drop=True)


# ---------------------------------------------------------------------------- purchases
def _build_purchases(biz: Biz):
    H, PL = biz.purchases, biz.purchase_lines
    if not PL.empty:
        PL = PL.copy()
        PL["pkey"] = [_resolve_pkey(biz, c, n) for c, n in zip(PL.get("product_code", [None] * len(PL)),
                                                             PL.get("product_name", [None] * len(PL)))]
        if "qty_unit" in PL.columns and "unit" in PL.columns:
            PL["unit"] = [u if not is_blank(u) else qu for u, qu in zip(PL["unit"], PL["qty_unit"])]
        if "amount" not in PL.columns or PL["amount"].isna().all():
            if "qty" in PL.columns and "unit_cost" in PL.columns:
                PL["amount"] = PL["qty"] * PL["unit_cost"]
        if not H.empty:
            hm = H.drop_duplicates("receipt_id").set_index("receipt_id")
            for c in ("date", "supplier_id", "supplier_name", "status", "note", "total"):
                if c in hm.columns and (c not in PL.columns or PL[c].isna().all()):
                    PL[c] = PL["receipt_id"].map(hm[c])
        PL["base_qty"] = [q * (biz.unit_factor.get((k, fold_key(u)), 1.0) if (k and not is_blank(u)) else 1.0)
                          if q is not None and not np.isnan(q) else np.nan
                          for q, k, u in zip(PL.get("qty", [np.nan] * len(PL)), PL["pkey"],
                                             PL.get("unit", [None] * len(PL)))]
        if "qty_base" in PL.columns:
            PL["base_qty"] = PL["qty_base"].where(PL["qty_base"].notna(), PL["base_qty"])
        if "unit_cost" in PL.columns:
            PL["unit_cost_base"] = [c / (biz.unit_factor.get((k, fold_key(u)), 1.0) if (k and isinstance(u, str)) else 1.0)
                                    if c is not None and not (isinstance(c, float) and np.isnan(c)) else np.nan
                                    for c, k, u in zip(PL["unit_cost"], PL["pkey"], PL.get("unit", [None] * len(PL)))]
        biz.purchase_lines = PL
        # lots recorded on purchase lines (no separate lot list)
        if biz.lots.empty and "lot" in PL.columns and PL["lot"].notna().any():
            cols = {"receipt_id": PL["receipt_id"], "pkey": PL["pkey"], "lot": PL["lot"],
                    "expiry": PL["expiry"] if "expiry" in PL.columns else None, "qty_base": PL["base_qty"],
                    "unit_cost": PL["unit_cost_base"] if "unit_cost_base" in PL.columns else np.nan}
            biz.lots = pd.DataFrame(cols)
        elif not biz.lots.empty and "lot" in PL.columns and "unit_cost_base" in PL.columns:
            lc = dict(zip(PL["lot"], PL["unit_cost_base"]))
            if "unit_cost" not in biz.lots.columns or biz.lots["unit_cost"].isna().all():
                biz.lots = biz.lots.assign(unit_cost=biz.lots["lot"].map(lc))
    if H.empty and not PL.empty:
        agg = {}
        for c in ("date", "supplier_id", "supplier_name", "status", "note", "total"):
            if c in PL.columns:
                agg[c] = "first"
        H = PL.groupby("receipt_id", sort=False).agg(agg).reset_index()
        if "total" not in H.columns or H["total"].isna().all():
            H["total"] = H["receipt_id"].map(PL.groupby("receipt_id")["amount"].sum())
    if not H.empty:
        H = H.copy()
        if "total" not in H.columns or H["total"].isna().all():
            if not PL.empty and "amount" in PL.columns:
                H["total"] = H["receipt_id"].map(PL.groupby("receipt_id")["amount"].sum())
        H["cancelled"] = H["status"].map(lambda v: bool(re.search(r"\bhuy\b|cancel", fold(str(v)))) if not is_blank(v) else False) \
            if "status" in H.columns else False
        if not biz.suppliers.empty and "supplier_id" in biz.suppliers.columns and "supplier_id" in H.columns:
            sm = dict(zip(biz.suppliers["supplier_id"], biz.suppliers.get("supplier_name", biz.suppliers["supplier_id"])))
            if "supplier_name" not in H.columns:
                H["supplier_name"] = H["supplier_id"].map(sm)
        biz.purchases = H


# ---------------------------------------------------------------------------- misc
def _build_misc(biz: Biz):
    if not biz.stock_end.empty:
        S = biz.stock_end.copy()
        S["pkey"] = [_resolve_pkey(biz, c, n) for c, n in zip(S.get("product_code", [None] * len(S)),
                                                            S.get("product_name", [None] * len(S)))]
        biz.stock_end = S
    elif not biz.products.empty and biz.products["stock_qty"].notna().any():
        biz.stock_end = pd.DataFrame({"pkey": biz.products["key"], "stock_qty": biz.products["stock_qty"]})
    if not biz.prices.empty:
        P = biz.prices.copy()
        P["pkey"] = [_resolve_pkey(biz, c, n) for c, n in zip(P.get("product_code", [None] * len(P)),
                                                            P.get("product_name", [None] * len(P)))]
        biz.prices = P
    if not biz.lots.empty:
        Lt = biz.lots.copy()
        if "product_code" in Lt.columns:
            Lt["pkey"] = Lt["product_code"]
        biz.lots = Lt
    if not biz.deliveries.empty:
        D = biz.deliveries.copy()
        D["pkey"] = [_resolve_pkey(biz, c, n) for c, n in zip(D.get("product_code", [None] * len(D)),
                                                            D.get("product_name", [None] * len(D)))]
        biz.deliveries = D


BUILDER_VERSION = "b9"


def load_biz(biz_dir: str, biz_id: Optional[str] = None) -> Biz:
    """build_biz with an optional pickle cache (env DATAMATE_CACHE)."""
    from .reader import _cache_path
    cp = _cache_path(biz_dir)
    if cp:
        cp = cp.replace("raw_", f"biz{BUILDER_VERSION}_")
        if os.path.exists(cp):
            import pickle
            try:
                with open(cp, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                pass
    b = build_biz(biz_dir, biz_id)
    if cp:
        import pickle
        try:
            with open(cp, "wb") as fh:
                pickle.dump(b, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return b
