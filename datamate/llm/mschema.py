"""Canonical SQL views + M-Schema rendering (XiYan-SQL's schema representation).

The views already encode the rulebook's data semantics (valid orders, base units, line
kinds, customer identity), so the SQL generator only needs to express the question.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..ingest.builder import Biz

VIEW_DOCS = {
    "orders": ("một dòng một đơn bán", {
        "order_id": "mã đơn (khoá chính)",
        "order_date": "ngày đặt đơn (đơn thuộc kỳ theo ngày này)",
        "order_hour": "giờ tạo đơn 0-23 (ca sáng 6-13, ca chiều 14-21)",
        "status": "completed / cancelled / returned (có trả hàng, kể cả một phần) / shipping",
        "is_valid": "1 nếu đơn được tính doanh thu và số đơn (không huỷ, không trả)",
        "customer_key": "mã định danh khách (NULL = khách lẻ, không tính là khách)",
        "customer_name": "tên khách",
        "customer_group": "nhóm khách",
        "staff": "mã nhân viên bán",
        "channel": "kênh bán: store / phone / zalo / ecommerce / facebook ...",
        "payment_method": "cash / transfer / ewallet / cod / debt (ghi nợ)",
        "revenue": "doanh thu của đơn = tổng thành tiền dòng hàng (đã trừ giảm giá dòng; không trừ giảm giá đơn, không VAT, không phí giao)",
        "total_due": "số khách phải trả cả đơn (sau giảm giá đơn, gồm phí giao)",
    }),
    "sales_lines": ("một dòng một mặt hàng trong đơn (đã bỏ dòng âm/trả, dòng trùng lặp)", {
        "order_id": "mã đơn", "line_no": "số thứ tự dòng trong đơn", "order_date": "ngày đặt đơn", "product_key": "khoá mặt hàng",
        "product_name": "tên mặt hàng", "product_group": "nhóm hàng", "kind": "item / service_fee / voucher / points / shipping",
        "unit": "đơn vị tính của dòng", "qty": "số lượng theo đơn vị của dòng", "base_qty": "số lượng quy về đơn vị cơ bản",
        "base_unit": "đơn vị cơ bản của mặt hàng", "unit_price": "đơn giá theo đơn vị của dòng, trước chiết khấu",
        "amount": "thành tiền dòng (đã trừ giảm giá dòng)", "cost_per_unit": "giá vốn một đơn vị bán ghi trên dòng",
        "is_valid": "1 nếu đơn của dòng hợp lệ (tính doanh thu)",
    }),
    "products": ("danh mục hàng", {
        "product_key": "khoá", "code": "mã hàng", "name": "tên", "product_group": "nhóm hàng", "base_unit": "đơn vị cơ bản",
        "price": "giá bán hiện tại", "cost": "giá vốn một đơn vị cơ bản (danh mục)", "active_ingredient": "hoạt chất",
        "is_prescription": "1 nếu thuốc kê đơn",
    }),
    "payments": ("một dòng một lần khách trả tiền cho một đơn", {
        "order_id": "mã đơn", "pay_no": "lần trả thứ mấy của đơn", "pay_date": "ngày trả", "amount": "số tiền trả", "method": "hình thức",
    }),
    "purchases": ("một dòng một phiếu nhập", {
        "receipt_id": "mã phiếu", "receipt_date": "ngày phiếu", "supplier": "nhà cung cấp", "total": "tổng tiền phiếu",
        "cancelled": "1 nếu phiếu huỷ",
    }),
    "purchase_lines": ("một dòng một mặt hàng trong phiếu nhập", {
        "receipt_id": "mã phiếu", "line_no": "số thứ tự dòng trong phiếu", "receipt_date": "ngày phiếu", "product_key": "khoá mặt hàng", "unit": "đơn vị nhập",
        "qty": "số lượng theo đơn vị nhập", "base_qty": "số lượng quy về đơn vị cơ bản", "unit_cost": "giá nhập một đơn vị nhập",
        "amount": "tiền nhập của dòng", "lot": "số lô", "expiry": "hạn dùng",
    }),
    "expenses": ("sổ chi phí", {"expense_date": "ngày", "amount": "số tiền", "category": "nhóm chi"}),
    "deliveries": ("sổ giao hàng: một dòng một mặt hàng trong một đợt giao",
                   {"order_id": "mã đơn", "trip_no": "đợt giao", "delivery_date": "ngày giao", "product_key": "khoá", "base_qty": "số lượng giao"}),
}


def _d(x):
    return x if isinstance(x, dt.date) else None


def build_views(b: Biz) -> Dict[str, pd.DataFrame]:
    V: Dict[str, pd.DataFrame] = {}
    O = b.orders
    if not O.empty:
        V["orders"] = pd.DataFrame({
            "order_id": O["order_id"], "order_date": O["date"].map(_d),
            "order_hour": O["hour"] if "hour" in O.columns else None,
            "status": O["status_final"], "is_valid": O["valid"].astype(int),
            "customer_key": O.get("cust_key"), "customer_name": O.get("customer_name"),
            "customer_group": O.get("cust_group"), "staff": O.get("staff"), "channel": O.get("channel_n"),
            "payment_method": O.get("pay_method_n"), "revenue": O.get("revenue"), "total_due": O.get("due"),
        })
    L = b.lines
    if not L.empty:
        # product group: catalogue column, else the group written on the sales lines (as the executor does)
        pg = {}
        if not b.products.empty and "group" in b.products.columns:
            pg = {k: g for k, g in zip(b.products["key"], b.products["group"]) if isinstance(g, str) and g.strip()}
        if not pg and "product_group" in L.columns:
            for k, g in zip(L["pkey"], L["product_group"]):
                if k and isinstance(g, str) and g.strip():
                    pg.setdefault(k, g.strip())
        sub = L[~L["neg"]]
        V["sales_lines"] = pd.DataFrame({
            "order_id": sub["order_id"], "line_no": sub.groupby("order_id", dropna=False).cumcount() + 1,
            "order_date": sub["date"].map(_d), "product_key": sub["pkey"],
            "product_name": sub["pkey"].map(lambda k: b.product_name(k) if k else None),
            "product_group": sub["pkey"].map(lambda k: pg.get(k)), "kind": sub["kind"], "unit": sub["unit_eff"],
            "qty": sub["qty"], "base_qty": sub["base_qty"],
            "base_unit": sub["pkey"].map(lambda k: b.base_unit(k) if k else None),
            "unit_price": sub["unit_price"], "amount": sub["amount"], "cost_per_unit": sub["cost_unit"],
            "is_valid": sub["valid"].astype(int),
        })
    if not b.products.empty:
        P = b.products
        drug = getattr(b, "_drug", {}) or {}
        V["products"] = pd.DataFrame({
            "product_key": P["key"], "code": P["code"], "name": P["key"].map(b.product_name), "product_group": P["group"],
            "base_unit": P["base_unit"], "price": P["price"], "cost": P["cost"],
            "active_ingredient": P["key"].map(lambda k: (drug.get(k) or (None, None))[0]),
            "is_prescription": P["key"].map(lambda k: int(bool((drug.get(k) or (None, None))[1]))),
        })
    if not b.payments.empty:
        Pm = b.payments
        V["payments"] = pd.DataFrame({"order_id": Pm["order_id"], "pay_no": Pm.groupby("order_id", dropna=False).cumcount() + 1,
                                      "pay_date": Pm["pay_date"], "amount": Pm["pay_amount"],
                                      "method": Pm.get("method_n")})
    if not b.purchases.empty:
        H = b.purchases
        V["purchases"] = pd.DataFrame({"receipt_id": H["receipt_id"], "receipt_date": H.get("date"),
                                       "supplier": H.get("supplier_name", H.get("supplier_id")), "total": H.get("total"),
                                       "cancelled": H["cancelled"].astype(int)})
    if not b.purchase_lines.empty:
        PL = b.purchase_lines
        V["purchase_lines"] = pd.DataFrame({"receipt_id": PL["receipt_id"], "line_no": PL.groupby("receipt_id", dropna=False).cumcount() + 1,
                                            "receipt_date": PL.get("date"),
                                            "product_key": PL["pkey"], "unit": PL.get("unit"), "qty": PL.get("qty"),
                                            "base_qty": PL.get("base_qty"), "unit_cost": PL.get("unit_cost"),
                                            "amount": PL.get("amount"), "lot": PL.get("lot"), "expiry": PL.get("expiry")})
    if not b.expenses.empty:
        E = b.expenses
        V["expenses"] = pd.DataFrame({"expense_date": E.get("date"), "amount": E.get("amount"), "category": E.get("category")})
    if not b.deliveries.empty:
        D = b.deliveries
        V["deliveries"] = pd.DataFrame({"order_id": D["order_id"], "trip_no": D.get("trip_no"), "delivery_date": D.get("date"),
                                        "product_key": D.get("pkey"), "base_qty": D.get("base_qty", D.get("qty_base"))})
    return V


def _type(s: pd.Series) -> str:
    x = s.dropna()
    if x.empty:
        return "TEXT"
    v = x.iloc[0]
    if isinstance(v, (bool, np.bool_)):
        return "BOOLEAN"
    if isinstance(v, (int, np.integer)):
        return "INTEGER"
    if isinstance(v, (float, np.floating)):
        return "DOUBLE"
    if isinstance(v, dt.datetime):
        return "TIMESTAMP"
    if isinstance(v, dt.date):
        return "DATE"
    return "TEXT"




# ============================================================================ keys and M-Schema
# Declared keys of the canonical views (the views, not the raw files, are what the SQL sees).
# Composite keys are explicit so PFKeyIdentifier can complete them; every declaration is
# verified on the data (``verify_keys``) and dropped if it does not hold.
VIEW_KEYS: Dict[str, Dict] = {
    "orders": {"pk": ["order_id"], "date": "order_date", "guards": ["is_valid", "status"]},
    "sales_lines": {"pk": ["order_id", "line_no"], "date": "order_date", "guards": ["is_valid", "kind"]},
    "products": {"pk": ["product_key"]},
    "payments": {"pk": ["order_id", "pay_no"], "date": "pay_date"},
    "purchases": {"pk": ["receipt_id"], "date": "receipt_date", "guards": ["cancelled"]},
    "purchase_lines": {"pk": ["receipt_id", "line_no"], "date": "receipt_date"},
    "expenses": {"pk": [], "date": "expense_date"},
    "deliveries": {"pk": [], "date": "delivery_date"},
}
VIEW_FKS: List[Tuple[str, str, str, str]] = [
    ("sales_lines", "order_id", "orders", "order_id"),
    ("sales_lines", "product_key", "products", "product_key"),
    ("payments", "order_id", "orders", "order_id"),
    ("purchase_lines", "receipt_id", "purchases", "receipt_id"),
    ("purchase_lines", "product_key", "products", "product_key"),
    ("deliveries", "order_id", "orders", "order_id"),
    ("deliveries", "product_key", "products", "product_key"),
]


class MSchema:
    """Semi-structured schema (same data model and text format as XGenerationLab/M-Schema), extended with
    composite primary keys, a per-table date column and business-condition ("guard") columns."""

    def __init__(self, db_id: str = "Anonymous"):
        self.db_id = db_id
        self.tables: Dict[str, Dict] = {}
        self.foreign_keys: List[List[str]] = []      # [table, column, ref_table, ref_column]
        self.issues: List[str] = []                  # key declarations rejected by the data

    # -- construction (official API names) -----------------------------------
    def add_table(self, name: str, comment: str = "", date_col: Optional[str] = None, guards: Optional[List[str]] = None):
        self.tables[name] = {"fields": {}, "comment": comment, "date": date_col, "guards": list(guards or [])}

    def add_field(self, table: str, field: str, field_type: str = "", primary_key: bool = False, comment: str = "",
                  examples: Optional[list] = None):
        self.tables[table]["fields"][field] = {"type": field_type, "primary_key": primary_key, "comment": comment,
                                               "examples": list(examples or [])}

    def add_foreign_key(self, table: str, field: str, ref_table: str, ref_field: str):
        self.foreign_keys.append([table, field, ref_table, ref_field])

    # -- queries --------------------------------------------------------------
    def columns(self) -> List[str]:
        return [f"{t}.{c}" for t, ti in self.tables.items() for c in ti["fields"]]

    def primary_key(self, table: str) -> List[str]:
        return [c for c, f in self.tables.get(table, {}).get("fields", {}).items() if f.get("primary_key")]

    def guards(self, table: str) -> List[str]:
        ti = self.tables.get(table, {})
        return [c for c in ti.get("guards", []) if c in ti.get("fields", {})]

    def date_column(self, table: str) -> Optional[str]:
        ti = self.tables.get(table, {})
        c = ti.get("date")
        return c if c and c in ti.get("fields", {}) else None

    def edges(self, table: str) -> List[Tuple[str, str, str, str]]:
        """FK edges touching a table, oriented from it: (table, column, other_table, other_column)."""
        out = []
        for t, c, rt, rc in self.foreign_keys:
            if t == table:
                out.append((t, c, rt, rc))
            elif rt == table:
                out.append((rt, rc, t, c))
        return out

    # -- rendering ------------------------------------------------------------
    def to_mschema(self, selected_columns: Optional[List[str]] = None, example_num: int = 3,
                   extra_examples: Optional[Dict[str, List[str]]] = None) -> str:
        sel = None if selected_columns is None else {s.lower() for s in selected_columns}
        tabs = None if sel is None else {s.split(".")[0] for s in sel}
        extra = {k.lower(): list(v) for k, v in (extra_examples or {}).items()}
        out = [f"【DB_ID】 {self.db_id}", "【Schema】"]
        for t, ti in self.tables.items():
            if tabs is not None and t.lower() not in tabs:
                continue
            out.append(f"# Table: {t}, {ti['comment']}" if ti.get("comment") else f"# Table: {t}")
            lines = []
            for c, f in ti["fields"].items():
                key = f"{t}.{c}".lower()
                if sel is not None and key not in sel:
                    continue
                line = f"({c}:{f['type'].upper()}"
                if f.get("comment"):
                    line += f", {f['comment']}"
                if f.get("primary_key"):
                    line += ", Primary Key"
                retrieved = extra.get(key, [])
                ex = list(dict.fromkeys(str(e) for e in retrieved + f.get("examples", []) if e is not None))
                ex = ex[:max(example_num, len(retrieved))]
                if f["type"].upper() in ("DATE", "TIMESTAMP"):
                    ex = ex[:1]
                elif ex and max(len(e) for e in ex) > 50 and not retrieved:
                    ex = []
                if ex:
                    line += f", Examples: [{', '.join(ex)}]"
                lines.append(line + ")")
            out.append("[")
            out.append(",\n".join(lines))
            out.append("]")
        fks = [fk for fk in self.foreign_keys if tabs is None or (fk[0].lower() in tabs and fk[2].lower() in tabs)]
        if fks:
            out.append("【Foreign keys】")
            out.extend(f"{t}.{c}={rt}.{rc}" for t, c, rt, rc in fks)
        return "\n".join(out)

    def dump(self) -> Dict:
        return {"db_id": self.db_id, "tables": self.tables, "foreign_keys": self.foreign_keys, "issues": self.issues}


def verify_keys(V: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, List[str]], List[Tuple[str, str, str, str]], List[str]]:
    """Keep only key declarations that hold on the data: PK unique and non-null; FK values (mostly) present in
    the referenced key (>= 95 %: hand-kept books contain a few dangling ids)."""
    issues, pks, fks = [], {}, []
    for t, meta in VIEW_KEYS.items():
        if t not in V:
            continue
        pk = [c for c in meta.get("pk", []) if c in V[t].columns]
        if pk and len(pk) == len(meta["pk"]):
            df = V[t][pk]
            if df.isna().any().any() or df.duplicated().any():
                issues.append(f"{t}: ({', '.join(pk)}) is not unique / has nulls -> no primary key")
                pk = []
        else:
            pk = []
        pks[t] = pk
    for t, c, rt, rc in VIEW_FKS:
        if t not in V or rt not in V or c not in V[t].columns or rc not in V[rt].columns:
            continue
        child = V[t][c].dropna()
        if child.empty:
            continue
        ok = float(child.isin(set(V[rt][rc].dropna())).mean())
        if ok >= 0.95:
            fks.append((t, c, rt, rc))
        else:
            issues.append(f"{t}.{c} -> {rt}.{rc}: only {ok:.0%} of values match -> no foreign key")
    return pks, fks, issues


def build_mschema(biz_id: str, V: Dict[str, pd.DataFrame], n_examples: int = 3) -> MSchema:
    ms = MSchema(biz_id)
    pks, fks, ms.issues = verify_keys(V)
    for name, df in V.items():
        desc, docs = VIEW_DOCS.get(name, ("", {}))
        meta = VIEW_KEYS.get(name, {})
        ms.add_table(name, desc, meta.get("date"), meta.get("guards"))
        for c in df.columns:
            ex = [str(v) for v in df[c].dropna().astype(str).unique()[:n_examples]]
            ms.add_field(name, c, _type(df[c]), c in pks.get(name, []), docs.get(c, ""), ex)
    for t, c, rt, rc in fks:
        ms.add_foreign_key(t, c, rt, rc)
    return ms


def render_mschema(biz_id: str, V: Dict[str, pd.DataFrame], n_examples: int = 3) -> str:
    """Full-schema M-Schema text (the baseline / fallback schema of the Schema Filter)."""
    return build_mschema(biz_id, V, n_examples).to_mschema()
