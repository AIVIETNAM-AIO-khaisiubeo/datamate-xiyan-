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
        "order_id": "mã đơn", "order_date": "ngày đặt đơn", "product_key": "khoá mặt hàng",
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
        "order_id": "mã đơn", "pay_date": "ngày trả", "amount": "số tiền trả", "method": "hình thức",
    }),
    "purchases": ("một dòng một phiếu nhập", {
        "receipt_id": "mã phiếu", "receipt_date": "ngày phiếu", "supplier": "nhà cung cấp", "total": "tổng tiền phiếu",
        "cancelled": "1 nếu phiếu huỷ",
    }),
    "purchase_lines": ("một dòng một mặt hàng trong phiếu nhập", {
        "receipt_id": "mã phiếu", "receipt_date": "ngày phiếu", "product_key": "khoá mặt hàng", "unit": "đơn vị nhập",
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
        pg = {}
        if not b.products.empty:
            pg = dict(zip(b.products["key"], b.products["group"]))
        sub = L[~L["neg"]]
        V["sales_lines"] = pd.DataFrame({
            "order_id": sub["order_id"], "order_date": sub["date"].map(_d), "product_key": sub["pkey"],
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
        V["payments"] = pd.DataFrame({"order_id": Pm["order_id"], "pay_date": Pm["pay_date"], "amount": Pm["pay_amount"],
                                      "method": Pm.get("method_n")})
    if not b.purchases.empty:
        H = b.purchases
        V["purchases"] = pd.DataFrame({"receipt_id": H["receipt_id"], "receipt_date": H.get("date"),
                                       "supplier": H.get("supplier_name", H.get("supplier_id")), "total": H.get("total"),
                                       "cancelled": H["cancelled"].astype(int)})
    if not b.purchase_lines.empty:
        PL = b.purchase_lines
        V["purchase_lines"] = pd.DataFrame({"receipt_id": PL["receipt_id"], "receipt_date": PL.get("date"),
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


def render_mschema(biz_id: str, V: Dict[str, pd.DataFrame], n_examples: int = 3) -> str:
    out = [f"【DB_ID】 {biz_id}", "【Schema】"]
    for name, df in V.items():
        desc, cols = VIEW_DOCS.get(name, ("", {}))
        out.append(f"# Table: {name}, {desc}")
        out.append("[")
        rows = []
        for c in df.columns:
            ex = [str(v) for v in df[c].dropna().astype(str).unique()[:n_examples]]
            pk = ", Primary Key" if (name == "orders" and c == "order_id") or (name == "products" and c == "product_key") else ""
            rows.append(f"({c}:{_type(df[c])}{pk}, {cols.get(c, '')}, Examples: [{', '.join(ex)}])")
        out.append(",\n".join(rows))
        out.append("]")
    out.append("【Foreign keys】")
    if "sales_lines" in V and "orders" in V:
        out.append("sales_lines.order_id=orders.order_id")
    if "payments" in V:
        out.append("payments.order_id=orders.order_id")
    if "sales_lines" in V and "products" in V:
        out.append("sales_lines.product_key=products.product_key")
    if "purchase_lines" in V:
        out.append("purchase_lines.receipt_id=purchases.receipt_id")
    return "\n".join(out)
