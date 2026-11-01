"""Label-free ingest diagnostics: internal consistency of the canonical model of each shop."""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATAMATE_CACHE", ".cache")
import numpy as np
from datamate.ingest.builder import load_biz
root = sys.argv[1]
for bid in sys.argv[2:] or sorted(os.listdir(os.path.join(root, "data"))):
    b = load_biz(os.path.join(root, "data", bid), bid)
    O, L = b.orders, b.lines
    r = {"orders": len(O), "lines": len(L), "range": f"{b.data_start}..{b.data_end}", "flags": b.flags,
         "other": [t.name for t in b.other]}
    if not O.empty:
        r["status"] = O["status_final"].value_counts().to_dict()
        r["raw_status"] = dict(collections.Counter(map(str, O.get("status", []))).most_common(8)) if "status" in O else None
        for c in ("goods_total", "order_total", "order_discount", "shipping_fee", "paid", "remaining", "trade_in"):
            if c in O.columns:
                r[c] = round(float(O[c].fillna(0).sum()))
        r["revenue_sum"] = round(float(O["revenue"].fillna(0).sum()))
        if "goods_total" in O.columns and O["has_lines"].any():
            m = O[O["has_lines"]]
            d = (m["goods_total"] - m["revenue"]).abs()
            r["goods_vs_lines_mismatch"] = f"{int((d > 1).sum())}/{len(m)}"
        if "order_total" in O.columns and "goods_total" in O.columns:
            calc = O["goods_total"].fillna(0) - O.get("order_discount", 0 * O["goods_total"]).fillna(0) + O.get("shipping_fee", 0 * O["goods_total"]).fillna(0)
            r["total_vs_calc_mismatch"] = f"{int(((O['order_total'] - calc).abs() > 1).sum())}/{len(O)}"
        r["cust_keys"] = int(O["cust_key"].nunique()) if "cust_key" in O else 0
        r["pay"] = O["pay_method_n"].value_counts().head(6).to_dict() if "pay_method_n" in O else None
        r["channel"] = O["channel_n"].value_counts().head(6).to_dict() if "channel_n" in O else None
        r["hours"] = int(O["hour"].notna().sum()) if "hour" in O else 0
    if not L.empty:
        r["kinds"] = L["kind"].value_counts().to_dict()
        r["neg"] = int(L["neg"].sum())
        r["nan_amount"] = int(L["amount"].isna().sum()); r["nan_date"] = int(L["date"].isna().sum())
        qp = L["qty"] * L["unit_price"] - L["line_discount"].fillna(0)
        bad = ((qp - L["amount"]).abs() > 1) & L["qty"].notna() & L["unit_price"].notna()
        r["qty*price!=amount"] = f"{int(bad.sum())}/{len(L)}"
        r["no_pkey_items"] = int(((L["kind"] == "item") & L["pkey"].isna()).sum())
        r["n_products"] = int(L["pkey"].nunique())
        r["base!=qty"] = int((L["base_qty"] != L["qty"]).sum())
    print("####", bid); print(json.dumps(r, ensure_ascii=False, default=str))
