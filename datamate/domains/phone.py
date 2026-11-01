"""Phone-shop books: item type (máy mới / máy cũ / phụ kiện) from the abbreviation sheet,
trade-ins deducted from what the customer pays (not from revenue), brand / model line.

Activated from the data shape (an abbreviation sheet whose types include a 'phụ kiện'-like
category, or an IMEI column), never from a business id.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from ..textnorm import fold, fold_key

ACCESSORY = re.compile(r"phu kien|accessor")
USED = re.compile(r"may cu|\bcu\b|used|2nd")
NEW = re.compile(r"may moi|\bmoi\b|new")
DOMAIN_VOCAB = set("""may moi cu phu kien chiec dien thoai dong hang thu tra khach phai bot tien sau gom
ca lan tinh tru doi""".split())


def is_phone_shop(b, L=None) -> bool:
    L = b.lines if L is None else L
    if L.empty or "product_group" not in L.columns:
        return False
    groups = {fold(str(g)) for g in L["product_group"].dropna().unique()}
    return any(ACCESSORY.search(g) for g in groups) and ("imei" in L.columns or any(USED.search(g) for g in groups))


def brand_of(name: str) -> str:
    f = fold(name)
    if f.startswith("iphone") or f.startswith("ip"):
        return "Apple"
    return name.split()[0] if name else name


def model_of(name: str) -> str:
    m = re.match(r"^(.*?)\s+\d+\s*(?:GB|TB)\b", name, re.I)
    return (m.group(1) if m else name).strip()


def enrich_orders(b, O: pd.DataFrame, tradein: pd.DataFrame) -> pd.DataFrame:
    if tradein is not None and not tradein.empty and "order_id" in tradein.columns and "trade_in" in tradein.columns:
        t = tradein.dropna(subset=["order_id"]).groupby("order_id")["trade_in"].sum()
        cur = O["trade_in"] if "trade_in" in O.columns else pd.Series(0.0, index=O.index)
        O["trade_in"] = cur.fillna(0.0) + O["order_id"].map(t).fillna(0.0)
    return O


def parse(rp, s: str, q: str, ti, P) -> Optional["Frame"]:
    from ..nlu.frame import Frame
    b = rp.b
    if not b.flags.get("phone"):
        return None
    phones = bool(re.search(r"dien thoai|chiec|\bmay\b(?! cu thu)", s)) and not re.search(r"phu kien(?! ra)", s.replace("khong tinh phu kien", "").replace("bo phu kien", "").replace("tru phu kien", ""))
    if re.search(r"thu may cu|may cu thu lai|tien may cu|tru vao don", s) and not re.search(r"khach phai tra|phai tra", s):
        if not P:
            return None
        if re.search(r"%|phan tram|chiem", s):
            return Frame("phone_tradein_share", [P], answer_type="percent")
        return Frame("phone_tradein_total", [P], answer_type="number")
    if re.search(r"khach phai tra|phai tra", s):
        if not P:
            return None
        if re.search(r"trung binh|binh quan|moi don", s):
            return Frame("phone_pay_avg", [P], answer_type="number")
        return Frame("phone_pay_total", [P], answer_type="number")
    mb = re.search(r"\b(samsung|apple|iphone|xiaomi|oppo|vivo|realme|nokia|huawei|honor|tecno|itel)\b", s)
    if mb and re.search(r"%|phan tram|chiem", s) and re.search(r"may|so may|chiec", s):
        if not P:
            return None
        return Frame("phone_brand_share", [P], params={"brand": mb.group(1)}, answer_type="percent")
    if re.search(r"dong may", s) and re.search(r"nhieu|nhat|top|xep", s):
        if not P:
            return None
        k = re.search(r"\btop\s*(\d+)|\b(\d+) dong may", s)
        k = int(k.group(1) or k.group(2)) if k else 1
        return Frame("phone_top_models", [P], k=k, answer_type="list" if k > 1 else "string")
    if re.search(r"trung binh|binh quan", s) and re.search(r"moi may|1 may|mot may|moi chiec|1 chiec", s):
        if not P:
            return None
        return Frame("phone_avg_price", [P], params={"phones": True}, answer_type="number")
    if re.search(r"\bmay (?:don|bill|hoa don|phieu|ngay|khach|lan)\b", s) and not re.search(r"chiec|dien thoai|may moi|may cu", s):
        return None  # 'mấy đơn' (how many orders), not 'máy' (phones)
    if phones and re.search(r"(bao nhieu|may|bn|so) (chiec|may|dien thoai)\b|ban (?:ra )?(?:duoc |dc )?(?:bao nhieu|may|bn)", s) and \
            not re.search(r"tien|doanh thu", s):
        if not P:
            return None
        grp = "used" if re.search(r"may cu|dien thoai cu", s) and not re.search(r"may moi", s) else \
            "new" if re.search(r"may moi", s) and not re.search(r"may cu", s) else "phones"
        pk, _ = rp._product(q)
        return Frame("phone_units", [P], filters={"product": pk} if pk else {}, params={"grp": grp}, answer_type="number")
    return None


def _grp_keys(ex, grp: str):
    keys = set()
    gm = ex.product_groups()
    for k, g in gm.items():
        fg = fold(str(g))
        if grp == "phones" and not ACCESSORY.search(fg):
            keys.add(k)
        elif grp == "used" and USED.search(fg):
            keys.add(k)
        elif grp == "new" and NEW.search(fg) and not USED.search(fg):
            keys.add(k)
    return keys


def run(fe, fr):
    from ..engine.answer import Answer
    from ..engine.executor import Unanswerable, money, r2, _num
    ex = fe.ex
    it = fr.intent
    P = ex.clip(fr.periods[0])
    O = ex.orders(P, True)
    if it in ("phone_tradein_total", "phone_tradein_share"):
        if "trade_in" not in O.columns:
            raise Unanswerable("không có sổ thu máy cũ")
        t = O["trade_in"].fillna(0)
        if it == "phone_tradein_total":
            return Answer("number", money(t.sum()))
        return Answer("percent", r2((t > 0).sum() / max(1, len(O)) * 100))
    if it in ("phone_pay_total", "phone_pay_avg"):
        due = O["due"].fillna(0)
        if it == "phone_pay_total":
            return Answer("number", money(due.sum()))
        return Answer("number", money(due.sum() / max(1, len(O))))
    L = ex.lines(P)
    phones = L[L["pkey"].isin(_grp_keys(ex, "phones"))]
    if it == "phone_units":
        sub = L[L["pkey"].isin(_grp_keys(ex, fr.params.get("grp", "phones")))]
        if fr.filters.get("product"):
            sub = L[L["pkey"] == fr.filters["product"]]
        return Answer("number", _num(sub["base_qty"].sum()))
    if it == "phone_avg_price":
        q = phones["base_qty"].sum()
        if q == 0:
            raise Unanswerable("không có máy bán ra")
        return Answer("number", money(phones["amount"].sum() / q))
    if it == "phone_brand_share":
        br = fr.params["brand"].lower()
        br = "apple" if br in ("iphone", "apple") else br
        names = phones["pkey"].map(ex.pname)
        m = names.map(lambda n: fold(brand_of(n)) == br)
        tot = phones["base_qty"].sum()
        if tot == 0:
            raise Unanswerable("không có máy bán ra")
        return Answer("percent", r2(phones.loc[m, "base_qty"].sum() / tot * 100))
    if it == "phone_top_models":
        names = phones["pkey"].map(ex.pname).map(model_of)
        s = phones.groupby(names)["base_qty"].sum()
        items = sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))
        k = fr.k or 1
        if not items:
            raise Unanswerable("không có máy bán ra")
        if len(items) > k and items[k - 1][1] == items[k][1]:
            return Answer("abstain", reason="tie at boundary")
        if k == 1:
            return Answer("string", items[0][0])
        return Answer("list", [n for n, _ in items[:k]])
    raise KeyError(it)
