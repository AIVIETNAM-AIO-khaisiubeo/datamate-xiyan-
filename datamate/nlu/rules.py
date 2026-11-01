"""Rule-based semantic parser: normalised Vietnamese question -> Frame.

This is generator G1 of the multi-generator ensemble.  It is high-precision by design:
when it is unsure it returns ``None`` (no candidate) instead of a guess, so that the
selector can fall back to the LLM generators or abstain.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import List, Optional, Tuple

from ..ingest.builder import Biz
from ..textnorm import fold, fold_words, normalize_question, repair_fold
from .frame import Frame
from .linker import Linker
from .timeparse import LunarCalendar, Period, TimeInfo, month_period, parse_time, prev_month, tet_period


def _has(s: str, *pats: str) -> bool:
    return any(re.search(p, s) for p in pats)


REV = r"doanh thu|doanh so|ban duoc (?:tong cong )?(?:bao nhieu|may|nhieu|bn)? ?tien|ban duoc .*tien|tien ban|thu ve|thu duoc|ban ra .*tien|mang ve|ban dat|nhieu tien|tien hang|chi tieu|mua .*tien"
COUNT_WORDS = r"(?:bao nhieu|may|so|dem|tong cong|total|co)"
_NUM_WORDS = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}


def _money_in(s: str) -> Optional[float]:
    m = re.search(r"(\d{1,3}(?:[.,]\d{3})+|\d+)\s*(?:d|dong|vnd)?\b", s)
    if not m:
        return None
    return float(re.sub(r"[.,]", "", m.group(1)))


def _threshold(s: str) -> Optional[Tuple[str, float]]:
    """'tu 55.000 dong tro len' / '>= 590.000' / 'dat 41.000 dong tro len' / 'tren 1.000.000'."""
    m = re.search(r"(tu|>=|≥|dat|it nhat|toi thieu|tren|hon|qua|duoi|<=|nho hon|lon hon)\s*(\d{1,3}(?:[.,]\d{3})+|\d+)\s*(k|nghin|ngan|tr|trieu)?\s*(?:d|dong|vnd)?\s*(tro len|tro xuong)?", s)
    if not m:
        return None
    v = float(re.sub(r"[.,]", "", m.group(2)))
    if m.group(3) in ("k", "nghin", "ngan"):
        v *= 1000
    elif m.group(3) in ("tr", "trieu"):
        v *= 1_000_000
    w = m.group(1)
    if w in ("tu", ">=", "≥", "dat", "it nhat", "toi thieu"):
        op = ">=" if (m.group(4) != "tro xuong") else "<="
    elif w in ("tren", "hon", "qua", "lon hon"):
        op = ">"
    elif w in ("duoi", "nho hon"):
        op = "<"
    else:
        op = "<="
    return op, v


def _count_threshold(s: str, noun: str) -> Optional[Tuple[str, int]]:
    """'mua it nhat 3 don' / 'tu 4 don tro len' / '>= 8 don' / 'tren 5 don'."""
    m = re.search(r"(it nhat|tu|>=|≥|toi thieu|tren|hon|qua)\s*(\d+)\s*" + noun + r"(?:\s*(tro len))?", s)
    if not m:
        m2 = re.search(r"\b(?:o|trong|co)\s+(\d+)\s*" + noun + r"(?: khac nhau)? tro len", s)
        if m2:
            return ">=", int(m2.group(1))
        return None
    w = m.group(1)
    op = ">" if w in ("tren", "hon", "qua") else ">="
    return op, int(m.group(2))


def _topk(s: str) -> Optional[int]:
    m = re.search(r"\btop\s*(\d+)\b", s)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\s+(?:mat hang|san pham|mon|hang|khach|khach hang|nguoi|nhan vien|nha cung cap|ncc|best seller|sp|loai)\b", s)
    if m and int(m.group(1)) <= 50:
        return int(m.group(1))
    m = re.search(r"\b(hai|ba|bon|nam|sau|bay|tam|chin|muoi)\s+(?:mat hang|san pham|mon|hang|khach|khach hang)\b", s)
    if m:
        return _NUM_WORDS[m.group(1)]
    return None


class RuleParser:
    def __init__(self, biz: Biz, cal: Optional[LunarCalendar] = None):
        self.b = biz
        self.cal = cal
        self.lk = Linker(biz)

    # ------------------------------------------------------------------ filters
    def _filters(self, s: str, q: str) -> dict:
        f = {}
        if _has(s, r"\bca sang\b"):
            f["shift"] = "sang"
        if _has(s, r"\bca chieu\b"):
            f["shift"] = "chieu" if "shift" not in f else f["shift"]
        if _has(s, r"chuyen khoan|\bck\b"):
            f["pay"] = "transfer"
        elif _has(s, r"tien mat|\btm\b|tien tuoi|\bcash\b"):
            f["pay"] = "cash"
        elif _has(s, r"\bcod\b"):
            f["pay"] = "cod"
        elif _has(s, r"vi dien tu|\bqua vi\b|bang vi\b|momo|zalopay|zalo pay"):
            f["pay"] = "ewallet"
        if _has(s, r"(?:ban qua|qua|dat qua|kenh) (?:dien thoai|goi dien|phone)"):
            f["channel"] = "phone"
        elif _has(s, r"zalo"):
            f["channel"] = "zalo"
        elif _has(s, r"san tmdt|san thuong mai|shopee|lazada|tiki|tiktok"):
            f["channel"] = "ecommerce"
        elif _has(s, r"facebook|\bfb\b"):
            f["channel"] = "facebook"
        elif _has(s, r"tai quay|tai cua hang|tai tiem|tai quan|truc tiep tai"):
            f["channel"] = "store"
        if _has(s, r"(?:khong|\bk\b|hong|hok|ko|chang) (?:can |phai )?(?:thuoc )?(?:ke don|don bac si)|tru thuoc ke don|ngoai thuoc ke don|\botc\b|khong can don"):
            f["rx"] = False
        elif _has(s, r"ke don|theo don"):
            f["rx"] = True
        m = re.search(r"\bsize\s+(nb|xxxl|xxl|xl|m|l|s|\d{1,2})\b", s)
        if m:
            f["size"] = m.group(1).upper()
        m = re.search(r"\bsua (?:bot |cong thuc )?(?:so) (\d{1,2})\b", s)
        if m:
            f["size"] = m.group(1)
        br = self.lk.brand(s)
        if br:
            f["brand"] = br
        g = self.lk.group(q) or self.lk.group(self._qr) or self.lk.group(s)
        if g:
            f["group"] = g
        cg = self.lk.cust_group(q) or self.lk.cust_group(self._qr) or self.lk.cust_group(s)
        if cg and _has(s, r"khach|nhom"):
            f["cust_group"] = cg
        ing = self.lk.ingredient(q) or self.lk.ingredient(self._qr)
        if ing:
            f["ingredient"] = ing
        ms = re.search(r"(?:khach|nguoi) (?:hang )?(?:o|tai|song o|nha o) (?:duong|pho) ([a-z0-9]+(?: [a-z0-9]+){0,3}?)(?= mua| la\b| co\b| chiem| nhieu| bao| tong| thang| tuan| nam|$)", s)
        if ms:
            f["street"] = ms.group(1).strip()
        st = self.lk.staff_ref(q)
        if st and _has(s, r"nhan vien|nv\d|\bnv\b|nguoi ban"):
            f["staff"] = st
        return f

    def _product(self, q: str) -> Tuple[Optional[str], Optional[str]]:
        for text in (q, self._qr):
            ps = self.lk.products(text)
            if ps:
                name, keys, _ = max(ps, key=lambda x: len(x[0]))
                if len(keys) == 1:
                    return keys[0], name
                return None, name
        fz = self.lk.product_fuzzy(self._qr, 0.9)
        if fz and len(fz[1]) == 1 and len(fz[0].split()) >= 3:
            return fz[1][0], fz[0]
        return None, None

    # ------------------------------------------------------------------ main
    def parse(self, question: str, today: dt.date) -> Optional[Frame]:
        nq = normalize_question(question)
        s = " " + nq + " "
        ti = parse_time(nq, today, self.cal)
        self._qr = repair_fold(question)
        q = question
        fr = self._probe_unanswerable(self._mask_entities(s, q), q)
        if fr is None:
            fr = self._parse(s, q, ti, today)
        if fr is not None:
            fr.params.setdefault("nq", nq)
        return fr

    def _mask_entities(self, s: str, q: str) -> str:
        """Hide product / customer names so probes do not fire on e.g. 'Sữa tươi trân châu đường đen'."""
        out = s
        names = set()
        for text in (q, self._qr, s):
            names |= {n for n, _, _ in self.lk.products(text)}
            names |= {n for n, _, _ in self.lk.customers(text)}
        fs = " " + s + " "
        for n in list(self.lk.staff) + list(self.lk.suppliers) + list(self.lk.groups) + list(self.lk.cust_groups) + \
                list(self.lk.ingredients) + list(getattr(self.lk, "_extra_names", [])):
            if n and (" " + n + " ") in fs:
                names.add(n)
        for n in sorted(names, key=len, reverse=True):
            out = out.replace(" " + n + " ", " _e_ ")
        return out

    # ------------------------------------------------------------------ unanswerable probes
    def _probe_unanswerable(self, s: str, q: str) -> Optional[Frame]:
        def U(reason):
            return Frame("unanswerable", reason=reason, answer_type="unanswerable")
        if _has(s, r"\bkhen\b|lanh nghe|tay nghe|tan tam|le phep|chu dao|than thien|quy men|nhiet tinh|dang tin cay|"
                   r"uy tin|hai long|de thuong|vui ve|trung thuc|cham soc tot|phuc vu tot|thai do"):
            return U("Dữ liệu không ghi đánh giá chủ quan về nhân viên, khách hay nhà cung cấp")
        if _has(s, r"nen uong|lieu dung|moi lan (?:uong )?may|tac dung phu|chong chi dinh|uong luc nao|cach dung|bao quan"):
            return U("Câu hỏi cần kiến thức chuyên môn ngoài dữ liệu cửa hàng")
        if _has(s, r"khach (?:hang )?(?:nu|nam)\b|gioi tinh|\bphu nu\b|\bdan ong\b|\bdan ba\b"):
            C = self.b.customers
            has_gender = (not C.empty) and any(c for c in C.columns if c in ("gender",))
            if not has_gender:
                return U("Dữ liệu không ghi giới tính khách")
        if _has(s, r"(?:khach|nha|tai|song) (?:o |tai )?duong [a-z]|\bo duong [a-z]|dia chi|\bnha o\b|khach (?:o|tai) (?:phuong|quan)\b"):
            C = self.b.customers
            has_addr = (not C.empty) and "address" in C.columns and C["address"].notna().any()
            if not has_addr:
                return U("Dữ liệu không ghi địa chỉ khách")
        if _has(s, r"\bvi (?:loi|mua nham|giao tre|giao cham|hong|het han|khong thich)|ly do|nguyen nhan") and \
                _has(s, r"huy|tra hang|tra lai|doi tra"):
            R = self.b.returns
            if R.empty or "reason" not in R.columns or R["reason"].isna().all():
                return U("Dữ liệu không ghi lý do huỷ/trả hàng")
        if _has(s, r"nhom (?:khach )?(?:than thiet|vip|si|thuong|bac|vang|kim cuong)|khach (?:hang )?(?:than thiet|vip|si)\b"):
            if not self.lk.cust_groups:
                return U("Dữ liệu không ghi nhóm khách")
        return None

    def _period(self, ti: TimeInfo) -> Optional[Period]:
        if ti.tet and self.cal:
            y = ti.tet[0][0]
            p = tet_period(self.cal, y)
            if p:
                return p
        if ti.periods:
            return ti.periods[0]
        return None

    def _parse(self, s: str, q: str, ti: TimeInfo, today: dt.date) -> Optional[Frame]:
        f = self._filters(s, q)
        pkey, pname = self._product(q)
        P = self._period(ti)
        k = _topk(s)
        if self.b.flags.get("repair"):
            from ..domains import repair
            fr = repair.parse(self, s, q, ti, P)
            if fr is not None:
                return fr
        if self.b.flags.get("phone"):
            from ..domains import phone
            fr = phone.parse(self, s, q, ti, P)
            if fr is not None:
                return fr

        # ---------------------------------------------------------------- newer generic intents
        if _has(s, r"gio cao diem|khung gio nao|gio nao .*(?:dong|nhieu don|dong khach|dat khach)"):
            if not P:
                return None
            return Frame("peak_hour", [P], answer_type="number")
        mw = re.search(r"trung binh (?:moi |mot |1 )?(?:ngay )?(thu (?:hai|ba|tu|nam|sau|bay|2|3|4|5|6|7)|chu nhat|cn)\b", s)
        if mw:
            if not P:
                return None
            wd = _weekday(mw.group(1))
            metric = "order_count" if _has(s, r"\bdon\b|bill|\bluot\b|\bphieu\b|\bxe\b") and not _has(s, r"doanh thu|tien") else "revenue"
            return Frame("avg_weekday", [P], params={"weekday": wd, "metric": metric}, answer_type="number")
        if _has(s, r"voucher|vau cho|dung diem|tru diem|doi diem|phi ship|phi giao hang|tien ship") and \
                _has(s, r"bao nhieu|bn|tong|het|duoc|giam"):
            if not P:
                return None
            what = "voucher" if _has(s, r"voucher|vau cho") else "points" if _has(s, r"diem") else "shipping"
            return Frame("special_lines", [P], params={"what": what}, answer_type="number")
        if _has(s, r"thanh vien moi|khach moi dang ky|dang ky (?:thanh vien|moi)|khach (?:hang )?moi"):
            if not P:
                return None
            return Frame("new_members", [P], answer_type="number")
        ma = re.search(r"(?:khach|nguoi) (?:hang )?(?:o|tai|song o|nha o) (?:duong|pho) ([a-z0-9 ]+?)(?: mua| la| co|\?|$| chiem| nhieu| bao)", s)
        if ma and _has(s, r"bao nhieu khach|may khach|so khach|co bao nhieu"):
            return Frame("customers_on_street", params={"street": ma.group(1).strip()}, answer_type="number")
        # ---------------------------------------------------------------- price effect
        if _has(s, r"tang gia|len gia|doi gia|tang len|tang tu") and _has(s, r"\d+ ngay") and _has(s, r"thay doi|tang hay giam|nhieu hay it|so|chenh|%|phan tram"):
            if not pkey:
                return None
            d = ti.explicit_dates[0] if ti.explicit_dates else None
            m = re.search(r"len (?:gia )?(\d{1,3}(?:[.,]\d{3})+|\d+)\s*(?:d|dong)", s)
            newp = float(re.sub(r"[.,]", "", m.group(1))) if m else None
            n = ti.window_days or int(re.search(r"(\d+) ngay", s).group(1))
            avg = _has(s, r"trung binh moi ngay|binh quan moi ngay")
            return Frame("price_effect", filters={"product": pkey}, params={"date": d, "new_price": newp, "n": n, "avg": avg},
                         answer_type="percent")
        # ---------------------------------------------------------------- price on a day
        if _has(s, r"gia ban|don gia|ban ra voi|gia bao nhieu|gia bn|\bban\b.*\bgia\b(?! doan)") and not _has(s, r"\bnhap\b|binh quan|\bvon\b") and ti.explicit_dates:
            if not pkey:
                return None
            return Frame("price_on", filters={"product": pkey}, params={"date": ti.explicit_dates[0]}, answer_type="number")
        # ---------------------------------------------------------------- purchases
        if _has(s, r"nhap tru (?:so )?ban|chenh lech nhap|nhap (?:va|voi) ban|so nhap tru"):
            if not pkey or not P:
                return None
            u = self.lk.unit_after(q, ("bao nhieu", "may", "duoc bao nhieu", "duoc", "la bao nhieu", "nhieu")) or _unit_tail(s)
            return Frame("purchase_minus_sales", [P], filters={"product": pkey}, unit=u, answer_type="number")
        if _has(s, r"nhap.*binh quan|binh quan.*nhap|gia nhap binh quan|nhap vao gia"):
            if not pkey or not P:
                return None
            m = re.search(r"(?:mot|1|moi)\s+([a-z0-9]+(?: \d+m3| 4m3)?)\b", s)
            u = m.group(1) if m else None
            return Frame("avg_purchase_price", [P], filters={"product": pkey}, unit=u, answer_type="number")
        if _has(s, r"nha cung cap|\bncc\b"):
            if _has(s, r"\bnao\b|\bai\b|la ai"):
                if not P:
                    return None
                by = "count" if _has(s, r"chinh|nhieu phieu|nhieu lan nhap|so phieu") else "amount"
                if _has(s, r"tin cay|uy tin|tot nhat|chat luong"):
                    return Frame("unanswerable", reason="Không có dữ liệu đánh giá nhà cung cấp", answer_type="unanswerable")
                return Frame("top_suppliers", [P], by=by, k=k or 1, answer_type="string" if not k or k == 1 else "list")
        if _has(s, r"nhap hang|phieu nhap|tien nhap|hang nhap|nhap vao|tien hang nhap") and not _has(s, r"ban"):
            if not P:
                return None
            return Frame("purchase_total", [P], answer_type="number")
        # ---------------------------------------------------------------- expiry / lots
        if _has(s, r"het han\b|han dung|han su dung|\bhsd\b"):
            snap = ti.snapshot or (ti.explicit_dates[0] if ti.explicit_dates else today)
            if _has(s, r"gan het han nhat|het han som nhat|han som nhat"):
                if not pkey:
                    return None
                return Frame("nearest_expiry", snapshot=snap, filters={"product": pkey}, answer_type="date")
            end = None
            m = re.search(r"(\d+) ngay toi", s)
            n = int(m.group(1)) if m else None
            if len(ti.explicit_dates) >= 2:
                end = ti.explicit_dates[-1]
            elif n is not None:
                end = snap + dt.timedelta(days=n)
            elif ti.explicit_dates and ti.explicit_dates[-1] > snap:
                end = ti.explicit_dates[-1]
            if end is None:
                return None
            if _has(s, r"tien von|gia tri|bao nhieu tien|bn tien|dang nam"):
                what = "value"
            elif _has(s, r"may lo|bao nhieu lo|so lo"):
                what = "lot_count"
            elif _has(s, r"(may|bao nhieu|so) (mat hang|san pham|thuoc|loai)"):
                what = "product_count"
            else:
                what = "product_list"
            return Frame("expiring", snapshot=snap, params={"end": end, "what": what},
                         answer_type="number" if what != "product_list" else "list")
        # ---------------------------------------------------------------- inventory
        if _has(s, r"het hang") and _has(s, r"may|bao nhieu|so|dem"):
            snap = ti.snapshot or (ti.explicit_dates[0] if ti.explicit_dates else today)
            return Frame("out_of_stock", snapshot=snap, answer_type="number")
        if _has(s, r"(gia tri|\bdang\b|bao nhieu tien|tien hang|tien von).*\b(kho|ton)\b|\b(kho|ton)\b.*(gia von|dang bao nhieu tien|gia tri)") and not _has(s, r"het han|han dung"):
            snap = ti.snapshot or (ti.explicit_dates[0] if ti.explicit_dates else today)
            return Frame("stock_value", snapshot=snap, answer_type="number")
        if pkey and _has(s, r"ton kho|trong kho|ton cuoi|con lai may|con lai bao nhieu|kho con|con ton|con bao nhieu|con may|ton bao nhieu|ton bn") \
                and not _has(s, r"\bno\b|thieu tien"):
            snap = ti.snapshot or (ti.explicit_dates[0] if ti.explicit_dates else today)
            if _has(s, r"chan|nguyen") and _has(s, r"\ble\b"):
                m = re.search(r"(?:may|bao nhieu|bn|duoc)\s+([a-z0-9]+(?: 4m3| 10m| 25kg)?)\s+(?:chan|nguyen)", s)
                if not m:
                    m = re.search(r"\b([a-z0-9]+(?: 4m3| 10m| 25kg)?)\s+(?:chan|nguyen)\b", s)
                big = m.group(1) if m else None
                cands = [x for x in re.findall(r"(?:may|bao nhieu|bn|duoc)\s+([a-z0-9]+(?: 4m3| 10m| 25kg| 30)?)\b", s)]
                if big:
                    cands.append(big)
                return Frame("stock_split", snapshot=snap, filters={"product": pkey}, unit=big, answer_type="list",
                             params={"unit_cands": cands})
            u = self.lk.unit_after(q) or _unit_tail(s)
            return Frame("stock", snapshot=snap, filters={"product": pkey}, unit=u, answer_type="number")
        # ---------------------------------------------------------------- debts
        if _has(s, r"no them|mua chiu them|ghi no them|chiu them") and _has(s, r"tra"):
            cs = self.lk.customers(q) or self.lk.customers(self._qr)
            if not cs and not self.lk.cust_names:
                return Frame("unanswerable", reason="Dữ liệu không ghi tên/mã khách nên không tách được nợ theo khách",
                             answer_type="unanswerable")
            if not cs or len(cs[0][1]) != 1 or not P:
                return None
            return Frame("debt_flow", [P], filters={"customer": cs[0][1][0]}, answer_type="list")
        if _has(s, r"thu hoi.*(cong )?no|tra (bot )?no|tra no|khach tra (bot )?no|thu no|tien khach tra bot|tra bot no") and not _has(s, r"con no"):
            if not P:
                return None
            return Frame("repayments", [P], answer_type="number")
        if _has(s, r"con no|cong no|no chua thu|con thieu|chua tra het|dang no|no nhieu nhat|tuoi no|chua thu|khach no|thieu tien|khoan no|\bno co tuoi"):
            snap = ti.snapshot or (ti.explicit_dates[0] if ti.explicit_dates else today)
            if _has(s, r"tuoi no|nhom tuoi|0.7|0 7") and not _has(s, r"co tuoi (?:qua|tren|hon)"):
                if _has(s, r"qua 15|tren 15|hon 15") and not _has(s, r"0.7|0 7"):
                    return Frame("debt_overdue", snapshot=snap, params={"days": 15, "op": ">"}, answer_type="number")
                return Frame("debt_aging", snapshot=snap, params={"buckets": [(0, 7), (8, 15), (16, None)]}, answer_type="list")
            m = re.search(r"(?:da |dat |co tuoi )?(?:hon|qua|tren) (\d+) ngay", s)
            if m:
                return Frame("debt_overdue", snapshot=snap, params={"days": int(m.group(1)), "op": ">"}, answer_type="number")
            if _has(s, r"nhieu nhat|lon nhat|top"):
                return Frame("top_debtors", snapshot=snap, k=k or 3, answer_type="table")
            if _has(s, r"(bao nhieu|may|so|bn|dem) (khach|nguoi)"):
                return Frame("debtor_count", snapshot=snap, answer_type="number")
            return Frame("debt_total", snapshot=snap, answer_type="number")
        # ---------------------------------------------------------------- customers by order count
        thr = _count_threshold(s, r"(?:don|lan|bill)")
        if thr and _has(s, r"khach"):
            if not P:
                return None
            if _has(s, r"%|phan tram|ti le|chiem"):
                return Frame("cust_min_orders_pct", [P], params={"op": thr[0], "n": thr[1]}, answer_type="percent")
            return Frame("cust_min_orders", [P], params={"op": thr[0], "n": thr[1]}, answer_type="number")
        if thr and _has(s, r"mat hang|san pham|mon|hang") and _has(s, r"co mat|xuat hien|nam trong|trong it nhat|o it nhat|co trong|it nhat|duoc mua o|mua o|\bo \d+ don"):
            if not P:
                return None
            return Frame("products_min_orders", [P], params={"op": thr[0], "n": thr[1]}, answer_type="number")
        # ---------------------------------------------------------------- unsold products
        if _has(s, r"khong ban (?:ra )?(?:duoc|dc)|bi e\b|mon e\b|hang e\b|ban cham|khong ban ra") and _has(s, r"mat hang|san pham|mon|hang"):
            if not P:
                return None
            return Frame("unsold_products", [P], answer_type="number")
        # ---------------------------------------------------------------- days with revenue threshold
        if _has(s, r"(may|bao nhieu|dem|so|nhieu) ngay") and _threshold(s):
            if not P:
                return None
            return Frame("days_revenue", [P], params={"thr": _threshold(s)}, answer_type="number")
        # ---------------------------------------------------------------- orders with amount threshold
        if _has(s, r"(may|bao nhieu|dem|so|co) (don|bill|hoa don)") and _threshold(s) and _has(s, r"tien hang|doanh thu|gia tri|tri gia|tong tien"):
            if not P:
                return None
            return Frame("orders_min_amount", [P], params={"thr": _threshold(s)}, filters=f, answer_type="number")
        # ---------------------------------------------------------------- deliveries
        if _has(s, r"dot giao|dot hang|chuyen giao|chuyen hang|xe .*giao|xe cho|giao .*dot|cho giao"):
            if not P:
                return None
            return Frame("delivery_trips", [P], answer_type="number")
        # ---------------------------------------------------------------- profit
        if _has(s, r"lai rong|loi nhuan rong|tru (?:het )?chi phi"):
            if not P:
                return None
            return Frame("net_profit", [P], answer_type="number")
        if _has(s, r"lai gop|loi gop|loi nhuan gop|loi nhat|lai nhat|lai nhieu nhat|loi nhieu nhat") and _has(s, r"hang nao|mat hang nao|san pham nao|mon nao|sp nao|\bnao\b|la gi|gi\b"):
            if not P:
                return None
            return Frame("top_products", [P], by="profit", k=k or 1, filters=f, answer_type="string" if not k or k == 1 else "list")
        if _has(s, r"lai gop|loi gop|loi nhuan gop"):
            if not P:
                return None
            return Frame("gross_profit", [P], filters=f, answer_type="number")
        # ---------------------------------------------------------------- rankings
        ql = q.lower()
        if re.search(r"hãng nào|hiệu nào|thương hiệu nào|hãng gì", ql) and _has(s, r"nhieu nhat|chay nhat|cao nhat|nhieu \w+ nhat|nhieu \w+ nhut"):
            if not P:
                return None
            return Frame("top_brands", [P], filters={k_: v for k_, v in f.items() if k_ in ("group", "size")},
                         answer_type="string")
        if _has(s, r"ban chay|ban nhieu nhat|hot nhat|best ?seller|chay nhat|ban duoc nhieu nhat|ban dc nhieu nhat|ban ra nhieu nhat|ban nhieu|ua chuong|ban (?:duoc |dc |ra )?nhieu \w+ nhat") and not _has(s, r"ngay nao|thang nao|nhan vien|khach"):
            if not P:
                return None
            by = "orders" if _has(s, r"ua chuong|nhieu don nhat") else "qty"
            return Frame("top_products", [P], by=by, k=k or 1, filters=f,
                         answer_type="list" if (k and k > 1) or _has(s, r"nhung mat hang|liet ke|nhung san pham|nhung gi|nhung mon") else "string")
        if _has(s, r"nhan vien|nguoi ban") and _has(s, r"gioi nhat|cao nhat|nhieu nhat|nhieu tien nhat|\bnao\b|la ai|\bai\b") and not _has(s, r"ma nv|nv\d"):
            if not P:
                return None
            if _has(s, r"tan tam|le phep|chu dao|than thien|tay nghe|lanh nghe|duoc khen|quy men|nhiet tinh"):
                return Frame("unanswerable", reason="Không có dữ liệu đánh giá nhân viên", answer_type="unanswerable")
            by = "days" if _has(s, r"cham chi|nhieu ngay") else "revenue"
            return Frame("top_staff", [P], by=by, k=k or 1, answer_type="string" if not k or k == 1 else "list")
        if _has(s, r"khach|\bai\b|nguoi mua") and _has(s, r"nhieu nhat|cao nhat|top|lon nhat|\bnao\b.*nhat|\bai\b.*nhat|nhieu \w+ nhat|nhieu \w+ \w+ nhat") and not _has(s, r"\bno\b"):
            if not P:
                return None
            if _has(s, r"nhieu don|nhieu lan|so don|nhieu hoa don|nhieu bill"):
                by = "orders"
            elif _has(s, r"trung thanh|nhieu ngay"):
                by = "days"
            elif _has(s, r"sop"):
                by = "aov"
            elif _has(s, r"doi y|huy nhieu|nhieu don huy"):
                by = "cancels"
            elif _has(s, r"chi tieu|tien|mua nhieu|chi nhieu|tong tien"):
                by = "revenue"
            else:
                return None
            return Frame("top_customers", [P], by=by, k=k or 1,
                         answer_type="list" if (k and k > 1) else "string")
        # ---------------------------------------------------------------- best day / month
        if _has(s, r"ngay nao|ngay ma") and _has(s, REV + r"|ban dat|nhieu don|dat khach"):
            if not P:
                return None
            by = "orders" if _has(s, r"nhieu don|dat khach") else "revenue"
            return Frame("best_day", [P], by=by, answer_type="date")
        if _has(s, r"thang nao") and len(ti.periods) >= 1:
            months = _months_list(ti)
            if not months:
                return None
            return Frame("best_month", months, answer_type="date")
        # ---------------------------------------------------------------- monthly list / increasing
        months = _months_list(ti)
        if _has(s, r"tang lien tuc|luon cao hon|thang sau .* cao hon|tang deu|cao hon thang .* va thang"):
            if len(months) < 2:
                return None
            return Frame("increasing", months, answer_type="boolean")
        if len(months) >= 2 and _has(s, REV) and _has(s, r"lan luot|moi thang|tung thang|ghi \d so|theo thu tu|tu thang cu|moi ky|liet ke"):
            return Frame("monthly_list", months, answer_type="list")
        # ---------------------------------------------------------------- cancel
        if _has(s, r"\bhuy\b"):
            if _has(s, r"ti le|%|phan tram|chiem"):
                if not P:
                    return None
                return Frame("cancel_rate", [P], filters={k_: v for k_, v in f.items() if k_ in ("shift", "channel", "pay")}, answer_type="percent")
            if _has(s, r"vi |ly do|nguyen nhan|do "):
                return Frame("unanswerable", reason="Dữ liệu không ghi lý do huỷ", answer_type="unanswerable")
            if not P:
                return None
            return Frame("cancel_count", [P], filters={k_: v for k_, v in f.items() if k_ in ("shift", "channel", "pay")}, answer_type="number")
        # ---------------------------------------------------------------- growth / comparisons
        is_pct = _has(s, r"%|phan tram")
        if _has(s, r"thay doi|tang hay giam|hon hay kem|chenh|tang/giam|tang giam|cao hay thap|nhieu hay it|so voi|so vs|len hay xuong|"
                   r"\bso (?:thang|ky|tuan|quy|nam|cung|ca)\b|\bso\b.*\b(?:truoc|lien truoc|ngoai)\b") and is_pct:
            metric = "order_count" if _has(s, r"so don|bao nhieu don|may don|so bill") else "revenue"
            # shift comparison
            if _has(s, r"ca chieu") and _has(s, r"ca sang"):
                if not P:
                    return None
                return Frame("growth", [P], params={"metric": metric, "mode": "shift"}, answer_type="percent")
            if ti.tet:
                if ti.before_tet_days is not None:
                    return Frame("growth", [], params={"metric": metric, "mode": "before_tet", "year": ti.tet[0][0],
                                                       "days": ti.before_tet_days}, answer_type="percent")
                if len(ti.tet) >= 1:
                    return Frame("growth", [], params={"metric": metric, "mode": "tet_yoy", "year": ti.tet[0][0]}, answer_type="percent")
            if not P:
                return None
            if ti.mtd_compare or P.kind == "mtd":
                return Frame("growth", [P], params={"metric": metric, "mode": "mtd"}, filters=f, answer_type="percent")
            if ti.yoy:
                return Frame("growth", [P], params={"metric": metric, "mode": "yoy"}, filters=f, answer_type="percent")
            if len(ti.periods) >= 2:
                return Frame("growth", ti.periods[:2], params={"metric": metric, "mode": "explicit"}, filters=f, answer_type="percent")
            return Frame("growth", [P], params={"metric": metric, "mode": "prev"}, filters=f, answer_type="percent")
        # ---------------------------------------------------------------- share of units (diapers by size / brand ...)
        if is_pct and (f.get("size") or f.get("brand")) and _has(s, r"chiem|ti trong|ti le|trong tong|tren tong|\bso (?:bich|lon|hop|goi|cai|chai|chiec|may)|tinh theo so"):
            if not P:
                return None
            base = f.get("group") or self.lk.group_loose(s)
            if not base:
                return None
            num = {k_: v for k_, v in f.items() if k_ in ("size", "brand")}
            return Frame("units_share", [P], filters=num, params={"base_group": base}, answer_type="percent")
        # ---------------------------------------------------------------- shares
        if is_pct and _has(s, r"chiem|ti trong|dong gop|bang .*tong|ti le|tren tong|trong tong|chiem bao"):
            if not P:
                return None
            part = {k_: v for k_, v in f.items() if k_ in ("group", "rx", "channel", "pay", "cust_group", "ingredient", "shift")}
            if pkey and not part:
                part["product"] = pkey
            if not part:
                return None
            return Frame("share", [P], filters=part, answer_type="percent")
        # ---------------------------------------------------------------- average order value
        if _has(s, r"trung binh (?:moi|mot|tren mot|1) (?:don|bill|hoa don)|gia tri trung binh|binh quan (?:moi|mot|1) (?:don|bill|hoa don)|trung binh .*moi don|moi don .*trung binh|binh quan .*don|moi don .*binh quan|(?:mot|1|moi) don .*binh quan"):
            if not P:
                return None
            ff = {k_: v for k_, v in f.items() if k_ in ("cust_group", "shift", "channel", "pay", "staff")}
            return Frame("aov", [P], filters=ff, answer_type="number")
        # ---------------------------------------------------------------- order count
        if _has(s, r"(bao nhieu|may|so|dem|tong cong|co|total|nhieu) (don|bill|hoa don|order)|\bso don\b|don hang") and not _has(s, r"tien|doanh thu|thu ve"):
            if not P:
                return None
            ff = {k_: v for k_, v in f.items() if k_ in ("shift", "channel", "pay", "cust_group", "staff", "group", "product")}
            if pkey:
                ff["product"] = pkey
            return Frame("order_count", [P], filters=ff, answer_type="number")
        # ---------------------------------------------------------------- units sold
        unit = self.lk.unit_after(s, ("so", "bao nhieu", "may", "nhieu", "duoc", "tong so", "ra", "tong cong")) or None
        if unit and _has(s, r"ban|ra|duoc|tieu thu") and not _has(s, r"tien|doanh thu"):
            if not P:
                return None
            ff = dict(f)
            if pkey:
                ff["product"] = pkey
                return Frame("units", [P], filters=ff, unit=unit, answer_type="number")
            if ("size" in ff or "brand" in ff) and "group" not in ff:
                g_ = self.lk.group_loose(s)
                if g_:
                    ff["group"] = g_
            if "size" in ff or "group" in ff or "brand" in ff:
                return Frame("units", [P], filters=ff, answer_type="number")
            if _has(s, r"cong moi|tinh moi|tat ca|cac mat hang|moi mat hang|moi mon|moi loai|tong cong|cong tat ca") or True:
                ff["base_unit"] = unit
                return Frame("units", [P], filters=ff, answer_type="number")
        if pkey and _has(s, r"ban duoc (?:tong cong )?(?:bao nhieu|may)|ban ra (?:bao nhieu|may)") and not _has(s, r"tien|doanh thu"):
            if not P:
                return None
            u = _unit_tail(s)
            ff = dict(f)
            ff["product"] = pkey
            return Frame("units", [P], filters=ff, unit=u, answer_type="number")
        if _has(s, r"khach o [a-z].* hay (?:khach )?o [a-z]|\bben nao\b|nhieu hon hay|hon hay khach"):
            return Frame("unsupported", reason="so sánh hai nhóm khách theo nơi ở chưa hỗ trợ")
        # ---------------------------------------------------------------- revenue split by channel / payment
        multi = _multi_filters(s)
        if len(multi) >= 2 and _has(s, REV):
            if not P:
                return None
            return Frame("revenue_multi", [P], params={"parts": multi}, answer_type="list")
        # ---------------------------------------------------------------- revenue (default)
        if _has(s, REV) or _has(s, r"ban duoc|ban dc|thu duoc|bao nhieu tien|ban ra"):
            if not P:
                return None
            ff = dict(f)
            if pkey:
                ff["product"] = pkey
            elif not any(ff.get(k_) is not None for k_ in ("group", "ingredient", "rx", "cust_group", "channel", "pay", "shift", "staff", "street", "size", "brand")) \
                    and _product_slot(ti.text_wo_time):
                return Frame("unsupported", reason="câu hỏi nêu một mặt hàng không tìm thấy trong dữ liệu")
            if _has(s, r"khach") and not ff.get("cust_group"):
                cs = self.lk.customers(q)
                if cs and len(cs[0][1]) == 1:
                    ff["customer"] = cs[0][1][0]
                elif _has(s, r"khach (nu|nam)\b|gioi tinh|khach o duong|duong |dia chi|quan \d"):
                    return Frame("unanswerable", reason="Dữ liệu không ghi giới tính/địa chỉ khách", answer_type="unanswerable")
            if _has(s, r"nhan vien") and not ff.get("staff"):
                return None
            return Frame("revenue", [P], filters=ff, answer_type="number")
        return None


_MULTI = [
    ("channel", "phone", r"dien thoai|goi dien"), ("channel", "zalo", r"zalo"),
    ("channel", "store", r"tai quay|tai cua hang|tai tiem|tai quan"),
    ("channel", "ecommerce", r"san tmdt|san thuong mai|shopee|lazada"), ("channel", "facebook", r"facebook"),
    ("pay", "transfer", r"chuyen khoan"), ("pay", "cash", r"tien mat|\btm\b"), ("pay", "cod", r"\bcod\b"),
    ("pay", "ewallet", r"vi dien tu"), ("pay", "debt", r"ghi no|mua chiu"),
]


def _multi_filters(s: str) -> List[dict]:
    hits = []
    for kind, val, rx in _MULTI:
        m = re.search(rx, s)
        if m:
            hits.append((m.start(), {kind: val}))
    hits.sort(key=lambda x: x[0])
    kinds = {list(h[1])[0] for h in hits}
    if len(hits) >= 2 and len(kinds) == 1:
        return [h[1] for h in hits]
    return []


_SLOT_STOP = set("""cua hang tiem quan shop minh toi cho hoi xem giup gium check tong cong doanh thu thu ve duoc bao nhieu
tien la vay nhi a trong ban ra nay do the ha z ta roi ca cac moi don hang hoa chiec cai""".split())


def _product_slot(t: str) -> bool:
    """True when a revenue question names something to sell ('bán <X> được', 'tiền bán <X>', 'mặt hàng <X>',
    'doanh thu của <X>') that no linker resolved."""
    t = " " + t + " "
    for rx in (r" ban (.+?) (?:duoc|dc|thu ve|thu duoc)\b", r" tien ban (.+?) (?:la|thang|duoc|$)", r" mat hang (.+?) (?:thu ve|ban|la|$)",
               r" doanh thu (?:cua|cho) (.+?) (?:trong|la|thang|$)"):
        m = re.search(rx, t)
        if m:
            words = [w for w in m.group(1).split() if w not in _SLOT_STOP and not w.isdigit()]
            if len(words) >= 1:
                return True
    return False


def _weekday(w: str) -> int:
    w = w.replace("thu ", "")
    return {"hai": 0, "2": 0, "ba": 1, "3": 1, "tu": 2, "4": 2, "nam": 3, "5": 3, "sau": 4, "6": 4, "bay": 5, "7": 5,
            "chu nhat": 6, "cn": 6}.get(w, 6)


def _unit_tail(s: str) -> Optional[str]:
    m = re.search(r"(?:bao nhieu|may|bn|nhieu)\s+([a-z0-9]+)\s*(?:$|\?|chan|le|trong|vay|nhi|ha|z|a|do|the)", s.strip())
    return m.group(1) if m else None


def _months_list(ti: TimeInfo) -> List[Period]:
    out = []
    for p in ti.periods:
        if p.kind == "month_range":
            y, m = p.start.year, p.start.month
            while (y, m) <= (p.end.year, p.end.month):
                out.append(month_period(y, m))
                y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        elif p.kind in ("month", "quarter", "mtd"):
            out.append(p)
    seen, res = set(), []
    for p in out:
        if (p.start, p.end) not in seen:
            seen.add((p.start, p.end))
            res.append(p)
    return res
