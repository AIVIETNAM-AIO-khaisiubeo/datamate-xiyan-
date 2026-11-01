"""Service-ticket books (e.g. motorbike repair): one row = one ticket (lượt xe / phiếu).

Activated purely from the data shape (a sales-like table with vehicle / work / mechanic roles),
never from a business id.  Implements the generic parts of the 'Sửa xe' rulebook section:
tickets incl. warranty tickets, paid tickets, warranty rate, mechanic rankings & commission,
distinct vehicles (plates) per model, repeat vehicles.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..textnorm import fold, fold_key

PLATE = re.compile(r"(\d{2}\s*[A-Z]{1,2}\d?\s*-\s*\d{3}\.?\d{2})", re.I)
MODEL_ALIASES = {"ab": "airblade", "airblade": "airblade", "ex": "exciter", "exciter": "exciter",
                 "alpha": "wavealpha", "wavealpha": "wavealpha", "rsx": "waversx", "waversx": "waversx",
                 "shmode": "shmode", "sh": "sh", "lead": "lead", "vision": "vision", "future": "future",
                 "sirius": "sirius", "janus": "janus", "grande": "grande", "vario": "vario", "winner": "winner",
                 "winnerx": "winnerx", "blade": "blade", "jupiter": "jupiter", "nouvo": "nouvo", "pcx": "pcx"}


def _nick(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"^(a|anh|c|chi|chị|e|em|chu|chú|bac|bác)\s*[.\s]\s*", "", s, flags=re.I)
    return s.strip()


def enrich(biz, L: pd.DataFrame) -> pd.DataFrame:
    if "vehicle" not in L.columns:
        return L
    plates, models = [], []
    for v in L["vehicle"]:
        v = "" if v is None else str(v).replace("α", " alpha")   # 'Wave α' = 'Wave Alpha'
        m = PLATE.search(v)
        plates.append(re.sub(r"\s", "", m.group(1)).upper() if m else None)
        raw = (v[:m.start()] + v[m.end():]).strip() if m else v.strip()
        models.append(raw)
    L["plate"] = plates
    L["model_raw"] = models
    L["model_key"] = [MODEL_ALIASES.get(fold_key(m), fold_key(m)) if m else None for m in models]
    note = L["note"] if "note" in L.columns else pd.Series([None] * len(L), index=L.index)
    amt = pd.to_numeric(L.get("amount"), errors="coerce") if "amount" in L.columns else pd.Series([None] * len(L))
    L["warranty"] = [bool(re.search(r"bao hanh|\bbh\b", fold(str(n)))) if n is not None else False for n in note]
    # mechanic nickname -> full name from the staff list
    full = {}
    if not biz.staff.empty and "staff_name" in biz.staff.columns:
        for n in biz.staff["staff_name"].dropna():
            full.setdefault(fold_key(str(n).split()[-1]), str(n).strip())
    if "mechanic" in L.columns:
        mech = []
        for m in L["mechanic"]:
            if m is None or str(m).strip() == "":
                mech.append(None)
                continue
            nk = _nick(m)
            mech.append(full.get(fold_key(nk.split()[-1]) if nk else "", nk))
        L["staff"] = mech
    biz.flags["repair"] = True
    disp = {}
    for k, raw in zip(L["model_key"], L["model_raw"]):
        if k:
            disp.setdefault(k, Counter())[raw] += 1
    biz._model_name = {k: c.most_common(1)[0][0] for k, c in disp.items()}
    return L


ORDER_FIELDS = ("plate", "model_key", "model_raw", "warranty", "work")
DOMAIN_VOCAB = set("""tho phieu luot xe bao hanh hoa hong hue lanh lam mang toi vo zo sua chiec khac nhau quen lan
khach dong bien so nhan duoc tiem quay lai thu tien ra dem nhieu nhat cao man thay nhot lap dau ruot va dien
den bong ac quy duong rua kim phun noi bo thang phanh sen nhong xich loc gio bugi lop curoa nuoc mat may
truoc sau co""".split())

# Job taxonomy of the free-text 'Làm gì' cell; first match wins (order matters).  Rulebook: plain 'nhớt'
# is engine oil (not gear oil 'nhớt láp'), plain 'ruột' is a new tube (not a patch 'vá'), 'sửa điện'
# excludes bulbs and batteries.
_JOBS = [("nhot_lap", r"nhot lap|dau lap|\blap\b|nhot hop so|nhot so\b"), ("nhot_may", r"nhot|motul|castrol|shell|dau may"),
         ("va_ruot", r"\bva(?: ruot| xe| lop| vo| sam)?\b"), ("thay_ruot", r"\bruot\b"), ("sua_dien", r"sua dien|lam dien"),
         ("den", r"\bden\b"), ("ac_quy", r"ac quy|\bbinh\b"), ("bao_duong", r"bao duong|\bbd\b|\bbdtq\b"),
         ("kim_phun", r"(?:\brua |ve sinh |\bvs )?kim phun"), ("rua_xe", r"\brua\b"), ("noi", r"\bnoi\b"),
         ("sen", r"(?:\bbo )?(?:nhong|xich|\bsen\b)|\bnsd\b"),
         ("phanh", r"\bbo\b|phanh|\bthang\b"), ("loc_gio", r"loc gio"), ("bugi", r"bu ?gi"),
         ("curoa", r"cu ?roa"), ("vo", r"\bvo\b|\blop\b"), ("nuoc_mat", r"nuoc (?:lam )?mat")]
_QJOBS = [("nhot_lap", r"nhot lap|dau lap|nhot hop so|\blap\b"), ("va_ruot", r"\bva\b"),
          ("thay_ruot", r"\bruot\b"), ("nhot_may", r"\bnhot\b|dau may"), ("sua_dien", r"sua dien|lam dien"),
          ("den", r"bong den|thay den|den pha"), ("ac_quy", r"ac quy"), ("bao_duong", r"bao duong"),
          ("kim_phun", r"kim phun"), ("rua_xe", r"rua xe"), ("noi", r"\bnoi\b"), ("sen", r"\bsen\b|nhong|xich"),
          ("phanh", r"bo thang|ma phanh|phanh|\bbo (?:truoc|sau)\b"), ("loc_gio", r"loc gio"), ("bugi", r"bu ?gi"),
          ("curoa", r"cu ?roa"), ("vo", r"thay vo|vo xe|\blop\b"), ("nuoc_mat", r"nuoc (?:lam )?mat")]


def job_items(cell) -> List[str]:
    """Jobs done on one ticket ('Làm gì' cell); patterns consume their text so 'nhớt láp' is not also 'nhớt'."""
    out = []
    if cell is None or (isinstance(cell, float) and cell != cell):
        return out
    for part in re.split(r"[,;+/&]", fold(str(cell))):
        w = " " + part + " "
        for job, rx in _JOBS:
            if re.search(rx, w):
                out.append(job)
                w = re.sub(rx, " ", w)
    return out


def question_job(s: str) -> Optional[str]:
    for job, rx in _QJOBS:
        if re.search(rx, s):
            return job
    return None


# ---------------------------------------------------------------------------- parsing
def parse(rp, s: str, q: str, ti, P) -> Optional["Frame"]:
    from ..nlu.frame import Frame
    b = rp.b
    if not b.flags.get("repair"):
        return None
    mech = _mechanic_in(rp, s)
    is_pct = bool(re.search(r"%|phan tram|ti le|chiem", s))
    if re.search(r"hoa hong|hue hong", s):
        if not P:
            return None
        if re.search(r"bang|moi tho|tung tho|cac tho|xep", s):
            return Frame("repair_commission_table", [P], answer_type="table")
        if re.search(r"tho (?:nao|mo|ai)|\bai\b", s) and re.search(r"cao nhat|nhieu nhat", s):
            return Frame("repair_top_commission", [P], answer_type="string")
        if mech:
            return Frame("repair_commission", [P], filters={"staff": mech}, answer_type="number")
        return None
    if re.search(r"bao hanh", s):
        if not P:
            return None
        if is_pct:
            return Frame("repair_warranty_rate", [P], answer_type="percent")
        if re.search(r"khong tinh|tru|bo", s):
            pass
        elif re.search(r"(bao nhieu|may|so|dem|bn) (phieu|luot|xe|lan)|phieu bao hanh|luot bao hanh|quay lai bao hanh", s):
            return Frame("repair_warranty_count", [P], filters={"staff": mech} if mech else {}, answer_type="number")
    if re.search(r"khach quen|(?:tu|it nhat) (?:2|hai) lan|nhieu lan", s) and re.search(r"\bxe\b", s):
        if not P:
            return None
        return Frame("repair_repeat_vehicles", [P], params={"n": 2}, answer_type="number")
    mm = re.search(r"(?:bao nhieu|may|bn|so) (?:chiec xe|chiec|xe) ([a-z0-9 ]+?) (?:khac nhau|duoc|vao|toi|den|mang|zo|vo)", s)
    if mm and P:
        mk = MODEL_ALIASES.get(fold_key(mm.group(1)), fold_key(mm.group(1)))
        if mk in getattr(b, "_model_name", {}):
            return Frame("repair_distinct_vehicles", [P], params={"model": mk}, answer_type="number")
    if re.search(r"dong xe (?:nao|mo|gi)", s) and re.search(r"tien|doanh thu", s):
        if not P:
            return None
        return Frame("repair_top_model", [P], answer_type="string")
    if re.search(r"\btho\b", s) and re.search(r"(?:nao|mo|ai)\b", s) and re.search(r"nhat", s):
        if not P:
            return None
        if re.search(r"cham chi|nhieu ngay", s):
            by = "days"
        elif re.search(r"tien|doanh thu", s):
            by = "revenue"
        else:
            by = "tickets"
        return Frame("top_staff", [P], by=by, answer_type="string")
    if re.search(r"(?:trung binh|binh quan) (?:mot |1 |moi )?(?:phieu|luot)|(?:mot|1|moi) phieu .*(?:trung binh|binh quan)", s):
        if not P:
            return None
        return Frame("repair_avg_paid_ticket", [P], filters={"staff": mech} if mech else {}, answer_type="number")
    if re.search(r"(bao nhieu|may|so|dem|bn|tong|duoc) (?:phieu|luot|xe)|luot xe|phieu sua", s) and not re.search(
            r"tien|doanh thu", s):
        if not P:
            return None
        mw = re.search(r"\b(thu (?:hai|ba|tu|nam|sau|bay|[2-7])|chu nhat)\b", s)
        if mw and re.search(r"trung binh|binh quan", s):
            from ..nlu.rules import _weekday
            if question_job(s.replace("phieu sua", "")):
                return Frame("unsupported", reason="trung bình theo thứ của một loại việc chưa hỗ trợ")
            return Frame("avg_weekday", [P], params={"weekday": _weekday(mw.group(1)), "metric": "order_count"},
                         filters={"staff": mech} if mech else {}, answer_type="number")
        job = question_job(re.sub(r"phieu sua(?: xe)?|sua xe|xe vo sua|vao sua|zo sua|vo sua", " ", s))
        if job:
            qual = next((w for w in ("truoc", "sau") if job == "phanh" and re.search(r"\b" + w + r"\b", s)), None)
            return Frame("repair_job_count", [P], filters={"staff": mech} if mech else {},
                         params={"job": job, "qual": qual}, answer_type="number")
        if re.search(r"\bthay\b|\blam (?!duoc|dc|ra)", s.replace("phieu sua", "")):
            return Frame("unsupported", reason="loại việc chưa nhận diện được")
        return Frame("order_count", [P], filters={"staff": mech} if mech else {}, answer_type="number")
    if mech and re.search(r"tien|doanh thu|ban", s):
        if not P:
            return None
        return Frame("revenue", [P], filters={"staff": mech}, answer_type="number")
    return None


def _mechanic_in(rp, s: str) -> Optional[str]:
    names = set()
    L = rp.b.lines
    if "staff" in L.columns:
        names = {n for n in L["staff"].dropna().unique()}
    best = None
    for n in names:
        fn = " " + fold(n) + " "
        if fn in s and (best is None or len(n) > len(best)):
            best = n
    if best:
        return best
    for n in names:
        last = fold(str(n).split()[-1])
        if re.search(r"\btho " + re.escape(last) + r"\b", s):
            return n
    return None


# ---------------------------------------------------------------------------- execution
def run(fe, fr):
    from ..engine.answer import Answer
    from ..engine.executor import Unanswerable, money, r2
    ex = fe.ex
    it = fr.intent
    P = ex.clip(fr.periods[0]) if fr.periods else None
    O = ex.orders(P, True, {k: v for k, v in fr.filters.items() if k == "staff"})
    paid = O[~O["warranty"].astype(bool)] if "warranty" in O.columns else O
    if it == "repair_warranty_count":
        return Answer("number", int(O["warranty"].astype(bool).sum()))
    if it == "repair_warranty_rate":
        if len(paid) == 0:
            raise Unanswerable("không có phiếu thu tiền")
        return Answer("percent", r2(O["warranty"].astype(bool).sum() / len(paid) * 100))
    if it == "repair_avg_paid_ticket":
        if len(paid) == 0:
            raise Unanswerable("không có phiếu thu tiền")
        return Answer("number", money(paid["revenue"].sum() / len(paid)))
    if it == "repair_repeat_vehicles":
        c = paid.groupby("plate").size()
        return Answer("number", int((c >= fr.params.get("n", 2)).sum()))
    if it == "repair_job_count":
        job, qual = fr.params["job"], fr.params.get("qual")
        n = 0
        for w in paid.get("work", []):
            if job in job_items(w) and (not qual or re.search(r"\b" + qual + r"\b", fold(str(w)))):
                n += 1
        return Answer("number", n)
    if it == "repair_distinct_vehicles":
        return Answer("number", int(O.loc[O["model_key"] == fr.params["model"], "plate"].nunique()))
    if it == "repair_top_model":
        s = O.groupby("model_key")["revenue"].sum()
        if s.empty:
            raise Unanswerable("không có phiếu")
        mx = s.max()
        k = sorted(k for k, v in s.items() if v == mx)[0]
        return Answer("string", ex.b._model_name.get(k, k))
    comm = _commission_pct(ex.b)
    if it in ("repair_commission", "repair_commission_table", "repair_top_commission"):
        if not comm:
            raise Unanswerable("không có tỉ lệ hoa hồng")
        rev = O.groupby("staff")["revenue"].sum()
        rows = [[n, money(rev.get(n, 0.0) * pct / 100.0)] for n, pct in comm.items()]
        rows.sort(key=lambda r: (-r[1], r[0]))
        if it == "repair_commission":
            n = fr.filters["staff"]
            if n not in comm:
                raise Unanswerable("không có tỉ lệ hoa hồng của thợ")
            return Answer("number", money(rev.get(n, 0.0) * comm[n] / 100.0))
        if it == "repair_top_commission":
            return Answer("string", rows[0][0])
        return Answer("table", {"columns": ["tho", "tien_hoa_hong"], "rows": rows})
    raise KeyError(it)


def _commission_pct(b) -> Dict[str, float]:
    out = {}
    S = b.staff
    if not S.empty and "staff_name" in S.columns and "commission" in S.columns:
        for n, c in zip(S["staff_name"], S["commission"]):
            try:
                if n is not None and c is not None:
                    out[str(n).strip()] = float(c)
            except (TypeError, ValueError):
                pass
    return out
