"""Question-form vs. frame consistency (a cheap verifier in the spirit of XiYan's selector).

The rule parser falls back to broad intents ('revenue', 'units', ...) when a phrasing is new; the
answer then *looks* fine but answers a different question.  The surface form of a Vietnamese
question constrains its answer type strongly ('... nào' asks for a name, '%' for a percent,
'lần lượt' for a list, '... không?' for yes/no), so a frame whose answer type contradicts those cues
is not trusted (selective prediction: abstain instead of a confident wrong answer).

Returns a short reason string, or None when the frame is consistent with the question.
"""
from __future__ import annotations

import re
from typing import Optional

_WHICH = re.compile(r"\b(?:(?:mat )?hang|san pham|sp|mon|mat hang|khach|khach hang|nhan vien|nv|tho|ngay|thang|"
                    r"nha cung cap|ncc|dong(?: may| xe)?|hang|loai|nhom|ca|kenh|size|so|lo|xe|may|bua|hom|tuan|"
                    r"quy|gio|thu)\s+(?:nao|mo|gi|chi)\b|\bla ai\b|\bai la\b|\bai\b(?= (?:ban|mua|lam|no|chi|nhap))")
_PCT = re.compile(r"%|phan tram|\bti le\b|\bty le\b|\bti trong\b|\bty trong\b|\bchiem\b")
_LIST = re.compile(r"\blan luot\b|\btung thang\b|\bmoi thang (?:bao nhieu|bn|duoc)|\btung (?:ben|kenh|nhom|nguoi)\b|"
                   r"\bmoi (?:kenh|ben|nhom|nguoi) (?:bao nhieu|bn|duoc)|\bliet ke\b|\bke (?:giup|ra|lan luot)\b|"
                   r"\bnhung (?:mat hang|san pham|mon|sp)\b|\btheo thu tu\b")
_YESNO = re.compile(r"(?:phai|dung|dung vay|dung khong|co) (?:khong|ko|hong|hok|k)\b|\bdung khong\b|\bco phai\b|"
                    r"(?:khong|ko|hong|hok)\s*\??\s*$|\bhay khong\b")
_DAYCOUNT = re.compile(r"\b(?:may|bao nhieu|bn|bnhiu|so) (?:bua|ngay|hom)\b")
SCALAR = {"number", "percent"}
_NONZERO = {"revenue", "units", "order_count", "aov", "phone_units", "purchased_qty", "delivered_qty", "purchase_total",
            "share", "phone_pay_total", "phone_pay_avg", "repair_job_count", "delivery_trips", "gross_profit"}


def check_form(nq: str, ans, intent: Optional[str] = None) -> Optional[str]:
    """The checks that need only the question and the answer (no frame): usable for every generator, SQL included.
    Callers pass the ORIGINAL question: an LLM rewrite can change the form of the question it then answers."""
    if ans is None or ans.type in ("abstain", "unanswerable"):
        return None
    s = " " + nq + " "
    t = ans.type
    if ans.value is None or (isinstance(ans.value, (list, dict)) and not ans.value):
        return "empty value"
    # '%' asked, scalar money/count returned (percent questions are answered by share/growth/rate intents)
    if _PCT.search(s) and t == "number" and not re.search(r"\b(?:tien|bao nhieu tien|so tien)\b.*\b(?:giam|bot|chiet khau) \d", s):
        if not re.search(r"\d\s*(?:%|phan tram)", s):  # '%' in a product name / threshold ('kem 5%', 'giam 10%')
            return "percent asked, number returned"
    # 'which X' asked, a scalar returned
    if t in SCALAR and _WHICH.search(s) and not re.search(r"\b(?:may|bao nhieu|bn|bnhiu) (?:gio|h)\b", s):
        if not re.search(r"\b(?:bao nhieu|bn|bnhiu|may) (?:mat hang|san pham|sp|mon|khach|don|ngay|nguoi|lo|xe|phieu|luot)\b", s):
            return "which-question, scalar returned"
    # list asked, single scalar returned
    if t in SCALAR and _LIST.search(s) and intent not in ("debt_flow",):
        return "list asked, scalar returned"
    # yes/no asked
    if t in SCALAR and _YESNO.search(s) and re.search(r"cao hon|thap hon|tang|giam|nhieu hon|it hon|lon hon", s):
        return "yes/no asked, scalar returned"
    return None


def check(nq: str, fr, ans) -> Optional[str]:
    """nq: normalised (folded) question; fr: Frame; ans: Answer."""
    if fr is None or ans is None or ans.type in ("abstain", "unanswerable"):
        return None
    s = " " + nq + " "
    t = ans.type
    it = fr.intent
    why = check_form(nq, ans, it)
    if why:
        return why
    # questions are generated about things that happened: a zero sum / count means the filter or period is wrong
    if t in SCALAR and ans.value == 0 and it in _NONZERO:
        return "zero result"
    # several distinct months mentioned, single scalar that is not a comparison
    months = {(p.start, p.end) for p in (fr.periods or [])}
    named_item = bool((getattr(fr, "filters", None) or {}).get("product"))  # 'T8' inside a product name
    if t in SCALAR and len(months) <= 1 and it not in ("growth", "price_effect") and not named_item and \
            len(set(re.findall(r"\bthang (\d{1,2})\b", s))) >= 2 and \
            not re.search(r"so (?:voi )?|\bvs\b|hon|kem|chenh|tang|giam|cung ky|tu thang|den (?:het )?thang|toi (?:het )?thang", s):
        return "several months asked, single period used"
    # 'nhóm X' named but no group-like filter resolved (shop without a group column, or an unknown group name)
    filt = getattr(fr, "filters", None) or {}
    mg = re.search(r"\bnhom (?:hang |mat hang |san pham )?([a-z]+)", s)
    if mg and not named_item and mg.group(1) not in ("tuoi", "khach", "no") and not mg.group(1).isdigit() and \
            not any(filt.get(k) for k in ("group", "cust_group", "ingredient", "size", "brand")) and \
            not (getattr(fr, "params", None) or {}).get("base_group") and it not in ("debt_aging", "unanswerable"):
        return "group named but not resolved"
    # 'how many days' asked, money returned
    if it in ("revenue", "aov", "units") and _DAYCOUNT.search(s):
        return "day count asked"
    return None
