"""Noise augmentation that imitates real Vietnamese chat questions.

Used for (a) robustness evaluation of the parser without labels (augment a labelled train
question, the answer must not change) and (b) building SFT data for the LLM generators.
"""
from __future__ import annotations

import random
import re
import unicodedata
from typing import Callable, List

from .textnorm import fold

_TELEX_TONE = {"́": "s", "̀": "f", "̉": "r", "̃": "x", "̣": "j"}
_TELEX_VOWEL = {"â": "aa", "ă": "aw", "ê": "ee", "ô": "oo", "ơ": "ow", "ư": "uw", "đ": "dd"}

ABBR = [
    (r"\bdoanh thu\b", ["dthu", "doanh thu", "dt", "revenue"]),
    (r"\bbao nhiêu\b", ["bnhiu", "bn", "bao nhiu", "nhiêu", "bnhiêu", "bao nhiêu"]),
    (r"\bhôm nay\b", ["hnay", "hôm nay", "bữa nay"]),
    (r"\bhôm qua\b", ["hqua", "hôm qua", "bữa qua"]),
    (r"\bđược\b", ["đc", "dc", "được", "đươc"]),
    (r"\bkhông\b", ["ko", "k", "hông", "không"]),
    (r"\bsản phẩm\b", ["sp", "sản phẩm"]),
    (r"\bmặt hàng\b", ["mặt hàng", "mh", "món"]),
    (r"\btrước\b", ["trc", "trước"]),
    (r"\bvậy\b", ["z", "v", "vậy", "rứa"]),
    (r"\bnào\b", ["nào", "mô"]),
    (r"\bnhiều\b", ["nhìu", "nhiều", "nhiu"]),
    (r"\bmình\b", ["mik", "mk", "mình"]),
    (r"\bgiúp\b", ["júp", "giúp", "jup"]),
    (r"\bgiùm\b", ["jùm", "giùm"]),
    (r"\bcho\b", ["cko", "cho"]),
    (r"\bnhất\b", ["nhất", "nhứt"]),
    (r"\bnày\b", ["này", "ni"]),
    (r"\bđơn hàng\b", ["đh", "đơn hàng", "đơn", "bill"]),
    (r"\btổng\b", ["tổng", "total"]),
]

FILLERS_HEAD = ["", "", "", "ê ", "e ơi ", "ad ơi ", "cho mình hỏi ", "check giùm ", "xem júp ", "alo "]
FILLERS_TAIL = ["", "", "", " z", " vậy", " nhỉ", " hè", " rứa", " nha", " ạ", " ta", " 🤔"]


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s).replace("đ", "d").replace("Đ", "D")


def telexify_word(w: str) -> str:
    """'tiền' -> 'tieenf' (simulates typing without IME)."""
    d = unicodedata.normalize("NFD", w)
    tone = ""
    base = []
    for ch in d:
        if ch in _TELEX_TONE:
            tone = _TELEX_TONE[ch]
        else:
            base.append(ch)
    b = unicodedata.normalize("NFC", "".join(base))
    out = "".join(_TELEX_VOWEL.get(c, c) for c in b)
    return out + tone


def date_shorthand(s: str, rng: random.Random) -> str:
    def mon(m):
        mo, yr = m.group(1), m.group(2)
        forms = [f"tháng {mo}/{yr}", f"Th{mo}/{yr}", f"T{mo}/{yr}", f"{mo}-{yr}", f"{int(mo):02d}/{yr}", f"{mo}/{yr[2:]}",
                 f"th{mo}/{yr}", f"tháng {mo} năm {yr}"]
        return rng.choice(forms)
    s = re.sub(r"tháng (\d{1,2})/(\d{4})", mon, s)
    s = re.sub(r"quý (\d)/(\d{4})", lambda m: rng.choice([f"quý {m.group(1)}/{m.group(2)}", f"Q{m.group(1)}/{m.group(2)}",
                                                          f"Q{m.group(1)}/{m.group(2)[2:]}"]), s)
    return s


def augment(q: str, rng: random.Random, level: float = 0.5) -> str:
    s = q
    if rng.random() < level:
        s = date_shorthand(s, rng)
    for rx, forms in ABBR:
        if re.search(rx, s, flags=re.I) and rng.random() < level:
            s = re.sub(rx, rng.choice(forms), s, count=1, flags=re.I)
    if rng.random() < level * 0.5:
        words = s.split()
        if words:
            i = rng.randrange(len(words))
            if re.search(r"[À-ỹđ]", words[i]) and not re.search(r"\d", words[i]):
                words[i] = telexify_word(words[i])
            s = " ".join(words)
    if rng.random() < level * 0.35:
        s = strip_accents(s)
    if rng.random() < level * 0.4:
        s = s.lower()
    s = rng.choice(FILLERS_HEAD) + s.rstrip("?.!") + rng.choice(FILLERS_TAIL) + rng.choice(["?", "", "", "?"])
    return re.sub(r"\s+", " ", s).strip()
