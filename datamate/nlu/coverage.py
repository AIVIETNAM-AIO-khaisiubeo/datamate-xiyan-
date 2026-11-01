"""Semantic-coverage check for the rule parser (out-of-distribution question detection).

A rule frame is only trusted when every content word of the question is *explained*:
by the vocabulary of the training questions, by a linked entity name of the shop, or by a
known filler / dialect word.  Words outside that set ("thợ", "hoa hồng", "trả góp", "bảo hành",
...) mean the question talks about a concept the deterministic engine does not model, so
the agent must not trust a parse that silently ignored it.  This is the abstention signal
that matters most on unseen business types.
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterable, List, Optional, Set

from ..textnorm import fold_words, normalize_question

FILLERS = set("""
a ah ak ad admin alo anh be bac chi chu cho co coi cko da day de dum do e em gium giup giu ha hai hen he
hoi hok ik jum ka khong kia la lam luon may minh mik mk mjk minh moi nao ne nghe nha nhe nhi nho oi ok
quan roi ru rua shop ta tam the thi tiem toi tui uh um va vay voi vs xem xin xiu z zo zay
ma mo ni nha nhe nhieu dzay hong check chec chek nek nhaa nhee nhoe ui uk ukm vang da
thanks thank cam on ban oi bé be chi anh em ad sep boss mi mk mik
""".split())

# general business-question words the rule parser understands even if no train question used them
GENERAL = set("""
xuong thap giua sanh chot thuong gio ua chuong chinh voucher ship diem trieu xong du mung xep dan lon nho
order sop dong khach nguong set muon som tre khung cao
""".split())


def build_vocab(questions: Iterable[str]) -> Set[str]:
    v = set()
    for q in questions:
        for t in normalize_question(q).split():
            if re.fullmatch(r"[a-z]+", t):
                v.add(t)
    return v


def save_vocab(v: Set[str], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sorted(v), fh, ensure_ascii=False)


def load_vocab(path: str) -> Optional[Set[str]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return set(json.load(fh))


def unexplained_tokens(nq: str, vocab: Set[str], entity_texts: Iterable[str]) -> List[str]:
    """Content tokens of the normalised question that nothing explains."""
    ent = set()
    for e in entity_texts:
        ent |= set(fold_words(e).split())
    out = []
    for t in nq.split():
        if not re.fullmatch(r"[a-z]+", t) or len(t) == 1:
            continue
        if t in vocab or t in ent or t in FILLERS or t in GENERAL:
            continue
        out.append(t)
    return out
