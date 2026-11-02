"""Evidence E for the SQL pipeline (XiYan's "external knowledge": business rules, synonyms).

In DataMate the evidence of a question is: today's date, the rulebook rows that concern it (retrieved by
character n-gram TF-IDF over the split's rulebook.md table rows) and the shop README conventions that
mention the same words.  Kept short so that it fits next to the M-Schema in every prompt.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

from ..textnorm import fold, normalize_question


def _grams(s: str) -> Counter:
    s = " " + re.sub(r"\d+", "#", fold(s)) + " "
    return Counter(s[i:i + 4] for i in range(len(s) - 3))


def rulebook_rows(text: str) -> List[str]:
    rows, section = [], ""
    for line in text.splitlines():
        t = line.strip()
        if t.startswith("#"):
            section = t.lstrip("# ").strip()
            continue
        if not t.startswith("|") or t.startswith("|---") or re.match(r"\|\s*(Luật|Cách nói)\s*\|", t):
            continue
        cells = [c.strip().replace("**", "") for c in t.strip("|").split("|")]
        if len(cells) >= 2 and cells[0]:
            rows.append(f"[{section}] {cells[0]}: {' | '.join(cells[1:])}")
    return rows


class EvidenceRetriever:
    def __init__(self, rulebook_text: str, readme_text: str = ""):
        self.rows = rulebook_rows(rulebook_text)
        self.readme = [l.strip("-* ").strip() for l in readme_text.splitlines()
                       if l.strip().startswith("- **") and len(l) < 400]
        docs = self.rows + self.readme
        self.vecs = [_grams(d) for d in docs]
        df = Counter()
        for v in self.vecs:
            df.update(set(v))
        n = max(1, len(docs))
        self.idf = {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}
        self.docs = docs

    def _cos(self, a: Counter, b: Counter) -> float:
        dot = sum(c * b.get(g, 0) * self.idf.get(g, 1.0) ** 2 for g, c in a.items())
        na = math.sqrt(sum((c * self.idf.get(g, 1.0)) ** 2 for g, c in a.items())) or 1.0
        nb = math.sqrt(sum((c * self.idf.get(g, 1.0)) ** 2 for g, c in b.items())) or 1.0
        return dot / (na * nb)

    def evidence(self, question: str, as_of: str, k: int = 4) -> str:
        v = _grams(normalize_question(question))
        sc = sorted(((self._cos(v, u), i) for i, u in enumerate(self.vecs)), reverse=True)
        picked = [self.docs[i] for s, i in sc[:k] if s > 0.05]
        return "\n".join([f"Hôm nay là {as_of}; mọi mốc thời gian tương đối tính theo ngày này, bản ghi sau hôm nay coi như chưa xảy ra."]
                         + [f"- {p}" for p in picked])
