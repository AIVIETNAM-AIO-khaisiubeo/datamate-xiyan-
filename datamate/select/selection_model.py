"""Candidate reorganisation + selection model (XiYan-SQL Algorithm 3).

    L  <- deformalize(L);  C <- groupby(L, execution result);  S_un <- union of filtered schemas
    if |C| = 1: return shortest SQL
    C'  <- clusters sorted by size (desc);  C'' <- each cluster sorted by generator order O
    if |C''_1| >= ceil(|L| / 2): present every candidate in C'' order       (strong consensus prior)
    else:                         present the shortest SQL of each cluster   (one per cluster, attention focus)
    l* <- M_c(Q, S_un, E, L')

Here M_c is the shared Qwen3-14B prompted with the XiYan selection template (a fine-tuned selector can be
plugged in through the same backend).  The choice becomes one more vote in the calibrated selector
(``Selector``), never a bypass of the abstention threshold.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from ..llm import prompts as P
from ..llm.backend import extract_json
from ..llm.sql_pipeline import SQLCandidate, deformalize


def _value_key(c: SQLCandidate) -> str:
    v = c.value
    if isinstance(v, float):
        v = round(v, 2)
    return f"{c.answer_type}:{v!r}"


def reorganize(cands: Sequence[SQLCandidate], order: Dict[str, float]) -> Tuple[List[SQLCandidate], str, int]:
    """Returns (ordered candidates to present, mode, size of the largest cluster)."""
    ok = [c for c in cands if c.ok]
    if not ok:
        return [], "none", 0
    clusters: "OrderedDict[str, List[SQLCandidate]]" = OrderedDict()
    for c in ok:
        clusters.setdefault(_value_key(c), []).append(c)
    if len(clusters) == 1:
        best = min(ok, key=lambda c: len(deformalize(c.sql)))
        return [best], "consistent", len(ok)
    cl = sorted(clusters.values(), key=lambda m: -len(m))
    cl = [sorted(m, key=lambda c: (-order.get(c.generator, 0.5), len(c.sql))) for m in cl]
    if len(cl[0]) >= math.ceil(len(ok) / 2):
        return [c for m in cl for c in m], "consensus", len(cl[0])
    return [min(m, key=lambda c: len(deformalize(c.sql))) for m in cl], "one_per_cluster", len(cl[0])


class SelectionModel:
    name = "G_select"

    def __init__(self, backend, max_candidates: int = 8):
        self.be = backend
        self.max_candidates = max_candidates

    def prompt(self, question: str, schema_union: str, evidence: str, shown: Sequence[SQLCandidate]) -> str:
        blocks = [P.candidate_block(i + 1, deformalize(c.sql), c.preview or "") for i, c in enumerate(shown)]
        return self.be.chat_prompt(P.SELECT_SYSTEM, P.select_user(question, schema_union, evidence, blocks))

    def choose(self, jobs: List[Dict]) -> Dict[str, Optional[SQLCandidate]]:
        """jobs: [{qid, question, schema_union, evidence, shown: [SQLCandidate]}] -> chosen candidate per qid."""
        todo = [j for j in jobs if len(j["shown"]) > 1]
        out = {j["qid"]: (j["shown"][0] if len(j["shown"]) == 1 else None) for j in jobs}
        if not todo or self.be is None:
            return out
        prompts = [self.prompt(j["question"], j["schema_union"], j["evidence"], j["shown"][:self.max_candidates]) for j in todo]
        texts = self.be.generate(prompts, n=1, temperature=0.0, max_tokens=200, json_schema=P.SELECT_SCHEMA, seed=17)
        for j, t in zip(todo, texts):
            js = extract_json(t[0]) if t else None
            try:
                i = int((js or {}).get("choice", 0)) - 1
            except (TypeError, ValueError):
                i = -1
            shown = j["shown"][:self.max_candidates]
            out[j["qid"]] = shown[i] if 0 <= i < len(shown) else None
        return out
