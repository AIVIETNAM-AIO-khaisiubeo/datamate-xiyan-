"""Candidate reorganisation + selection + abstention.

XiYan-SQL groups SQL candidates by execution result and lets a selection model pick among
the re-ordered groups.  Here every generator's *executed answer* is a vote; votes are grouped
by canonical answer value and each group gets a calibrated probability of being correct:

    logit P(group) = Σ_{v in group} logit(r_v)  -  λ · Σ_{v not in group} logit⁺(r_v)  + penalties

where r_v is the reliability of generator v on that intent (measured on the training split).
Following Chow's rule for the +1 / 0 / −1 metric, the agent answers only if P > τ (default
0.6, a safety margin over the theoretical 0.5), otherwise it abstains.  This mirrors the
abstention-as-a-decision view of "When Silence is Golden": abstaining is the correct action
whenever the expected reward of answering is negative.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..engine.answer import Answer

DEFAULT_RELIABILITY = {
    "G1_rule": 0.97,
    "G3_rewrite": 0.80,
    "G2_frame": 0.72,
    "G4_sql": 0.55,
    "G5_answerability": 0.70,
}


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class Candidate:
    gen: str
    answer: Answer
    intent: Optional[str] = None
    detail: Dict = field(default_factory=dict)


class Selector:
    def __init__(self, reliability_path: Optional[str] = None, tau: float = 0.6, lam: float = 0.6,
                 oov_cap: float = 0.3):
        self.tau = tau
        self.lam = lam
        self.oov_cap = oov_cap
        self.unseen_prior = 0.7
        self.rel: Dict[str, Dict[str, float]] = {}
        if reliability_path and os.path.exists(reliability_path):
            with open(reliability_path, encoding="utf-8") as fh:
                self.rel = json.load(fh)

    def r(self, gen: str, intent: Optional[str]) -> float:
        g = self.rel.get(gen, {})
        if intent and intent in g:
            return g[intent]
        if "__all__" in g:
            # an intent never validated on train: implemented from the rulebook text only
            return min(g["__all__"], self.unseen_prior)
        return DEFAULT_RELIABILITY.get(gen, 0.5)

    def select(self, cands: List[Candidate], trusted: Dict[str, bool] = None) -> (Answer, Dict):
        votes = [c for c in cands if c.answer.type != "abstain"]
        if not votes:
            return Answer("abstain"), {"reason": "no candidate"}
        groups: Dict[str, List[Candidate]] = defaultdict(list)
        for c in votes:
            groups[c.answer.key()].append(c)
        info = {"groups": []}
        best, best_p = None, -1.0
        for key, members in groups.items():
            s = 0.0
            seen = set()
            for c in members:
                if c.gen in seen:
                    continue  # several samples of one generator count once (self-consistency handled upstream)
                seen.add(c.gen)
                r = self.r(c.gen, _ikey(c))
                if trusted is not None and c.intent and trusted.get(c.intent) is False and c.gen == "G1_rule":
                    r = min(r, 0.35)
                if c.answer.meta.get("tie"):
                    r = min(r, 0.3)
                if c.answer.meta.get("oov") or c.answer.meta.get("conflict"):
                    # the question mentions concepts the parse ignored (unseen business vocabulary)
                    r = min(r, self.oov_cap)
                s += logit(r)
            others = set()
            for k2, m2 in groups.items():
                if k2 == key:
                    continue
                for c in m2:
                    if c.gen in others or c.gen in seen:
                        continue
                    others.add(c.gen)
                    s -= self.lam * max(0.0, logit(self.r(c.gen, _ikey(c))))
            s += _sanity_penalty(members[0].answer)
            p = sigmoid(s)
            info["groups"].append({"key": key[:200], "gens": sorted(seen), "p": round(p, 4)})
            if p > best_p:
                best, best_p = members[0].answer, p
        info["p"] = round(best_p, 4)
        if best_p < self.tau:
            return Answer("abstain", reason=f"low confidence {best_p:.2f}"), info
        return best, info


def _ikey(c: Candidate) -> Optional[str]:
    if c.answer.type == "unanswerable" and c.intent and c.intent != "unanswerable":
        return "unanswerable@" + c.intent
    return c.intent


def _sanity_penalty(a: Answer) -> float:
    v = a.value
    if a.type == "number" and isinstance(v, (int, float)) and v < 0:
        return -0.5
    if a.type == "percent" and isinstance(v, (int, float)) and abs(v) > 1000:
        return -1.5
    if a.type == "list" and isinstance(v, list) and len(v) == 0:
        return -3.0
    if a.type in ("string",) and (v is None or v == ""):
        return -3.0
    return 0.0
