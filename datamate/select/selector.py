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

Three guards, added after the first GPU run on public (0.777, below rule-only) showed the LLM routes
overriding correct rule answers:

* one vote per model: G2, G3, the G4 formats and G_select all come from the same LLM, so their errors are
  correlated; they form one family (``Candidate.family``) and agreeing with each other adds no vote;
* anchor: a *clean* rule answer (G1, no out-of-vocabulary / form-conflict / tie flag, rulebook row unchanged)
  cannot be replaced by an LLM answer.  A disagreeing LLM still lowers its probability, so the outcome is the
  rule answer or abstention (0 instead of risking -1).  Reliabilities are measured on train, where the LLM
  agrees with G1 97-99 % of the time, so they say little about the LLM when it disagrees;
* an ``unanswerable`` claim made only by the LLM is not eligible while G1 has a value answer;
* a group supported only by SQL needs ``min_sql_support`` executions (distinct SQL texts or prompts) that
  return the same value (XiYan's execution consistency); a single SQL gives no positive vote.
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
    "G4_base": 0.55,      # SQL generators (one model, three formats): uncalibrated until GPU logs exist
    "G4_cte": 0.55,
    "G4_alias": 0.55,
    "G_select": 0.60,     # selection model's pick among SQL candidates
    "G5_answerability": 0.70,
}


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


LLM_GEN_PREFIXES = ("G2_", "G3_", "G4_", "G5_", "G_select")


def is_llm_gen(gen: str) -> bool:
    return gen.startswith(LLM_GEN_PREFIXES)


@dataclass
class Candidate:
    gen: str
    answer: Answer
    intent: Optional[str] = None
    detail: Dict = field(default_factory=dict)

    @property
    def family(self) -> str:
        """Model identity: candidates of one family (every route of the same LLM) are one vote.
        Logs written before the family was recorded fall back to one shared LLM family."""
        return self.detail.get("family") or ("llm" if is_llm_gen(self.gen) else self.gen)


def _sql_support(members: List[Candidate]) -> int:
    """Executions behind a group: distinct SQL candidates (plus prompts that produced the same SQL).
    G_select re-picks one of them, so it adds no execution."""
    return sum(int(c.detail.get("support") or 1) for c in members if c.gen.startswith("G4"))


class Selector:
    def __init__(self, reliability_path: Optional[str] = None, tau: float = 0.6, lam: float = 0.6,
                 oov_cap: float = 0.3, anchor_gens=("G1_rule",), min_sql_support: int = 2, legacy: bool = False):
        self.tau = tau
        self.lam = lam
        self.oov_cap = oov_cap
        self.unseen_prior = 0.7
        # legacy=True reproduces the selector of the first GPU run (families by generator name, no anchor,
        # no SQL-support rule) so that logged runs can be replayed exactly (scripts/replay_selector.py)
        self.legacy = legacy
        self.anchor_gens = () if legacy else tuple(anchor_gens or ())
        self.min_sql_support = 0 if legacy else min_sql_support
        self.rel: Dict[str, Dict[str, float]] = {}
        if reliability_path and os.path.exists(reliability_path):
            with open(reliability_path, encoding="utf-8") as fh:
                self.rel = json.load(fh)

    def _fam(self, c: Candidate) -> str:
        return (c.detail.get("family") or c.gen) if self.legacy else c.family

    def _anchor(self, votes: List[Candidate], trusted=None) -> Optional[Candidate]:
        """A rule answer with no risk signal: LLM answers may lower its confidence but not replace it."""
        for c in votes:
            if c.gen not in self.anchor_gens:
                continue
            m = c.answer.meta
            if m.get("oov") or m.get("conflict") or m.get("tie"):
                continue
            if trusted is not None and c.intent and trusted.get(c.intent) is False:
                continue
            if _sanity_penalty(c.answer) < 0:
                continue
            return c
        return None

    def _family_votes(self, members: List[Candidate], trusted=None) -> Dict[str, float]:
        """family -> reliability of its vote for this group (its most reliable member speaks for it)."""
        out: Dict[str, float] = {}
        by_fam: Dict[str, List[Candidate]] = defaultdict(list)
        for c in members:
            by_fam[self._fam(c)].append(c)
        for f, cs in by_fam.items():
            r = max(self._r_eff(c, trusted) for c in cs)
            if self.min_sql_support and all("sql" in c.detail for c in cs) and _sql_support(cs) < self.min_sql_support:
                r = min(r, 0.5)                    # one SQL execution alone: no positive vote
            out[f] = r
        return out

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
        votes_by_group = {key: self._family_votes(members, trusted) for key, members in groups.items()}
        anchor = self._anchor(votes, trusted)
        rule_value = any(c.gen in self.anchor_gens and c.answer.type != "unanswerable" for c in votes)
        info = {"groups": []}
        if anchor is not None:
            info["anchor"] = anchor.gen
        best, best_p = None, -1.0
        for key, members in groups.items():
            fams = votes_by_group[key]
            s = sum(logit(r) for r in fams.values())
            # every other family that voted for a different answer counts against this group (once, its strongest vote)
            dissent: Dict[str, float] = {}
            for k2, f2 in votes_by_group.items():
                if k2 == key:
                    continue
                for f, r in f2.items():
                    if f not in fams:
                        dissent[f] = max(dissent.get(f, 0.0), r)
            s -= self.lam * sum(max(0.0, logit(r)) for r in dissent.values())
            s += _sanity_penalty(members[0].answer)
            p = sigmoid(s)
            llm_only = not any(c.gen in self.anchor_gens for c in members)
            eligible = True
            if anchor is not None and key != anchor.answer.key():
                eligible = False                          # an LLM answer does not replace a clean rule answer
            elif llm_only and rule_value and members[0].answer.type == "unanswerable":
                eligible = False                          # "unanswerable" from the LLM alone vs a rule value answer
            g = {"key": key[:200], "gens": sorted({c.gen for c in members}), "families": sorted(fams), "p": round(p, 4)}
            if not eligible:
                g["eligible"] = False
            info["groups"].append(g)
            if eligible and p > best_p:
                best, best_p = members[0].answer, p
        if best is None:
            return Answer("abstain", reason="no eligible candidate"), info
        info["p"] = round(best_p, 4)
        if best_p < self.tau:
            return Answer("abstain", reason=f"low confidence {best_p:.2f}"), info
        return best, info

    def _r_eff(self, c: Candidate, trusted=None) -> float:
        """Reliability of one candidate after its risk signals."""
        r = self.r(c.gen, _ikey(c))
        if trusted is not None and c.intent and trusted.get(c.intent) is False and c.gen == "G1_rule":
            r = min(r, 0.35)
        if c.answer.meta.get("tie"):
            r = min(r, 0.3)
        if c.answer.meta.get("oov") or c.answer.meta.get("conflict"):
            # the question mentions concepts the parse ignored (unseen business vocabulary)
            r = min(r, self.oov_cap)
        flags = c.detail.get("flags") or []
        if flags:
            from .sql_checks import penalty_factor
            r *= penalty_factor(flags)
        return r


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
