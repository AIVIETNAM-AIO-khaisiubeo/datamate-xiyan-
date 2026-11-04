"""Risk flags set on candidates just before selection (agent, validate.py and replay_selector.py all call this).

1. Question form vs answer, for every LLM candidate: "mấy máy" asks for a count, "% ..." for a percent, "... nào" for
   a name.  G2 / G3 answer an LLM *rewrite* of the question, and the rule checks inside the executor only see that
   rewrite; G4 answers with SQL and had no check at all.  The original question is checked here
   (datamate.nlu.consistency.check_form), so an LLM answer of the wrong form cannot fill in for the rule engine.
2. Rulebook rows the engine never saw (datamate.rulebook.new_rows): the routes that go through the rule executor
   (G1, G2, G3) cannot implement a concept they do not know, so their answers to a question naming such a row are
   flagged, "unanswerable" included (the concept is defined, so the question is answerable).  Only the SQL route,
   which gets the row as evidence, may answer; otherwise the agent abstains.

A flag is a 'conflict' in Answer.meta: the selector caps the candidate's reliability (oov_cap) and a flagged rule
answer is no anchor.  Flags are only added, never removed, so calling this twice changes nothing.
"""
from __future__ import annotations

from typing import Iterable, List

from ..nlu.consistency import check_form
from ..textnorm import normalize_question

RULE_EXECUTOR_GENS = ("G1_rule", "G2_frame", "G3_rewrite")


def apply_guards(question: str, cands: List, new_rows: Iterable[str] = ()) -> int:
    """Flag candidates in place; returns how many were flagged."""
    new_rows = list(new_rows or ())
    nq = normalize_question(question or "")
    n = 0
    for c in cands:
        a = c.answer
        if a.type == "abstain" or a.meta.get("conflict"):
            continue
        if new_rows and c.gen in RULE_EXECUTOR_GENS:
            a.meta["conflict"] = "rulebook row not implemented: " + "; ".join(new_rows)
            n += 1
            continue
        if c.gen == "G1_rule":
            continue                                   # already checked against the original question by the agent
        why = check_form(nq, a, c.intent)
        if why:
            a.meta["conflict"] = "question form: " + why
            n += 1
    return n
