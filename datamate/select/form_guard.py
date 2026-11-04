"""Risk flags set on candidates just before selection (agent, validate.py and replay_selector.py all call this).

1. Question form vs answer, for every LLM candidate: "mấy máy" asks for a count, "% ..." for a percent, "... nào" for
   a name.  G2 / G3 answer an LLM *rewrite* of the question, and the rule checks inside the executor only see that
   rewrite; G4 answers with SQL and had no check at all.  The original question is checked here
   (datamate.nlu.consistency.check_form), so an LLM answer of the wrong form cannot fill in for the rule engine.
2. Rulebook rows the engine never saw (datamate.rulebook.new_rows): the routes that go through the rule executor
   (G1, G2, G3) cannot implement a concept they do not know, so their answers to a question naming such a row are
   flagged, "unanswerable" included (the concept is defined, so the question is answerable).  Only the SQL route,
   which gets the row as evidence, may answer; otherwise the agent abstains.
3. Zero from an LLM route.  Questions are asked about things that happened: on train 4 of 2,462 numeric answers are 0
   (all four about items that did not sell or a price change with no effect).  A zero count or sum from SQL almost
   always means a filter matched no row (on public: 'SH Mode', 'Air Blade', 'Wave Alpha' visits counted as 0 while
   the repair book lists 13-21 of each in the month).  The rule executor already flags zero results of its frames
   (consistency.check); this applies the same rule to SQL.
4. Qualities the books do not record ('ngon nhất', 'chất lượng tốt nhất', 'khó tính nhất'): no column measures taste,
   quality or temperament, so an LLM value answer to such a question is a guess (usually the best seller).

A flag is a 'conflict' in Answer.meta: the selector caps the candidate's reliability (oov_cap) and a flagged rule
answer is no anchor.  Flags are only added, never removed, so calling this twice changes nothing.  Guards 3 and 4 only
flag LLM candidates: they can make the agent abstain, never answer.
"""
from __future__ import annotations

import re
from typing import Iterable, List

from ..nlu.consistency import check_form
from ..textnorm import normalize_question

RULE_EXECUTOR_GENS = ("G1_rule", "G2_frame", "G3_rewrite")

# the question itself is about something absent: a zero is a plausible answer there
_ABSENCE = re.compile(r"\bkhong (?:ai|ban|co ai|co don)\b|\bban khong duoc\b|\bchua ban\b|\bhang e\b|\bmat hang e\b|"
                      r"\be han\b|\bkhong (?:thay doi|doi)\b")
_CHANGE = re.compile(r"\btang\b|\bgiam\b|\bso voi\b|\bso vs\b|\bchenh\b|\bthay doi\b|\bhon\b")
_UNRECORDED = re.compile(r"\bngon (?:nhat|hon)\b|\bmon ngon\b|\bchat luong (?:tot |cao |kem |te )?(?:nhat|hon)\b|"
                         r"\bchat luong tot\b|\b(?:kho|de) tinh (?:nhat|hon)\b|\bkho chieu (?:nhat|hon)\b|"
                         r"\bdep (?:nhat|hon)\b|\bde thuong (?:nhat|hon)\b|\bthan thien (?:nhat|hon)\b|"
                         r"\buy tin (?:nhat|hon)\b|\bhai long (?:nhat|hon)\b")


def _is_llm(gen: str) -> bool:
    from .selector import is_llm_gen
    return is_llm_gen(gen)


def zero_result(nq: str, ans) -> bool:
    """A numeric 0 where the question does not ask about something absent (or about a % change)."""
    if ans.type not in ("number", "percent") or ans.value is None or isinstance(ans.value, bool):
        return False
    try:
        if float(ans.value) != 0.0:
            return False
    except (TypeError, ValueError):
        return False
    s = " " + nq + " "
    if _ABSENCE.search(s):
        return False
    return not (ans.type == "percent" and _CHANGE.search(s))


def apply_guards(question: str, cands: List, new_rows: Iterable[str] = ()) -> int:
    """Flag candidates in place; returns how many were flagged."""
    new_rows = list(new_rows or ())
    nq = normalize_question(question or "")
    unrecorded = _UNRECORDED.search(" " + nq + " ")
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
        if not why and _is_llm(c.gen):
            if zero_result(nq, a):
                why = "zero result"
            elif unrecorded and a.type != "unanswerable":
                why = "quality not recorded: " + unrecorded.group(0).strip()
        if why:
            a.meta["conflict"] = "question form: " + why
            n += 1
    return n
