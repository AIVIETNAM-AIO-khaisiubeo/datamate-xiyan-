"""Rebuild selector candidates from a run's per-question logs (validate.py, replay_selector.py).

Logs written by the first GPU runs miss a few fields the selector now reads; they are reconstructed here:
  * G2 "unanswerable" votes were not listed under "candidates" (only in "g2_json"),
  * the SQL support (prompts that produced the same SQL) is only in "sql_candidates[*].also",
  * families were not logged: every LLM generator then falls back to one shared family (Candidate.family).
"""
from __future__ import annotations

from typing import Dict, List

from ..engine.answer import Answer
from .selector import Candidate

DETAIL_KEYS = ("flags", "family", "support", "sql")


def candidates_from_log(lg: Dict, guards: bool = True) -> List[Candidate]:
    """Candidates of a logged question, with the same risk flags the agent sets before selecting (form_guard)."""
    cands = _candidates(lg)
    if guards:
        from .form_guard import apply_guards
        apply_guards(lg.get("question", ""), cands, lg.get("new_rulebook_rows") or [])
    return cands


def _candidates(lg: Dict) -> List[Candidate]:
    support_by_sql = {}
    for sc in lg.get("sql_candidates") or []:
        if sc.get("sql"):
            support_by_sql[sc["sql"]] = 1 + len(sc.get("also") or [])
    cands = []
    for c in lg.get("candidates", []):
        a = c.get("answer") or {}
        ans = Answer(a.get("type", "abstain"), a.get("answer"), reason=a.get("reason", ""), meta=dict(c.get("meta") or {}))
        det = {k: c[k] for k in DETAIL_KEYS if c.get(k)}
        if "sql" in det and "support" not in det and c.get("gen", "").startswith("G4"):
            det["support"] = support_by_sql.get(det["sql"], 1)
        cands.append(Candidate(c["gen"], ans, c.get("intent"), det))
    js = lg.get("g2_json")
    if isinstance(js, dict) and js.get("intent") == "unanswerable" and not any(c.gen == "G2_frame" for c in cands):
        cands.append(Candidate("G2_frame", Answer("unanswerable", reason=js.get("reason") or ""), "unanswerable", {}))
    return cands


def trusted_from_rulebook(rulebook_text: str, cands: List[Candidate]) -> Dict[str, bool]:
    """Same 'trusted' map the agent passes to the selector: intents whose rulebook row changed are not trusted."""
    from ..rulebook import parse_rulebook
    rb = parse_rulebook(rulebook_text or "")
    return {c.intent: rb.intent_trusted(c.intent) for c in cands if c.gen == "G1_rule" and c.intent}
