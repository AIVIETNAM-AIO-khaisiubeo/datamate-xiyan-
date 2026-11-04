"""Selector guards added after the first GPU run on public (0.777 < rule-only): the cases seen there, in miniature."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datamate.engine.answer import Answer  # noqa: E402
from datamate.select.logs import candidates_from_log  # noqa: E402
from datamate.select.selector import Candidate, Selector  # noqa: E402

REL = {"G1_rule": {"__all__": 0.985, "units": 0.985}, "G2_frame": {"__all__": 0.976, "units": 0.976,
                                                                    "unanswerable": 0.909},
       "G3_rewrite": {"__all__": 0.985, "units": 0.985}, "G4_base": {"__all__": 0.9}, "G4_cte": {"__all__": 0.9}}


def sel(**kw):
    s = Selector(None, tau=0.6, **kw)
    s.rel = REL
    return s


def g1(v, **meta):
    return Candidate("G1_rule", Answer("number", v, meta=meta), "units")


def llm(gen, v, typ="number", intent="units"):
    return Candidate(gen, Answer(typ, v), intent if typ != "unanswerable" else "unanswerable")


def test_two_routes_of_one_llm_do_not_outvote_a_clean_rule_answer():
    # public: "Điện thoại cũ ... bán ra mấy máy": G1 34, G2 and G3 (same LLM) 259870000
    cands = [g1(34), llm("G2_frame", 259870000), llm("G3_rewrite", 259870000)]
    a_old, _ = sel(legacy=True).select(cands)
    assert a_old.value == 259870000                           # the first GPU run: two LLM votes beat G1
    a, info = sel().select(cands)
    assert a.value == 34 or a.type == "abstain"               # now: the rule answer, or nothing
    assert info.get("anchor") == "G1_rule"
    assert any(g.get("eligible") is False for g in info["groups"])


def test_llm_fills_in_when_rule_is_silent_or_flagged():
    a, _ = sel().select([llm("G2_frame", 7), llm("G3_rewrite", 7)])
    assert a.value == 7
    a, _ = sel().select([g1(5, oov=["sộp"]), llm("G3_rewrite", 7)])
    assert a.value == 7                                       # flagged rule answer: the LLM may answer instead


def test_llm_unanswerable_does_not_replace_a_rule_value():
    s = sel()
    a, _ = s.select([g1(5, oov=["x"]), llm("G2_frame", None, "unanswerable")])
    assert a.type != "unanswerable"                           # flagged G1 value vs LLM-only "unanswerable"
    a, _ = s.select([g1(5), llm("G2_frame", None, "unanswerable")])
    assert a.type in ("number", "abstain")
    a, _ = s.select([llm("G2_frame", None, "unanswerable")])
    assert a.type == "unanswerable"                           # nothing else: the claim stands (if confident)


def test_a_single_sql_execution_gives_no_vote():
    one = [Candidate("G4_base", Answer("number", 9), None, {"sql": "select 9", "family": "m", "support": 1})]
    assert sel().select(one)[0].type == "abstain"
    two = one + [Candidate("G4_cte", Answer("number", 9), None, {"sql": "with t as (select 9) select * from t",
                                                                   "family": "m", "support": 1})]
    assert sel().select(two)[0].value == 9                    # two different SQLs, same result
    same_sql_twice = [Candidate("G4_base", Answer("number", 9), None, {"sql": "select 9", "family": "m", "support": 2})]
    assert sel().select(same_sql_twice)[0].value == 9         # one SQL produced by two prompts (schema/format)


def test_rule_only_runs_are_unchanged():
    for cands in ([g1(5)], [g1(5, oov=["x"])], [g1(5), g1(6)], []):
        a_new, i_new = sel().select(cands)
        a_old, i_old = sel(legacy=True).select(cands)
        assert a_new.to_submission("q") == a_old.to_submission("q") and i_new.get("p") == i_old.get("p")


def test_old_logs_are_rebuilt():
    lg = {"candidates": [{"gen": "G1_rule", "intent": "units", "answer": {"type": "number", "answer": 5}},
                         {"gen": "G3_rewrite", "intent": "units", "answer": {"type": "number", "answer": 5}},
                         {"gen": "G4_base", "answer": {"type": "number", "answer": 5}, "sql": "select 5"}],
          "g2_json": {"intent": "unanswerable", "reason": "không có cột"},
          "sql_candidates": [{"sql": "select 5", "also": ["S2/G4_base", "full/G4_base"]}]}
    cs = candidates_from_log(lg)
    by = {c.gen: c for c in cs}
    assert by["G2_frame"].answer.type == "unanswerable"     # was missing from "candidates" in old logs
    assert by["G4_base"].detail["support"] == 3
    assert by["G3_rewrite"].family == by["G4_base"].family == by["G2_frame"].family == "llm"
    assert by["G1_rule"].family == "G1_rule"


def test_extract_sql_handles_the_fine_tuned_output():
    from datamate.llm.generators import extract_sql
    sql = "SELECT SUM(revenue) FROM orders WHERE is_valid = 1"
    assert extract_sql(sql + "\n```") == sql                   # LoRA: the prompt already opened ```sql
    assert extract_sql("```sql\n" + sql + ";\n```") == sql
    assert extract_sql("Câu SQL:\n```\n" + sql + "\n```\nGiải thích ...") == sql
    assert extract_sql("WITH t AS (SELECT 1 AS x)\nSELECT x FROM t\n```") == "WITH t AS (SELECT 1 AS x)\nSELECT x FROM t"
    assert extract_sql("không biết") is None
