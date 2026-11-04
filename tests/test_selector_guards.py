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


def test_dissent_only_vetoes_when_the_llm_is_right_more_often_than_wrong_there():
    cands = [g1(34), llm("G3_rewrite", 99)]
    s = sel()
    a, _ = s.select(cands)
    assert a.value == 34                                      # no measured dissent reliability: 0.5, no veto
    s.rel = dict(REL, G3_rewrite=dict(REL["G3_rewrite"], __dissent__=0.15))
    assert s.select(cands)[0].value == 34                     # LLM usually wrong when it contradicts G1
    s.rel = dict(REL, G1_rule={"__all__": 0.8, "units": 0.8}, G3_rewrite=dict(REL["G3_rewrite"], __dissent__=0.9))
    assert s.select(cands)[0].type == "abstain"               # LLM usually right there: the rule answer is withheld


def test_form_guard_checks_llm_answers_against_the_original_question():
    from datamate.select.form_guard import apply_guards
    q = "Tháng 6/2026 điện thoại cũ bán ra mấy máy, chiếm bao nhiêu phần trăm?"
    cands = [Candidate("G3_rewrite", Answer("number", 259870000), "revenue"),
             Candidate("G4_base", Answer("percent", 12.5), None, {"sql": "select 1", "support": 2})]
    assert apply_guards(q, cands) == 1
    assert "percent asked" in cands[0].answer.meta["conflict"] and not cands[1].answer.meta.get("conflict")
    assert apply_guards(q, cands) == 0                        # idempotent


def test_unknown_rulebook_row_leaves_only_the_sql_route():
    from datamate.rulebook import concept_tokens, mentions, new_rows, parse_rulebook
    from datamate.select.form_guard import apply_guards
    rb = parse_rulebook("| Luật | Nghĩa |\n|---|---|\n| Khách chăm chỉ nhất | khách có nhiều ngày mua nhất |\n"
                        "| Doanh thu một đơn | tổng thành tiền |\n")
    rows = new_rows(rb, known={"doanh thu mot don"})
    assert [r.key for r in rows] == ["khach cham chi nhat"]
    toks = concept_tokens(rows[0].key)
    assert mentions("khach nao cham chi nhat thang 5", toks) and not mentions("khach mua nhieu nhat", toks)
    cands = [g1(5), llm("G2_frame", None, "unanswerable"),
             Candidate("G4_cte", Answer("string", "Lan"), None, {"sql": "select 1", "family": "m", "support": 2})]
    apply_guards("Khách nào chăm chỉ nhất tháng 5?", cands, ["Khách chăm chỉ nhất"])
    assert cands[0].answer.meta["conflict"].startswith("rulebook row") and cands[1].answer.meta.get("conflict")
    assert not cands[2].answer.meta.get("conflict")
    s = sel()
    s.rel = dict(REL, G4_cte={"__all__": 0.9})
    assert s.select(cands)[0].value == "Lan"                  # the rule answer is no anchor any more


def test_known_rulebook_rows_cover_train_and_public():
    import json
    from datamate.rulebook import new_rows, parse_rulebook
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    known = set(json.load(open(os.path.join(root, "artifacts", "known_rulebook_rows.json"), encoding="utf-8"))["rows"])
    for split in ("train", "public_test"):
        p = os.path.join(root, "data", split, "rulebook.md")
        if os.path.exists(p):
            assert new_rows(parse_rulebook(open(p, encoding="utf-8").read()), known) == []


# ---- after the 0.909 run: cases read from its public logs (no labels used) ----

def _q(gen, typ, v, intent=None, fam="qwen", **meta):
    det = {"family": fam}
    if gen.startswith("G4"):
        det.update(sql="select " + str(v) + " -- " + gen, support=3)
    return Candidate(gen, Answer(typ, v, meta=meta), intent, det)


def test_an_agreeing_flagged_route_does_not_withhold_the_rule_answer():
    # public "Th4/2026 máy mới chiếm mấy % tiền bán máy?": G1 77.26 (intent never seen on train: r 0.7), G3 the same
    # value but flagged (r 0.3), G4 73.25.  Summing logits gave p 0.5 and the correct rule answer was withheld.
    cands = [Candidate("G1_rule", Answer("percent", 77.26), "phone_rev_share"),
             _q("G3_rewrite", "percent", 77.26, "phone_rev_share", conflict="group named but not resolved"),
             _q("G4_base", "percent", 73.25), _q("G4_cte", "percent", 73.25), _q("G4_alias", "percent", 73.25)]
    a, info = sel().select(cands)
    assert a.value == 77.26 and info["p"] == 0.7
    weak_only = [g1(5, oov=["x"]), _q("G3_rewrite", "number", 5, "units", conflict="x")]
    assert sel().select(weak_only)[0].type == "abstain"     # two weak votes are still no answer


def test_zero_unrecorded_hour_and_count_guards():
    from datamate.select.form_guard import apply_guards

    def flag(q, c):
        apply_guards(q, [c])
        return c.answer.meta.get("conflict") or ""

    assert "zero result" in flag("Trong tháng 1/2026, bao nhiêu xe Air Blade khác nhau vào tiệm?",
                                 _q("G4_base", "number", 0))
    assert not flag("7 ngày gần nhất, cửa hàng có mấy mặt hàng ế hẳn, ko ai mua cái nào?", _q("G4_base", "number", 0))
    assert not flag("So số đĩa bán ra 7 ngày sau ngày tăng giá với 7 ngày trước, tăng hay giảm bao nhiêu %?",
                    _q("G4_base", "percent", 0.0))
    assert not flag("Trong tháng 1/2026, bao nhiêu xe Air Blade khác nhau vào tiệm?", g1(0))   # rule answers untouched
    assert "quality not recorded" in flag("Trong tháng 11/2025, sản phẩm nào ngon nhất?",
                                          _q("G4_alias", "string", "Rau muống xào tỏi"))
    assert not flag("Trong tháng 11/2025, sản phẩm nào ngon nhất?", _q("G2_frame", "unanswerable", None, "unanswerable"))
    assert not flag("Dép nào bán chạy nhất tháng 5?", _q("G4_alias", "string", "Dép tổ ong"))
    assert "hour asked" in flag("Tiệm mình thường đông khách nhất dào mấy giờ z?", _q("G2_frame", "date", "2026-05-02",
                                                                                     "best_day"))
    q = "Th3/2026 trung bình mỗi thứ Tư tiệm xửa mấy nượt xe?"
    assert "count asked" in flag(q, _q("G3_rewrite", "number", 200916, "aov"))
    assert not flag(q, _q("G3_rewrite", "number", 15, "repair_job_count"))


def test_solo_reliability_caps_the_llm_only_where_the_rule_is_unsure():
    s = sel()
    s.rel = dict(REL, G2_frame=dict(REL["G2_frame"], __solo__=0.55))
    assert sel().select([llm("G2_frame", 7)])[0].value == 7                # never measured: no cap (old files)
    assert s.select([llm("G2_frame", 7)])[0].type == "abstain"             # measured 0.55 alone: below tau
    assert s.select([g1(5, oov=["x"]), llm("G2_frame", 7)])[0].type == "abstain"
    assert s.select([g1(7), llm("G2_frame", 7)])[0].value == 7             # clean rule answer: no cap needed


def test_fit_measures_solo_reliability_where_g1_is_unsure():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import validate
    ref = lambda i, v: {"id": i, "type": "number", "answer": v, "alias": []}       # noqa: E731
    recs = [  # q0: clean G1, G2 and G4 agree; q1: G1 silent, G2 wrong, G4 right; q2: G1 flagged, G3 right
        {"id": "q0", "ref": ref("q0", 1), "cands": [g1(1), llm("G2_frame", 1), _q("G4_base", "number", 1)]},
        {"id": "q1", "ref": ref("q1", 2), "cands": [llm("G2_frame", 9), _q("G4_base", "number", 2)]},
        {"id": "q2", "ref": ref("q2", 3), "cands": [g1(3, oov=["x"]), llm("G3_rewrite", 3)]},
    ]
    rel = validate.fit(recs)
    assert rel["G2_frame"]["__solo__"] == round(5 / 11, 4)                 # q1 only (q0 had a clean G1): 0 of 1
    assert rel["G3_rewrite"]["__solo__"] == round(6 / 11, 4)               # q2: 1 of 1
    assert rel["G4_base"]["__solo__"] == round(7 / 12, 4)                  # SQL: q0 and q1 both count, 2 of 2
    assert "__solo__" not in rel["G1_rule"]
