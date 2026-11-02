"""End-to-end test of the XiYan SQL path (Schema Filter -> Algorithm 2 -> Algorithm 3) with an oracle mock LLM.

The mock answers NL2SQL prompts with the execution-verified train SQL in the base format and with a
near-miss mutation (no is_valid filter) in the CTE format, so the candidates disagree: the SQL checks must
flag the wrong one, every candidate must carry its provenance, and the selector must not prefer it.
Needs the train split (skipped otherwise).
"""
import json
import os
import re
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TRAIN = os.environ.get("DATAMATE_TRAIN", os.path.join(ROOT, "data", "train"))
SILVER = os.path.join(ROOT, "artifacts", "sql_silver.jsonl")


@pytest.mark.skipif(not (os.path.exists(os.path.join(TRAIN, "questions.jsonl")) and os.path.exists(SILVER)),
                    reason="train split / silver SQL not available")
def test_xiyan_sql_path_with_oracle_mock():
    from datamate.agent import DataMateAgent
    from datamate.llm import backend as B
    from datamate.llm import prompts as P
    from datamate.llm.sql_mutations import drop_is_valid
    from datamate.llm.sql_pipeline import MultiSQLGenerator
    from datamate.scorer import score_one
    from datamate.select.selection_model import SelectionModel

    silver = [json.loads(l) for l in open(SILVER, encoding="utf-8")]
    bid = "biz_101"
    items = [s for s in silver if s["business_id"] == bid and s["intent"] in ("revenue", "order_count", "aov")][:6]
    assert items
    by_q = {s["question"]: s for s in items}

    def fake(prompt: str) -> str:
        if "【Câu hỏi】" in prompt and prompt.rstrip().endswith("```sql") and "【SQL】" not in prompt:
            q = re.findall(r"【Câu hỏi】\n(.*?)\n", prompt)[-1]
            sql = by_q[q]["sql"]["base"]
            if P.SQL_FORMAT_HINTS["cte"] in prompt:
                sql = drop_is_valid(sql) or sql
            return f"```sql\n{sql}\n```"
        if "Ứng viên 1" in prompt:
            return '{"choice": 1}'
        return "{}"

    qs = [{"id": s["id"], "business_id": bid, "question": s["question"], "as_of_date": s["as_of_date"]} for s in items]
    agent = DataMateAgent(TRAIN, {"llm": {"backend": "mock"}, "g4_scope": "all", "use_g2": False, "use_g3": False})
    mock = B.MockBackend(fake)
    agent._llm_ready, agent.backend = True, mock
    agent.mgen = MultiSQLGenerator(mock, None)
    agent.selmodel = SelectionModel(mock)
    refs = {s["id"]: {"type": s["answer_type"], "answer": s["answer"]} for s in items}
    with tempfile.TemporaryDirectory() as d:
        res = agent.run(qs, os.path.join(d, "sub.jsonl"), os.path.join(d, "logs"))
        for q in qs:
            log = json.load(open(os.path.join(d, "logs", q["id"] + ".json"), encoding="utf-8"))
            sc = log["sql_candidates"]
            assert log["schema_filter"]["schemas"], "schema filter produced no schema"
            produced = {c["schema_id"] for c in sc} | {a.split("/")[0] for c in sc for a in c["also"]}
            assert "full" in produced and produced & {"S1", "S2"}
            for c in sc:                                      # provenance of every candidate
                for k in ("sql", "schema_id", "generator", "fmt", "family", "ok", "flags", "t_gen_s", "prompt_tokens"):
                    assert k in c
            good = [c for c in sc if c["fmt"] == "base" and c["ok"]]
            bad = [c for c in sc if c["fmt"] == "cte" and c["ok"]]
            assert good and all(not c["flags"] for c in good)
            if bad and any(c["value"] != good[0]["value"] for c in bad):
                assert all(any(f[0] == "missing_is_valid" for f in c["flags"]) for c in bad)
            assert score_one(res[q["id"]], refs[q["id"]]) == 1
