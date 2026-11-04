"""End-to-end smoke test of the full multi-generator agent with a mock LLM.

Needs the competition data (skipped otherwise):  DATAMATE_TRAIN=data/train python -m pytest -q tests/test_agent_mock.py
"""
import json
import os
import re
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN = os.environ.get("DATAMATE_TRAIN", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train"))


@pytest.mark.skipif(not os.path.exists(os.path.join(TRAIN, "questions.jsonl")), reason="train split not available")
def test_agent_with_mock_llm():
    from datamate.agent import DataMateAgent
    from datamate.llm import backend as B
    from datamate.llm.generators import FrameGenerator, RewriteGenerator, SQLGenerator

    def fake(prompt: str) -> str:
        if "Viết lại:" in prompt:
            m = re.findall(r"Câu gốc: (.*)\nViết lại:", prompt)
            return m[-1] if m else ""
        if "JSON:" in prompt:
            return json.dumps({"intent": "revenue", "time": "hôm qua"})
        if prompt.rstrip().endswith("```sql") or "```sql<" in prompt[-80:]:
            # the prompt opened the fence: answer the way the fine-tuned model does (SQL, then the closing fence)
            return "SELECT SUM(revenue) FROM orders WHERE is_valid = 1\n```"
        return "```sql\nSELECT SUM(revenue) FROM orders WHERE is_valid = 1\n```"

    qs = [json.loads(l) for l in open(os.path.join(TRAIN, "questions.jsonl"), encoding="utf-8")]
    bid = sorted({q["business_id"] for q in qs})[0]
    sub = [q for q in qs if q["business_id"] == bid][:15]
    agent = DataMateAgent(TRAIN, {"llm": {"backend": "mock"}, "g4_scope": "all"})
    mock = B.MockBackend(fake)
    agent._llm_ready = True
    agent.backend = mock
    agent.g3 = RewriteGenerator(mock, None)
    agent.g2 = FrameGenerator(mock, None, "")
    agent.g4 = SQLGenerator(mock, None, "")
    from datamate.llm.sql_pipeline import MultiSQLGenerator
    from datamate.select.selection_model import SelectionModel
    agent.mgen = MultiSQLGenerator(mock, None)
    agent.selmodel = SelectionModel(mock)
    with tempfile.TemporaryDirectory() as d:
        res = agent.run(sub, os.path.join(d, "sub.jsonl"), os.path.join(d, "logs"))
        assert len(res) == len(sub)
        lines = open(os.path.join(d, "sub.jsonl"), encoding="utf-8").read().splitlines()
        assert len(lines) == len(sub)
        log = json.load(open(os.path.join(d, "logs", sub[0]["id"] + ".json"), encoding="utf-8"))
        assert "final" in log and "candidates" in log and "elapsed_total_s" in log
        # every LLM route is logged with the backend's family, and replaying the logs reproduces the run
        from datamate.select.logs import candidates_from_log, trusted_from_rulebook
        fams, n_sql_ok = set(), 0
        for q in sub:
            lg = json.load(open(os.path.join(d, "logs", q["id"] + ".json"), encoding="utf-8"))
            fams |= {c.get("family") for c in lg["candidates"] if c["gen"] != "G1_rule"}
            n_sql_ok += sum(1 for c in lg.get("sql_candidates") or [] if c.get("ok"))
            cands = candidates_from_log(lg)
            ans, _ = agent.selector.select(cands, trusted_from_rulebook(agent.rulebook_text, cands))
            assert ans.to_submission(q["id"]) == lg["final"], q["id"]
        assert None not in fams and len(fams) <= 1
        assert n_sql_ok > 0                      # SQL written the fine-tuned way (no opening fence) executes
