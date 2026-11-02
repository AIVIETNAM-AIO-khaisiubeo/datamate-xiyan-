"""Measure the SQL content checks (datamate/select/sql_checks.py) on TRAIN, without any LLM.

* false-alarm rate: verified-correct silver SQL that the checks flag (should be ~0)
* detection rate:  controlled near-miss mutations (sql_mutations.py) that change the answer and are flagged

    python scripts/eval_sql_checks.py data/train
"""
import collections
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATAMATE_CACHE", os.path.join(ROOT, ".cache"))

from datamate.ingest.builder import load_biz  # noqa: E402
from datamate.llm.generators import sql_result_to_value  # noqa: E402
from datamate.llm.mschema import build_views  # noqa: E402
from datamate.llm.sql_mutations import mutate_all  # noqa: E402
from datamate.llm.sql_pipeline import SQLRunner  # noqa: E402
from datamate.nlu.rules import _topk  # noqa: E402
from datamate.nlu.timeparse import LunarCalendar, parse_time  # noqa: E402
from datamate.scorer import score_one  # noqa: E402
from datamate.select.sql_checks import lint  # noqa: E402
from datamate.textnorm import normalize_question  # noqa: E402


def main(split, silver=os.path.join(ROOT, "artifacts", "sql_silver.jsonl")):
    cal = LunarCalendar(os.path.join(split, "calendar.csv"))
    items = [json.loads(l) for l in open(silver, encoding="utf-8")]
    by = collections.defaultdict(list)
    for it in items:
        by[it["business_id"]].append(it)
    pos = [0, 0]                                          # [flagged, total] on correct SQL
    neg = collections.defaultdict(lambda: [0, 0])          # mutation -> [flagged, total answer-changing]
    fp_examples = collections.Counter()
    for bid, lst in sorted(by.items()):
        b = load_biz(os.path.join(split, "data", bid), bid)
        runner = SQLRunner(build_views(b))
        for it in lst:
            today = dt.date.fromisoformat(it["as_of_date"])
            nq = normalize_question(it["question"])
            per = parse_time(nq, today, cal).periods
            ref = {"type": it["answer_type"], "answer": it["answer"]}
            for fmt, sql in it["sql"].items():
                rows = runner.run(sql)
                fl = lint(sql, nq, per, today, it["answer_type"], _topk(nq), len(rows))
                pos[1] += 1
                if fl:
                    pos[0] += 1
                    fp_examples.update(f for f, _ in fl)
            base = it["sql"].get("base") or next(iter(it["sql"].values()))
            for name, m in mutate_all(base):
                try:
                    rows = runner.run(m)
                    v = sql_result_to_value(rows, it["answer_type"])
                except Exception:  # noqa: BLE001
                    continue
                if score_one({"type": it["answer_type"], "answer": v}, ref) == 1:
                    continue                                # mutation did not change the answer
                fl = lint(m, nq, per, today, it["answer_type"], _topk(nq), len(rows))
                neg[name][1] += 1
                neg[name][0] += bool(fl)
        runner.close()
        print(bid, flush=True)
    print(f"\ncorrect SQL flagged (false alarms): {pos[0]}/{pos[1]} = {pos[0] / max(1, pos[1]):.3%}  {dict(fp_examples)}")
    print(f"{'mutation':24s} detected / answer-changing")
    for name, (f, n) in sorted(neg.items()):
        print(f"{name:24s} {f:5d} / {n:5d}  ({f / max(1, n):.0%})")


if __name__ == "__main__":
    main(*sys.argv[1:])
