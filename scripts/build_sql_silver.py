"""Build execution-verified "silver" SQL for the TRAIN split (train labels only).

For each train question: rule frame -> answer must equal the gold answer -> compile the frame to SQL in the
three XiYan formats (base / cte / alias) -> execute on the canonical views -> keep the formats whose result
equals the gold answer.  Output: artifacts/sql_silver.jsonl (one line per question with >= 1 verified SQL).

    python scripts/build_sql_silver.py data/train [artifacts/sql_silver.jsonl]
"""
import collections
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATAMATE_CACHE", os.path.join(ROOT, ".cache"))

from datamate.engine.answer import FrameExecutor  # noqa: E402
from datamate.engine.executor import Executor  # noqa: E402
from datamate.ingest.builder import load_biz  # noqa: E402
from datamate.llm.frame2sql import FORMATS, compile_frame, tables_columns  # noqa: E402
from datamate.llm.generators import run_sql, sql_result_to_value  # noqa: E402
from datamate.llm.mschema import build_views  # noqa: E402
from datamate.nlu.rules import RuleParser  # noqa: E402
from datamate.nlu.timeparse import LunarCalendar  # noqa: E402
from datamate.scorer import score_one  # noqa: E402


def main(split, out=os.path.join(ROOT, "artifacts", "sql_silver.jsonl")):
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(split, "answers.jsonl"), encoding="utf-8")}
    cal = LunarCalendar(os.path.join(split, "calendar.csv"))
    by = collections.defaultdict(list)
    for q in qs:
        by[q["business_id"]].append(q)
    stat = collections.defaultdict(lambda: [0, 0, 0])   # intent -> [frames ok, compiled, verified]
    n = 0
    with open(out, "w", encoding="utf-8") as fo:
        for bid, items in sorted(by.items()):
            b = load_biz(os.path.join(split, "data", bid), bid)
            rp = RuleParser(b, cal)
            V = build_views(b)
            for q in items:
                today = dt.date.fromisoformat(q["as_of_date"])
                fr = rp.parse(q["question"], today)
                if fr is None:
                    continue
                ex = Executor(b, today)
                a = FrameExecutor(ex, cal).run(fr)
                ref = refs[q["id"]]
                if score_one(a.to_submission(q["id"]), ref) != 1:
                    continue
                st = stat[fr.intent]
                st[0] += 1
                sqls, ok = {}, {}
                for fmt in FORMATS:
                    c = compile_frame(fr, ex, fmt)
                    if c is None:
                        continue
                    sql, at = c
                    sqls[fmt] = sql
                    try:
                        v = sql_result_to_value(run_sql(sql, V), at)
                        ok[fmt] = score_one({"id": q["id"], "type": at, "answer": v}, ref) == 1
                    except Exception as e:  # noqa: BLE001
                        ok[fmt] = False
                if sqls:
                    st[1] += 1
                if any(ok.values()):
                    st[2] += 1
                    good = {k: v for k, v in sqls.items() if ok.get(k)}
                    rec = {"id": q["id"], "business_id": bid, "question": q["question"], "as_of_date": q["as_of_date"],
                           "intent": fr.intent, "answer_type": ref["type"], "answer": ref.get("answer"), "sql": good,
                           "columns": tables_columns(next(iter(good.values())), V)}
                    fo.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    n += 1
            print(bid, n, flush=True)
    tot = [sum(v[i] for v in stat.values()) for i in range(3)]
    print(f"{'intent':22s} frames compiled verified")
    for it, (a, c, v) in sorted(stat.items(), key=lambda kv: -kv[1][0]):
        if c:
            print(f"{it:22s} {a:6d} {c:8d} {v:8d}")
    print(f"{'ALL':22s} {tot[0]:6d} {tot[1]:8d} {tot[2]:8d}  -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
