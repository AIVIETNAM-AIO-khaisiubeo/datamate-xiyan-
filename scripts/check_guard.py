"""Run the question-form guard over a run's logs (no data reload): which answered questions would it block?

usage: python scripts/check_guard.py <logs dir> [answers.jsonl]
With answers, also reports how many blocked answers were actually correct (guard false positives).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datamate.nlu.consistency import check
from types import SimpleNamespace as NS
from datamate.engine.answer import Answer
from datamate.scorer import score_one

logs = sys.argv[1]
refs = {}
if len(sys.argv) > 2:
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2], encoding="utf-8")}
n = bad = 0
for f in sorted(os.listdir(logs)):
    lg = json.load(open(os.path.join(logs, f), encoding="utf-8"))
    if not lg.get("g1_frame"):
        continue
    j = json.loads(lg["g1_frame"])
    fr = NS(intent=j["intent"], params=j.get("params") or {}, filters=j.get("filters") or {}, periods=[NS(start=a, end=b) for a, b, *_ in (j.get("periods") or [])])
    c = (lg.get("candidates") or [{}])[0].get("answer", {})
    a = Answer(c.get("type", "abstain"), c.get("answer"))
    why = check(fr.params.get("nq", ""), fr, a)
    if why:
        n += 1
        ok = ""
        if refs:
            ok = score_one(c, refs[lg["id"]])
            bad += bool(ok)
        print(f"{why:40s} | {fr.intent:18s} | {ok} | {lg['question']}")
print("blocked", n, "of which correct", bad)
