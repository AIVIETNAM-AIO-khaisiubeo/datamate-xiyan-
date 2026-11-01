"""Run G1 (rules + guards) on selected questions of a split and print frame + answer.

usage: python scripts/try_q.py SPLIT_DIR "substring or id" [...]
"""
import sys, os, json, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATAMATE_CACHE", ".cache")
from datamate.agent import DataMateAgent

root = sys.argv[1]
keys = sys.argv[2:]
qs = [json.loads(l) for l in open(os.path.join(root, "questions.jsonl"), encoding="utf-8")]
sel = [q for q in qs if any(k == q["id"] or k.lower() in q["question"].lower() for k in keys)]
ag = DataMateAgent(root)
from datamate.ingest.builder import load_biz
from datamate.nlu.rules import RuleParser
cache = {}
for q in sel:
    bid = q["business_id"]
    if bid not in cache:
        b = load_biz(os.path.join(root, "data", bid), bid)
        cache[bid] = (b, RuleParser(b, ag.cal))
    b, rp = cache[bid]
    fr, a = ag._g1(rp, b, q["question"], dt.date.fromisoformat(q["as_of_date"]))
    print(bid, q["as_of_date"], q["question"])
    print("   ", fr.to_json() if fr else None)
    print("   =>", a.type, a.value, a.meta, a.reason)
