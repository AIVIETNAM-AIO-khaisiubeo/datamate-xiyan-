"""Label-free review of a run: dump question / G1 frame / final answer grouped by intent.

usage: python scripts/review_run.py <questions.jsonl> <logs dir> <out.txt> [intent ...]
"""
import sys, os, json, collections

qs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
logs = sys.argv[2]
want = set(sys.argv[4:])
by = collections.defaultdict(list)
for q in qs:
    p = os.path.join(logs, q["id"] + ".json")
    if not os.path.exists(p):
        continue
    lg = json.load(open(p, encoding="utf-8"))
    fr = json.loads(lg["g1_frame"]) if lg.get("g1_frame") else {}
    it = fr.get("intent", "-")
    if want and it not in want:
        continue
    fin = lg.get("final", {})
    by[it].append((q["business_id"], q["question"], fr.get("periods"), fr.get("filters"), fr.get("k"), fr.get("by"),
                   {k: v for k, v in (fr.get("params") or {}).items() if k != "nq"}, fin.get("type"), fin.get("answer"),
                   lg.get("selector", {}).get("p")))
with open(sys.argv[3], "w", encoding="utf-8") as f:
    for it, rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        f.write(f"\n=== {it} ({len(rows)})\n")
        for r in sorted(rows):
            f.write(" | ".join(map(str, r)) + "\n")
print({k: len(v) for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))})
