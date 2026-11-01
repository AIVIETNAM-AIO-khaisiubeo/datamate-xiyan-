"""Estimate per-intent generator reliability from an evaluation dump (train split only).

    python scripts/calibrate.py runs/eval_rules_out.jsonl artifacts/reliability.json

Reliability r = (correct + k·prior) / (correct + wrong + k), capped at 0.985.  The shrinkage
(k = 10, prior = 0.9) keeps in-sample over-confidence in check: an intent needs a long clean
record before it is trusted, and an intent with errors quickly falls under the abstention
threshold.  'unanswerable' predictions are calibrated as their own pseudo-intent.
"""
import collections
import json
import sys


def main(src: str, dst: str, gen: str = "G1_rule", k: float = 10.0, prior: float = 0.9):
    c = collections.Counter()
    w = collections.Counter()
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        it = r.get("intent")
        if not it or it == "NONE" or r["pred"].get("type") == "abstain":
            continue
        if r["pred"].get("type") == "unanswerable":
            it = "unanswerable@" + it
        if r["score"] == 1:
            c[it] += 1
        elif r["score"] == -1:
            w[it] += 1
    rel = {}
    tot_c, tot_w = sum(c.values()), sum(w.values())
    for it in set(c) | set(w):
        rel[it] = round(min(0.985, (c[it] + k * prior) / (c[it] + w[it] + k)), 4)
    rel["__all__"] = round(min(0.985, (tot_c + k * prior) / (tot_c + tot_w + k)), 4)
    try:
        cur = json.load(open(dst, encoding="utf-8"))
    except Exception:
        cur = {}
    cur[gen] = rel
    json.dump(cur, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1, sort_keys=True)
    print(f"{gen}: {len(rel)} intents, overall {rel['__all__']}")
    for it in sorted(rel, key=lambda x: rel[x]):
        print(f"  {it:40s} {rel[it]:.3f}  ok={c.get(it, 0)} wrong={w.get(it, 0)}")


if __name__ == "__main__":
    main(*sys.argv[1:3], *(sys.argv[3:4] or []))
