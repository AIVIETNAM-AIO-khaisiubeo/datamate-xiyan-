"""Measure the Schema Filter on TRAIN against the columns of execution-verified SQL (artifacts/sql_silver.jsonl).

Reports, for S_rtrv / S1 / S2 / full schema: column recall (share of the gold query's columns kept), share of
questions with *every* gold column kept (the condition for a correct SQL), and schema size.  Runs with the
heuristic column selector (no LLM) unless --llm is given.

    python scripts/eval_schema_filter.py data/train [--limit N]
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATAMATE_CACHE", os.path.join(ROOT, ".cache"))

from datamate.ingest.builder import load_biz  # noqa: E402
from datamate.llm.evidence import EvidenceRetriever  # noqa: E402
from datamate.llm.mschema import build_mschema, build_views  # noqa: E402
from datamate.llm.schema_filter import SchemaFilter, column_recall  # noqa: E402
from datamate.nlu.linker import Linker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--silver", default=os.path.join(ROOT, "artifacts", "sql_silver.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "schema_filter_eval.jsonl"))
    a = ap.parse_args()
    rb = open(os.path.join(a.split, "rulebook.md"), encoding="utf-8").read()
    ev = EvidenceRetriever(rb)
    items = [json.loads(l) for l in open(a.silver, encoding="utf-8")]
    if a.limit:
        items = items[:a.limit]
    by = collections.defaultdict(list)
    for it in items:
        by[it["business_id"]].append(it)
    agg = collections.defaultdict(lambda: [0.0, 0, 0])     # name -> [sum recall, n full-recall, sum size]
    per_intent = collections.defaultdict(lambda: [0, 0])
    n = 0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for bid, lst in sorted(by.items()):
            b = load_biz(os.path.join(a.split, "data", bid), bid)
            V = build_views(b)
            ms = build_mschema(bid, V)
            sf = SchemaFilter(ms, V, Linker(b))
            for it in lst:
                res = sf.run(it["question"], ev.evidence(it["question"], it["as_of_date"]))
                gold = it["columns"]
                sets = {"S_rtrv": [c for c, _ in res.retrieved], "full": res.full_columns}
                for s in res.schemas:
                    sets[s["id"]] = s["columns"]
                sets.setdefault("S2", sets.get("S1", []))
                row = {"id": it["id"], "intent": it["intent"], "gold": gold}
                for name, cols in sets.items():
                    r = column_recall(gold, cols)
                    agg[name][0] += r
                    agg[name][1] += r == 1.0
                    agg[name][2] += len(cols)
                    row[name] = {"recall": round(r, 3), "size": len(cols),
                                 "missing": sorted(set(gold) - set(cols))}
                per_intent[it["intent"]][0] += 1
                per_intent[it["intent"]][1] += column_recall(gold, sets["S2"]) == 1.0
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
            print(bid, n, flush=True)
    print(f"\n{'schema':8s} {'mean recall':>12s} {'all gold cols':>14s} {'mean #cols':>11s}   (n={n})")
    for name in ("S_rtrv", "S1", "S2", "full"):
        s, full, size = agg[name]
        print(f"{name:8s} {s / max(1, n):12.3f} {full / max(1, n):14.3f} {size / max(1, n):11.1f}")
    print("\nS2 keeps every gold column, by intent:")
    for it, (c, ok) in sorted(per_intent.items(), key=lambda kv: -kv[1][0]):
        print(f"  {it:18s} {ok}/{c}")


if __name__ == "__main__":
    main()
