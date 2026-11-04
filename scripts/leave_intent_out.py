"""Leave-intent-out: how the system does on a question type the rule engine does not know (train labels only).

Private questions will include concepts the rule engine (G1) was never written for.  G2 and G3 cannot help there:
they only rewrite the question for the same rule executor.  What remains is the SQL route (G4 + G_select) and
abstention.  On the logs of a GPU run over held-out shops (G4 run on every question, e.g. the notebook's
noisy_lora run) this script removes, for each intent in turn, every candidate of the rule executor (G1, G2, G3)
on the questions of that intent, and re-runs the selector with reliabilities fitted on the other shops.

The result estimates, per intent and overall: how often the SQL route answers an unknown kind of question
(coverage), how often it is then right (precision), and the PA it contributes.  PA >= 0 means the abstention
logic keeps unknown questions from costing points; precision tells whether G4 is worth trusting there.

    python scripts/leave_intent_out.py <eval>/noisy_lora/logs <noisy split>/answers.jsonl --folds 3
"""
import argparse
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from datamate.folds import fold_of as _fold_of  # noqa: E402
from datamate.scorer import score_one  # noqa: E402
from datamate.select.form_guard import RULE_EXECUTOR_GENS  # noqa: E402
from datamate.select.selector import Selector  # noqa: E402
import validate  # noqa: E402


def intent_of(r):
    g1 = next((c for c in r["cands"] if c.gen == "G1_rule"), None)
    return g1.intent if g1 is not None and g1.intent else "__none__"


def run(recs, folds=3, tau=0.6):
    fold_of = _fold_of({r["biz"] for r in recs}, folds)
    out = []
    for f in range(folds):
        rel = validate.fit([r for r in recs if fold_of[r["biz"]] != f])
        sel = Selector(None, tau=tau)
        sel.rel = rel
        for r in recs:
            if fold_of[r["biz"]] != f:
                continue
            sql = [c for c in r["cands"] if c.gen not in RULE_EXECUTOR_GENS]
            a, _ = sel.select(sql)
            s = score_one(a.to_submission(r["id"]), r["ref"])
            out.append({"intent": intent_of(r), "g4_ran": any(c.gen.startswith("G4") for c in r["cands"]),
                        "answered": a.type != "abstain", "score": s})
    return out


def table(rows, only_g4=True):
    by = collections.defaultdict(list)
    for x in rows:
        if only_g4 and not x["g4_ran"]:
            continue
        by[x["intent"]].append(x)
    lines = []
    for it, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        n, ans = len(xs), [x for x in xs if x["answered"]]
        pa = sum(x["score"] for x in xs) / n
        prec = sum(x["score"] == 1 for x in ans) / len(ans) if ans else float("nan")
        lines.append((it, n, len(ans) / n, prec, pa))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("answers")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.6)
    a = ap.parse_args()
    recs = validate.load(a.logs, a.answers)
    rows = run(recs, a.folds, a.tau)
    g4 = [x for x in rows if x["g4_ran"]]
    print(f"{len(recs)} questions; G4 ran on {len(g4)} (only those are scored below: elsewhere the SQL route "
          f"was not even tried)")
    print(f"  {'intent removed from G1':26s} {'n':>5s} {'coverage':>9s} {'precision':>10s} {'PA':>8s}")
    for it, n, cov, prec, pa in table(rows):
        print(f"  {it:26s} {n:5d} {cov:9.3f} {prec:10.3f} {pa:8.3f}")
    if g4:
        ans = [x for x in g4 if x["answered"]]
        print(f"  {'ALL':26s} {len(g4):5d} {len(ans) / len(g4):9.3f} "
              f"{(sum(x['score'] == 1 for x in ans) / len(ans)) if ans else float('nan'):10.3f} "
              f"{sum(x['score'] for x in g4) / len(g4):8.3f}")
    print("PA < 0 for an intent: on unknown questions of that kind the SQL route costs points; raise tau or "
          "min_sql_support. Coverage ~0: unknown questions are abstained (safe, 0 points).")


if __name__ == "__main__":
    main()
