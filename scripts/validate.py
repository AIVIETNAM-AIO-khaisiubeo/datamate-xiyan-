"""Business-level validation of the selector and its confidence (train labels only).

From a run's per-question logs (every generator's candidates are logged) this script
  1. splits the shops into K folds (a shop is never in both the fitting and the evaluation part),
  2. fits the per generator x intent reliabilities on K-1 folds (Bayesian shrinkage, as calibrate.py),
  3. re-runs the selector on the held-out fold and reports PA, coverage, precision when answering,
     Brier score and ECE of the selector's probability, a reliability diagram and a tau sweep,
  4. writes every wrong answer to <out>/errors.jsonl (question, intent, generators, p) for error analysis,
  5. optionally writes reliabilities fitted on all the given logs (--write-reliability), e.g. after a GPU run on
     shops the fine-tuned model never saw (predict --fold K with a model trained with --holdout-fold K).
     Only the LLM generators are written unless --include-g1.

    python scripts/validate.py runs/train/logs data/train/answers.jsonl --folds 5 [--write-reliability artifacts/reliability.json]

Caveat: the rule generator G1 was written while reading the whole train split, so its held-out numbers are
optimistic; the procedure is meant for the LLM generators and for the selector's calibration.
"""
import argparse
import collections
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datamate.engine.answer import Answer  # noqa: E402
from datamate.scorer import score_one  # noqa: E402
from datamate.select.selector import Candidate, Selector  # noqa: E402


def load(logs_dir, answers):
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(answers, encoding="utf-8")}
    recs = []
    for f in sorted(os.listdir(logs_dir)):
        lg = json.load(open(os.path.join(logs_dir, f), encoding="utf-8"))
        if lg["id"] not in refs:
            continue
        cands = []
        for c in lg.get("candidates", []):
            a = c.get("answer") or {}
            ans = Answer(a.get("type", "abstain"), a.get("answer"), reason=a.get("reason", ""), meta=dict(c.get("meta") or {}))
            det = {"flags": c.get("flags") or [], "family": c.get("family")} if c.get("flags") or c.get("family") else {}
            cands.append(Candidate(c["gen"], ans, c.get("intent"), det))
        recs.append({"id": lg["id"], "biz": lg["business_id"], "question": lg["question"], "cands": cands, "ref": refs[lg["id"]]})
    return recs


def ikey(c):
    if c.answer.type == "unanswerable" and c.intent and c.intent != "unanswerable":
        return "unanswerable@" + c.intent
    return c.intent or "__none__"


def fit(recs, k=10.0, prior=0.9, cap=0.985):
    cnt = collections.defaultdict(lambda: [0, 0])
    tot = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        for c in r["cands"]:
            if c.answer.type == "abstain":
                continue
            s = score_one(c.answer.to_submission(r["id"]), r["ref"])
            cnt[(c.gen, ikey(c))][0 if s == 1 else 1] += 1
            tot[c.gen][0 if s == 1 else 1] += 1
    rel = collections.defaultdict(dict)
    for (g, it), (ok, bad) in cnt.items():
        rel[g][it] = round(min(cap, (ok + k * prior) / (ok + bad + k)), 4)
    for g, (ok, bad) in tot.items():
        rel[g]["__all__"] = round(min(cap, (ok + k * prior) / (ok + bad + k)), 4)
    return dict(rel)


def evaluate(recs, rel, tau):
    sel, raw = Selector(None, tau=tau), Selector(None, tau=0.0)      # raw: the top group without abstention
    sel.rel = raw.rel = rel
    out = []
    for r in recs:
        a, info = sel.select(r["cands"])
        top, info0 = raw.select(r["cands"])
        p = info0.get("p", 0.0) or 0.0
        correct = top.type != "abstain" and score_one(top.to_submission(r["id"]), r["ref"]) == 1
        s = score_one(a.to_submission(r["id"]), r["ref"])
        out.append({"id": r["id"], "biz": r["biz"], "p": p, "answered": a.type != "abstain", "score": s,
                    "top_correct": correct, "gens": sorted({c.gen for c in r["cands"]}),
                    "intent": next((c.intent for c in r["cands"] if c.intent), None), "question": r["question"]})
    return out


def metrics(rows, bins=10):
    n = len(rows)
    ans = [r for r in rows if r["answered"]]
    pa = sum(r["score"] for r in rows) / max(1, n)
    prec = sum(r["score"] == 1 for r in ans) / max(1, len(ans))
    scored = [r for r in rows if r["p"] > 0]
    brier = sum((r["p"] - r["top_correct"]) ** 2 for r in scored) / max(1, len(scored))
    ece, diag = 0.0, []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sub = [r for r in scored if lo <= r["p"] < hi or (b == bins - 1 and r["p"] == 1.0)]
        if not sub:
            continue
        conf = sum(r["p"] for r in sub) / len(sub)
        acc = sum(r["top_correct"] for r in sub) / len(sub)
        ece += len(sub) / max(1, len(scored)) * abs(conf - acc)
        diag.append((lo, hi, len(sub), conf, acc))
    return {"n": n, "PA": pa, "coverage": len(ans) / max(1, n), "precision": prec, "brier": brier, "ece": ece, "diagram": diag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("answers")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "validate"))
    ap.add_argument("--write-reliability", default=None)
    ap.add_argument("--include-g1", action="store_true",
                    help="also overwrite G1_rule reliabilities (default: only the LLM generators are written)")
    a = ap.parse_args()
    recs = load(a.logs, a.answers)
    shops = sorted({r["biz"] for r in recs})
    from datamate.folds import fold_of as _fold_of
    fold_of = _fold_of(shops, a.folds)               # same shop -> fold assignment as training / predict --fold
    rows = []
    for f in range(a.folds):
        rel = fit([r for r in recs if fold_of[r["biz"]] != f])
        rows += evaluate([r for r in recs if fold_of[r["biz"]] == f], rel, a.tau)
    m = metrics(rows)
    os.makedirs(a.out, exist_ok=True)
    print(f"business-level {a.folds}-fold, {len(shops)} shops, {m['n']} questions, tau={a.tau}")
    print(f"  PA={m['PA']:.4f}  coverage={m['coverage']:.3f}  precision={m['precision']:.4f}  "
          f"Brier={m['brier']:.4f}  ECE={m['ece']:.4f}")
    print("  reliability diagram (p bin: n, mean p, accuracy of top candidate)")
    for lo, hi, n, conf, acc in m["diagram"]:
        print(f"    [{lo:.1f},{hi:.1f})  n={n:5d}  p={conf:.3f}  acc={acc:.3f}")
    print("  tau sweep:")
    for t in (0.5, 0.55, 0.6, 0.7, 0.8, 0.9):
        rr = []
        for f in range(a.folds):
            rel = fit([r for r in recs if fold_of[r["biz"]] != f])
            rr += evaluate([r for r in recs if fold_of[r["biz"]] == f], rel, t)
        mm = metrics(rr)
        print(f"    tau={t:.2f}  PA={mm['PA']:.4f}  coverage={mm['coverage']:.3f}  precision={mm['precision']:.4f}")
    with open(os.path.join(a.out, "errors.jsonl"), "w", encoding="utf-8") as fo:
        for r in rows:
            if r["answered"] and r["score"] == -1:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(m, open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8"), indent=1)
    print(f"  wrong answers -> {os.path.join(a.out, 'errors.jsonl')}")
    if a.write_reliability:
        rel = fit(recs)
        if not a.include_g1:
            rel.pop("G1_rule", None)
        try:
            cur = json.load(open(a.write_reliability, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cur = {}
        cur.update(rel)
        json.dump(cur, open(a.write_reliability, "w", encoding="utf-8"), ensure_ascii=False, indent=1, sort_keys=True)
        print(f"  reliabilities of {sorted(rel)} -> {a.write_reliability}")


if __name__ == "__main__":
    main()
