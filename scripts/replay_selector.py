"""Re-run only the selector on a logged run: a new submission.jsonl without running the LLM again (CPU, seconds).

Every generator's candidates are in the per-question logs, so a change to the selection policy (families, anchor,
reliabilities, tau) can be applied to an existing GPU run.  --legacy uses the selector of the first GPU runs and
must reproduce the logged answers; check that first, it shows the replay is faithful.

    python scripts/replay_selector.py runs/public_lora/logs --data data/public_test --legacy        # must match the run
    python scripts/replay_selector.py runs/public_lora/logs --data data/public_test \
        --reliability artifacts/reliability.json --out runs/public_lora/replay/submission.jsonl \
        --compare submissions/public_test_rule_only.jsonl

On a labelled split (answers.jsonl next to questions.jsonl) it also prints the PA.  Use it to compare policies on
train / held-out logs; on public it only shows what changes (no labels), see compare_submissions.py.
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from datamate.select.logs import candidates_from_log, trusted_from_rulebook  # noqa: E402
from datamate.select.selector import Selector  # noqa: E402


def replay(logs_dir, questions, rulebook_text, selector):
    out, stats = {}, collections.Counter()
    for q in questions:
        p = os.path.join(logs_dir, q["id"] + ".json")
        if not os.path.exists(p):
            out[q["id"]] = {"id": q["id"], "type": "abstain"}
            stats["no_log"] += 1
            continue
        lg = json.load(open(p, encoding="utf-8"))
        cands = candidates_from_log(lg)
        ans, info = selector.select(cands, trusted_from_rulebook(rulebook_text, cands))
        out[q["id"]] = ans.to_submission(q["id"])
        if any(g.get("eligible") is False for g in info.get("groups", [])):
            stats["llm_group_blocked"] += 1
        if (lg.get("final") or {}).get("type") is not None:
            same = json.dumps(lg["final"], sort_keys=True, ensure_ascii=False) == \
                json.dumps(out[q["id"]], sort_keys=True, ensure_ascii=False)
            stats["same_as_logged_final" if same else "differs_from_logged_final"] += 1
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("--data", required=True, help="split directory (questions.jsonl, rulebook.md[, answers.jsonl])")
    ap.add_argument("--reliability", default=os.path.join(ROOT, "artifacts", "reliability.json"))
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--legacy", action="store_true", help="selector of the first GPU runs (to check the replay)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None, help="another submission of the same split, e.g. rule-only")
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(os.path.join(a.data, "questions.jsonl"), encoding="utf-8") if l.strip()]
    rb_path = os.path.join(a.data, "rulebook.md")
    rb = open(rb_path, encoding="utf-8").read() if os.path.exists(rb_path) else ""
    sel = Selector(a.reliability, tau=a.tau, legacy=a.legacy)
    out, stats = replay(a.logs, qs, rb, sel)
    t = collections.Counter(r["type"] for r in out.values())
    print(f"{len(out)} questions: answered {sum(v for k, v in t.items() if k != 'abstain')}, abstain {t.get('abstain', 0)}, "
          f"unanswerable {t.get('unanswerable', 0)} | " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    apath = os.path.join(a.data, "answers.jsonl")
    if os.path.exists(apath):
        from datamate.scorer import penalized_accuracy
        refs = {json.loads(l)["id"]: json.loads(l) for l in open(apath, encoding="utf-8") if l.strip()}
        pa, c = penalized_accuracy(out, {k: v for k, v in refs.items() if k in out})
        print(f"PA={pa:.4f} {c}")
    if a.compare:
        import compare_submissions as cs
        old = cs.load(a.compare)
        cnt, _ = cs.compare(old, out)
        print(f"vs {os.path.basename(a.compare)}: " + ", ".join(f"{k} {cnt[k]}" for k in
                                                                ("same", "newly_answered", "changed", "newly_abstained")))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fo:
            for q in qs:
                fo.write(json.dumps(out[q["id"]], ensure_ascii=False) + "\n")
        print("->", a.out)


if __name__ == "__main__":
    main()
