"""Label-free comparison of two submissions of the same split (e.g. fine-tuned full system vs rule-only on public).

Public has no labels, so this does not give a PA. It shows *where* the two systems differ, which is where the
leaderboard score can move: questions newly answered (rule-only abstained), answers that changed, questions newly
abstained. With --logs (the per-question logs of the NEW run) it also says which generator produced each changed answer.
Every difference is written to --out for reading by hand; do not tune rules or prompts on these public questions
(see docs: public questions are for label-free error analysis only).

    python scripts/compare_submissions.py submissions/public_test_rule_only.jsonl runs/public_lora/submission.jsonl \
        --questions data/public_test/questions.jsonl --logs runs/public_lora/logs --out runs/public_lora/diff.jsonl
"""
import argparse
import collections
import json
import os


def load(path):
    return {r["id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


def key(r):
    return json.dumps([r.get("type"), r.get("answer")], ensure_ascii=False, sort_keys=True)


def kind(a, b):
    ta, tb = a.get("type", "abstain"), b.get("type", "abstain")
    if ta == "abstain" and tb != "abstain":
        return "newly_answered"
    if ta != "abstain" and tb == "abstain":
        return "newly_abstained"
    if key(a) == key(b):
        return "same"
    return "changed"


def compare(old, new):
    rows, cnt = [], collections.Counter()
    for qid in sorted(set(old) | set(new)):
        a, b = old.get(qid, {"type": "abstain"}), new.get(qid, {"type": "abstain"})
        k = kind(a, b)
        cnt[k] += 1
        if k != "same":
            rows.append({"id": qid, "kind": k, "old": a, "new": b})
    return cnt, rows


def winner(log_path):
    """Generators that voted for the final answer of a logged question."""
    try:
        lg = json.load(open(log_path, encoding="utf-8"))
    except (OSError, ValueError):
        return []
    fin = key(lg.get("final") or {})
    return sorted({c["gen"] for c in lg.get("candidates", []) if key(c.get("answer") or {}) == fin})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old", help="reference submission, e.g. rule-only")
    ap.add_argument("new", help="submission to inspect, e.g. fine-tuned full system")
    ap.add_argument("--questions", default=None, help="questions.jsonl, to print the question text")
    ap.add_argument("--logs", default=None, help="per-question logs of the NEW run")
    ap.add_argument("--out", default=None, help="write every difference here (jsonl)")
    a = ap.parse_args()
    old, new = load(a.old), load(a.new)
    qs = load(a.questions) if a.questions else {}
    cnt, rows = compare(old, new)
    n = sum(cnt.values())
    by_gen = collections.Counter()
    for r in rows:
        if qs:
            r["question"] = qs.get(r["id"], {}).get("question")
        if a.logs:
            r["gens"] = winner(os.path.join(a.logs, r["id"] + ".json"))
            by_gen[(r["kind"], "+".join(r["gens"]) or "?")] += 1
    for name, sub in (("old", old), ("new", new)):
        t = collections.Counter(r.get("type") for r in sub.values())
        print(f"{name}: {len(sub)} lines, answered {sum(v for k, v in t.items() if k != 'abstain')}, "
              f"abstain {t.get('abstain', 0)}, unanswerable {t.get('unanswerable', 0)}")
    print(f"{n} questions: " + ", ".join(f"{k} {cnt[k]}" for k in ("same", "newly_answered", "changed", "newly_abstained")))
    if by_gen:
        print("differences by the generators behind the new answer:")
        for (k, g), v in sorted(by_gen.items(), key=lambda kv: -kv[1]):
            print(f"  {k:16s} {g:40s} {v}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(rows)} differences -> {a.out}")
    print("Only the leaderboard scores these: each changed / newly answered question is +1 if right and -1 if wrong.")


if __name__ == "__main__":
    main()
