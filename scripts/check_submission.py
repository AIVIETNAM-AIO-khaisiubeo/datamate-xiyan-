"""Check a submission.jsonl before uploading it to the leaderboard.

  * one JSON object per line, UTF-8, exactly one line per question id of the split, no duplicates / extras
  * "type" in the allowed set; "answer" present and well-typed for value types
    (number/percent: int or float, boolean: bool, date: 'YYYY-MM-DD' or 'YYYY-MM', string: str,
     list: list, table: {"columns": [...], "rows": [[...], ...]}); abstain / unanswerable carry no answer
  * prints the type distribution; with answers.jsonl (train) also prints the local PA

    python scripts/check_submission.py runs/public/submission.jsonl data/public_test/questions.jsonl [answers.jsonl]
"""
import collections
import json
import re
import sys

TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}


def check_answer(t, a):
    if t in ("abstain", "unanswerable"):
        return None
    if a is None:
        return "missing answer"
    if t in ("number", "percent"):
        return None if isinstance(a, (int, float)) and not isinstance(a, bool) else "number expected"
    if t == "boolean":
        return None if isinstance(a, bool) else "boolean expected"
    if t == "date":
        return None if isinstance(a, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", a) else "date 'YYYY-MM-DD' expected"
    if t == "string":
        return None if isinstance(a, str) and a.strip() else "non-empty string expected"
    if t == "list":
        return None if isinstance(a, list) and a else "non-empty list expected"
    if t == "table":
        ok = isinstance(a, dict) and isinstance(a.get("columns"), list) and isinstance(a.get("rows"), list) and \
            all(isinstance(r, list) and len(r) == len(a["columns"]) for r in a["rows"])
        return None if ok else "table {columns, rows} expected"
    return "unknown type"


def main(sub, questions, answers=None):
    qids = [json.loads(l)["id"] for l in open(questions, encoding="utf-8") if l.strip()]
    errors, seen, types = [], set(), collections.Counter()
    preds = {}
    with open(sub, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                errors.append(f"line {i}: empty line")
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {i}: invalid JSON ({e})")
                continue
            qid, t = r.get("id"), r.get("type")
            if qid in seen:
                errors.append(f"line {i}: duplicate id {qid}")
            seen.add(qid)
            if t not in TYPES:
                errors.append(f"line {i}: bad type {t!r}")
                continue
            err = check_answer(t, r.get("answer"))
            if err:
                errors.append(f"line {i} ({qid}): {err}")
            types[t] += 1
            preds[qid] = r
    missing = [q for q in qids if q not in seen]
    extra = [q for q in seen if q not in set(qids)]
    if missing:
        errors.append(f"{len(missing)} question ids missing, e.g. {missing[:3]}")
    if extra:
        errors.append(f"{len(extra)} ids not in the split, e.g. {extra[:3]}")
    print(f"{len(seen)} lines for {len(qids)} questions; types: {dict(types)}")
    if answers:
        sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
        from datamate.scorer import score_one
        refs = {json.loads(l)["id"]: json.loads(l) for l in open(answers, encoding="utf-8")}
        sc = [score_one(preds.get(q), refs[q]) for q in qids if q in refs]
        print(f"local PA = {sum(sc) / max(1, len(sc)):.4f}  (correct {sc.count(1)}, wrong {sc.count(-1)}, abstain {sc.count(0)})")
    if errors:
        print("INVALID:")
        for e in errors[:30]:
            print("  " + e)
        sys.exit(1)
    print("OK: format valid")


if __name__ == "__main__":
    main(*sys.argv[1:])
