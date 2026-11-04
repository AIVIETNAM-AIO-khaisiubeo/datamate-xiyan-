"""Noisy copy of the held-out train shops, to measure the full system (G1 + LLM generators) on train labels.

Why: the rules of G1 were written while reading train, so on the original train questions G1 answers almost everything
correctly and the end-to-end PA is ~1.0 for every configuration (rule-only, base model, fine-tuned model). It cannot
tell them apart. Public / private questions are noisier (no accents, Telex slips, abbreviations, date shorthands). This
script rewrites each question of the chosen shops with datamate.augment, the same noise eval_noise.py uses; the shops'
data, rulebook, calendar and gold answers are unchanged (only the wording of the question changes, so the label stays
valid). No question is invented: every row is a variant of an original train question.

    python scripts/make_noisy_split.py data/train runs/noisy_fold0 --fold 0 [--n 1 --level 0.7]
    python -m datamate.predict --data runs/noisy_fold0 --llm none --out runs/noisy_fold0_rule/submission.jsonl

Do not pass --fold to predict on the output: it already contains only the held-out shops.
"""
import argparse
import json
import os
import random
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datamate.augment import augment  # noqa: E402
from datamate.folds import fold_shops  # noqa: E402


def link_or_copy(src, dst):
    if os.path.lexists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst, target_is_directory=os.path.isdir(src))
    except (OSError, NotImplementedError):          # Windows without symlink rights
        (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, dst)


def make(split, out, fold=None, folds=5, n=1, level=0.7, seed=1234, keep_unchanged=False):
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    refs = {}
    for l in open(os.path.join(split, "answers.jsonl"), encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            refs[r["id"]] = r
    if fold is not None:
        keep = fold_shops({q["business_id"] for q in qs}, fold, folds)
        qs = [q for q in qs if q["business_id"] in keep]
    rng = random.Random(seed)
    os.makedirs(out, exist_ok=True)
    stats = {"questions": len(qs), "variants": 0, "unchanged_skipped": 0}
    with open(os.path.join(out, "questions.jsonl"), "w", encoding="utf-8") as fq, \
            open(os.path.join(out, "answers.jsonl"), "w", encoding="utf-8") as fa:
        for q in qs:
            for i in range(n):
                text = augment(q["question"], rng, level)
                if text == q["question"] and not keep_unchanged:
                    stats["unchanged_skipped"] += 1
                    continue
                nid = f"{q['id']}~{i}"
                fq.write(json.dumps(dict(q, id=nid, question=text), ensure_ascii=False) + "\n")
                fa.write(json.dumps(dict(refs[q["id"]], id=nid), ensure_ascii=False) + "\n")
                stats["variants"] += 1
    for name in sorted(os.listdir(split)):
        if name not in ("questions.jsonl", "answers.jsonl"):
            link_or_copy(os.path.join(split, name), os.path.join(out, name))
    stats["shops"] = sorted({q["business_id"] for q in qs})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("out")
    ap.add_argument("--fold", type=int, default=None, help="only the shops of fold K (datamate/folds.py)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n", type=int, default=1, help="noisy variants per question")
    ap.add_argument("--level", type=float, default=0.7, help="noise level (0..1), as eval_noise.py")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep-unchanged", action="store_true", help="also keep variants the noise left unchanged")
    a = ap.parse_args()
    st = make(a.split, a.out, a.fold, a.folds, a.n, a.level, a.seed, a.keep_unchanged)
    print(f"{st['variants']} noisy questions from {st['questions']} originals of {len(st['shops'])} shops "
          f"({st['unchanged_skipped']} unchanged variants skipped) -> {a.out}")


if __name__ == "__main__":
    main()
