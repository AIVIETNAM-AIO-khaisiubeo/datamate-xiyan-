"""Robustness of the full rule-only agent to public-test-style noise, measured on TRAIN.

Each train question is rewritten with datamate.augment (dropped accents, Telex slips,
abbreviations, dialect fillers, date shorthands); the gold answer is unchanged, so PA on the
noisy copy estimates how the agent behaves on noisy unseen questions.

    python scripts/eval_noise.py data/train [n_variants] [level]
"""
import collections
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATAMATE_CACHE", os.path.join(ROOT, ".cache"))

from datamate.agent import DataMateAgent  # noqa: E402
from datamate.augment import augment  # noqa: E402
from datamate.scorer import penalized_accuracy  # noqa: E402


def main(split, n=1, level=0.7):
    rng = random.Random(1234)
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(split, "answers.jsonl"), encoding="utf-8")}
    noisy, nrefs = [], {}
    for q in qs:
        for i in range(int(n)):
            nq = dict(q)
            nq["id"] = f"{q['id']}~{i}"
            nq["question"] = augment(q["question"], rng, float(level))
            noisy.append(nq)
            nrefs[nq["id"]] = refs[q["id"]]
    agent = DataMateAgent(split, {"llm": {"backend": "none"}})
    out = os.path.join(ROOT, "runs", "noise", "submission.jsonl")
    res = agent.run(noisy, out, os.path.join(ROOT, "runs", "noise", "logs"))
    pa, c = penalized_accuracy(res, nrefs)
    print(f"noisy PA={pa:.4f} {c}")
    # where do abstentions / errors come from
    why = collections.Counter()
    for q in noisy:
        lg = json.load(open(os.path.join(ROOT, "runs", "noise", "logs", q["id"] + ".json"), encoding="utf-8"))
        if lg["final"]["type"] == "abstain":
            oov = [c for c in lg["candidates"] if c.get("gen") == "G1_rule"]
            why["no_parse" if not oov else "low_conf"] += 1
    print(why)


if __name__ == "__main__":
    main(*sys.argv[1:])
