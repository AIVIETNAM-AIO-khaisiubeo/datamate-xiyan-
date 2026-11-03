"""Build multi-task SFT data for the LLM generators from the TRAIN split only.

Tasks:
  rewrite : noisy question  -> canonical Vietnamese question (G3 target)
  frame   : noisy question  -> JSON query frame (G2 target; unanswerable questions -> intent 'unanswerable')

The prompts are exactly the inference prompts (shop vocabulary, rulebook digest, few-shot examples from the
train bank with the shop itself excluded), so the fine-tuned model sees at training what it sees at test.

Targets come from the rule generator's frames that reproduce the gold answer; every
question is additionally noised several times (augment.py) to mimic public/private test
style.  Usage:
    python training/build_sft_data.py --split data/train --out outputs/sft.jsonl --n-aug 3 [--holdout-fold 0]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATAMATE_CACHE", os.path.join(ROOT, ".cache"))

from datamate.augment import augment  # noqa: E402
from datamate.engine.answer import FrameExecutor  # noqa: E402
from datamate.engine.executor import Executor  # noqa: E402
from datamate.ingest.builder import load_biz  # noqa: E402
from datamate.llm import prompts as P  # noqa: E402
from datamate.llm.generators import render_frame  # noqa: E402
from datamate.nlu.rules import RuleParser  # noqa: E402
from datamate.nlu.timeparse import LunarCalendar  # noqa: E402
from datamate.scorer import score_one  # noqa: E402


def frame_to_llm_json(fr, b) -> dict:
    f = fr.filters
    js = {"intent": fr.intent, "time": fr.params.get("time_text") or "", "compare": fr.params.get("mode") or "none"}
    from datamate.llm.generators import render_time
    if fr.periods:
        js["time"] = render_time(fr.periods[0])
    if fr.snapshot:
        js["time"] = f"đến hết ngày {fr.snapshot:%d/%m/%Y}"
    if f.get("product"):
        js["product"] = b.product_name(f["product"])
    for k_src, k_dst in (("group", "group"), ("cust_group", "customer_group"), ("ingredient", "ingredient"),
                         ("staff", "staff"), ("channel", "channel"), ("pay", "payment"), ("shift", "shift"),
                         ("rx", "rx"), ("size", "size")):
        if f.get(k_src) is not None:
            js[k_dst] = f[k_src]
    if fr.k:
        js["k"] = fr.k
    if fr.by:
        js["by"] = fr.by
    if fr.unit:
        js["unit"] = fr.unit
    if fr.reason:
        js["reason"] = fr.reason
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", default=os.path.join(ROOT, "artifacts", "example_bank.jsonl"),
                    help="few-shot bank used at inference (scripts/build_artifacts.py)")
    ap.add_argument("--bank-out", default=None, help="optionally also write a bank of the kept examples")
    ap.add_argument("--n-aug", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-fold", type=int, default=None, help="leave the shops of this fold out (evaluation)")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    from datamate.agent import rulebook_digest, vocab_for
    from datamate.folds import fold_shops
    from datamate.llm.generators import ExampleBank
    rng = random.Random(a.seed)
    cal = LunarCalendar(os.path.join(a.split, "calendar.csv"))
    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8")]
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(a.split, "answers.jsonl"), encoding="utf-8")}
    digest = rulebook_digest(open(os.path.join(a.split, "rulebook.md"), encoding="utf-8").read())
    bank = ExampleBank.load(a.bank) if os.path.exists(a.bank) else None
    if bank is not None and not bank.has_shop_ids:
        sys.exit(f"{a.bank} has no shop ids: run `python scripts/build_artifacts.py {a.split}` first "
                 "(otherwise own-shop and held-out examples would leak into the training prompts)")
    by = {}
    for q in qs:
        by.setdefault(q["business_id"], []).append(q)
    holdout = fold_shops(by, a.holdout_fold, a.folds) if a.holdout_fold is not None else set()
    if holdout:
        print("held-out shops (not in the training data):", sorted(holdout))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fb = open(a.bank_out, "w", encoding="utf-8") if a.bank_out else None
    n = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for bid, items in sorted(by.items()):
            if bid in holdout:
                continue
            b = load_biz(os.path.join(a.split, "data", bid), bid)
            rp = RuleParser(b, cal)
            vocab = vocab_for(b, rp)                    # same shop vocabulary block as at inference
            if bank is not None:
                bank.exclude = {bid} | holdout          # no own-shop and no held-out examples as few-shot
            for q in items:
                today = dt.date.fromisoformat(q["as_of_date"])
                fr = rp.parse(q["question"], today)
                if fr is None:
                    continue
                ans = FrameExecutor(Executor(b, today), cal).run(fr)
                if score_one(ans.to_submission(q["id"]), refs[q["id"]]) != 1:
                    continue  # only frames that reproduce the gold answer become targets
                js = frame_to_llm_json(fr, b)
                canon = render_frame(fr, b.product_name) or q["question"]
                if fb:
                    fb.write(json.dumps({"q": q["question"], "nq": fr.params.get("nq"), "intent": fr.intent, "biz": bid,
                                         "frame_llm": json.dumps(js, ensure_ascii=False), "canon": canon},
                                        ensure_ascii=False) + "\n")
                variants = [q["question"]] + [augment(q["question"], rng, 0.6) for _ in range(a.n_aug)]
                for v in variants:
                    shots = bank.topk(v, 6) if bank is not None else []
                    rw_shots = [s for s in shots if s.get("canon")]
                    fr_shots = [{"q": s["q"], "frame": s["frame_llm"]} for s in shots if s.get("frame_llm")]
                    # prompts identical to RewriteGenerator / FrameGenerator at inference
                    fo.write(json.dumps({"task": "rewrite", "system": P.REWRITE_SYSTEM,
                                         "user": P.rewrite_user(v, q["as_of_date"], vocab, rw_shots),
                                         "target": canon}, ensure_ascii=False) + "\n")
                    fo.write(json.dumps({"task": "frame", "system": P.FRAME_SYSTEM,
                                         "user": P.frame_user(v, q["as_of_date"], digest, vocab, fr_shots),
                                         "target": json.dumps(js, ensure_ascii=False)}, ensure_ascii=False) + "\n")
                    n += 2
            print(bid, n, flush=True)
    if fb:
        fb.close()
    print("wrote", n, "examples to", a.out)


if __name__ == "__main__":
    main()
