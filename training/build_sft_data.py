"""Build multi-task SFT data for the LLM generators from the TRAIN split only.

Tasks (XiYan-style multi-task / multi-format fine-tuning):
  rewrite : noisy question  -> canonical Vietnamese question (G3 target)
  frame   : noisy question  -> JSON query frame                (G2 target)
  answerability : question + data profile -> {"answerable": ..., "reason": ...}
                  (abstention as a teachable skill, cf. "When Silence is Golden")

Targets come from the rule generator's frames that reproduce the gold answer; every
question is additionally noised several times (augment.py) to mimic public/private test
style.  Usage:
    python training/build_sft_data.py --split data/train --out outputs/sft.jsonl --n-aug 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    ap.add_argument("--bank-out", default="artifacts/example_bank.jsonl")
    ap.add_argument("--n-aug", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cal = LunarCalendar(os.path.join(a.split, "calendar.csv"))
    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8")]
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(a.split, "answers.jsonl"), encoding="utf-8")}
    by = {}
    for q in qs:
        by.setdefault(q["business_id"], []).append(q)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.bank_out)), exist_ok=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as fo, open(a.bank_out, "w", encoding="utf-8") as fb:
        for bid, items in sorted(by.items()):
            b = load_biz(os.path.join(a.split, "data", bid), bid)
            rp = RuleParser(b, cal)
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
                bank_item = {"q": q["question"], "nq": fr.params.get("nq"), "intent": fr.intent,
                             "frame_llm": json.dumps(js, ensure_ascii=False), "canon": canon}
                fb.write(json.dumps(bank_item, ensure_ascii=False) + "\n")
                variants = [q["question"]] + [augment(q["question"], rng, 0.6) for _ in range(a.n_aug)]
                for v in variants:
                    fo.write(json.dumps({"task": "rewrite", "system": P.REWRITE_SYSTEM,
                                         "user": P.rewrite_user(v, q["as_of_date"], "", []),
                                         "target": canon}, ensure_ascii=False) + "\n")
                    fo.write(json.dumps({"task": "frame", "system": P.FRAME_SYSTEM,
                                         "user": P.frame_user(v, q["as_of_date"], "", "", []),
                                         "target": json.dumps(js, ensure_ascii=False)}, ensure_ascii=False) + "\n")
                    n += 2
                if refs[q["id"]]["type"] == "unanswerable":
                    fo.write(json.dumps({"task": "answerability", "system": P.ANSWERABILITY_SYSTEM,
                                         "user": P.answerability_user(q["question"], q["as_of_date"], ""),
                                         "target": json.dumps({"answerable": False, "reason": ans.reason}, ensure_ascii=False)},
                                        ensure_ascii=False) + "\n")
                    n += 1
            print(bid, n, flush=True)
    print("wrote", n, "examples to", a.out)


if __name__ == "__main__":
    main()
