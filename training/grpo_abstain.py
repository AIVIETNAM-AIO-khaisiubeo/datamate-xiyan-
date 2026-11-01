"""GRPO fine-tuning with an abstention-aware reward ("When Silence is Golden", ICLR 2026).

The paper's CoT-SFT -> RL pipeline is adapted to DataMate: the policy produces a JSON query
frame (or {"intent": "unanswerable"} / {"intent": "abstain"}); the frame is *executed* by the
deterministic engine and rewarded with the competition metric itself:

    answer reward  = +1 correct, -1 wrong, 0 abstain      (Penalized Accuracy)
    unanswerable   = +1 if gold is unanswerable, -1 otherwise
    format reward  = +0.2 if the output parses as a valid frame, -0.5 otherwise

So the policy learns that abstaining beats guessing whenever P(correct) < 0.5, which is the
calibration property the paper reports SFT alone fails to provide.

    python training/grpo_abstain.py --model outputs/merged_sft --split data/train --out outputs/grpo
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_reward_fn(split: str):
    from datamate.engine.answer import Answer, FrameExecutor
    from datamate.engine.executor import Executor
    from datamate.ingest.builder import load_biz
    from datamate.llm.backend import extract_json
    from datamate.llm.generators import llm_json_to_text
    from datamate.nlu.rules import RuleParser
    from datamate.nlu.timeparse import LunarCalendar
    from datamate.scorer import score_one

    cal = LunarCalendar(os.path.join(split, "calendar.csv"))
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(split, "answers.jsonl"), encoding="utf-8")}
    cache = {}

    def parser(bid):
        if bid not in cache:
            b = load_biz(os.path.join(split, "data", bid), bid)
            cache[bid] = (b, RuleParser(b, cal))
        return cache[bid]

    def reward(completions, qid, business_id, as_of_date, **kw):
        out = []
        for comp, q_id, bid, asof in zip(completions, qid, business_id, as_of_date):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            js = extract_json(text)
            if not js or "intent" not in js:
                out.append(-0.5)
                continue
            ref = refs[q_id]
            if js["intent"] == "abstain":
                out.append(0.2)
                continue
            if js["intent"] == "unanswerable":
                out.append((1.0 if ref["type"] == "unanswerable" else -1.0) + 0.2)
                continue
            b, rp = parser(bid)
            today = dt.date.fromisoformat(asof)
            t = llm_json_to_text(js)
            fr = rp.parse(t, today) if t else None
            if fr is None:
                out.append(-0.3)
                continue
            ans = FrameExecutor(Executor(b, today), cal).run(fr)
            out.append(float(score_one(ans.to_submission(q_id), ref)) + 0.2)
        return out

    return reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--group", type=int, default=8)
    a = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from datamate.llm import prompts as P

    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8")]
    rows = [{"prompt": [{"role": "system", "content": P.FRAME_SYSTEM},
                        {"role": "user", "content": P.frame_user(q["question"], q["as_of_date"], "", "", [])}],
             "qid": q["id"], "business_id": q["business_id"], "as_of_date": q["as_of_date"]} for q in qs]
    ds = Dataset.from_list(rows).shuffle(seed=0)
    cfg = GRPOConfig(output_dir=a.out, max_steps=a.steps, num_generations=a.group, per_device_train_batch_size=a.group,
                     learning_rate=5e-6, beta=0.04, bf16=True, logging_steps=10, max_completion_length=256, seed=0,
                     temperature=0.8)
    tr = GRPOTrainer(model=a.model, reward_funcs=[build_reward_fn(a.split)], args=cfg, train_dataset=ds,
                     peft_config=LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    tr.train()
    tr.save_model(a.out)


if __name__ == "__main__":
    main()
