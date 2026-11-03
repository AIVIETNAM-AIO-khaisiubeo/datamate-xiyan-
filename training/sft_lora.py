"""(Q)LoRA supervised fine-tuning of ONE Qwen3-14B on every LLM task of the agent.

Tasks mixed in one run (the model learns them jointly; the prompt says which task / SQL format):
  rewrite, frame                                                       G3 / G2   training/build_sft_data.py
  nl2sql_base|cte|alias, question_inference, evidence_inference,
  self_refine, select                                                  G4 / G_select  training/build_sql_sft.py

    pip install -r requirements-train.txt
    # 80 GB GPU (A100/H100): BF16 LoRA        24-48 GB GPU (RTX 4090/5090, L40S, A6000): add --qlora
    python training/sft_lora.py --model /models/llm --data outputs/sft.jsonl outputs/sql_sft/sql_sft.jsonl \
        --out outputs/lora_all --qlora --max-per-task 6000
    python training/sft_lora.py --merge --model /models/llm --adapter outputs/lora_all --out outputs/merged

Merging folds the LoRA weights into the base weights, so the merged model has exactly the base parameter
count (14.8B) and the 15B rule is unchanged.  Train on TRAIN data only.
"""
from __future__ import annotations

import argparse
import collections
import json
import random


def load_rows(paths, max_per_task: int, seed: int):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("task", "?")].append(r)
    rng = random.Random(seed)
    out = []
    for task, lst in sorted(by.items()):
        rng.shuffle(lst)
        keep = lst[:max_per_task] if max_per_task else lst
        out += keep
        print(f"  {task:20s} {len(keep):6d} / {len(lst)}")
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", default=["outputs/sft.jsonl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--qlora", action="store_true", help="4-bit NF4 base weights (fits a 24-32 GB GPU)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=8192, help="long enough for schema + rulebook + few-shot prompts")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-per-task", type=int, default=6000, help="cap examples per task (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.merge:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)   # BF16 base, also after QLoRA
        m = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
        m.save_pretrained(a.out, safe_serialization=True)
        AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
        print("merged model ->", a.out)
        return

    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    print("training examples per task:")
    rows = load_rows(a.data, a.max_per_task, a.seed)

    def to_msgs(r):
        return {"messages": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]},
                             {"role": "assistant", "content": r["target"]}]}

    ds = Dataset.from_list([to_msgs(r) for r in rows])
    kw = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if a.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if a.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=a.batch,
                    gradient_accumulation_steps=a.grad_accum, bf16=True, logging_steps=20, save_strategy="steps",
                    save_steps=500, save_total_limit=2, max_length=a.max_len, assistant_only_loss=True,
                    lr_scheduler_type="cosine", warmup_ratio=0.03, gradient_checkpointing=True, seed=a.seed)
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    tr.train()
    tr.save_model(a.out)
    print("adapter ->", a.out)


if __name__ == "__main__":
    main()
