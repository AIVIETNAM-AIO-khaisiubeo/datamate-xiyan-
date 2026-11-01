"""LoRA supervised fine-tuning of the LLM generators (multi-task: rewrite / frame / answerability).

    pip install -r requirements-train.txt
    python training/sft_lora.py --model /models/llm --data outputs/sft.jsonl --out outputs/lora_sft
    python training/sft_lora.py --merge --model /models/llm --adapter outputs/lora_sft --out outputs/merged

The merged model keeps the base parameter count (LoRA weights are folded in), so the
15B-parameter budget is unchanged.
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="outputs/sft.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.merge:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
        m = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
        m.save_pretrained(a.out, safe_serialization=True)
        AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
        return

    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]

    def to_msgs(r):
        return {"messages": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]},
                             {"role": "assistant", "content": r["target"]}]}

    ds = Dataset.from_list([to_msgs(r) for r in rows]).shuffle(seed=0)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    peft_cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=2,
                    gradient_accumulation_steps=8, bf16=True, logging_steps=20, save_strategy="epoch",
                    max_length=a.max_len, assistant_only_loss=True, lr_scheduler_type="cosine", warmup_ratio=0.03,
                    gradient_checkpointing=True, seed=0)
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    tr.train()
    tr.save_model(a.out)


if __name__ == "__main__":
    main()
