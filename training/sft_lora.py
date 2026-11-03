"""(Q)LoRA supervised fine-tuning of ONE Qwen3-14B on every LLM task of the agent.

Tasks mixed in one run (the model learns them jointly; the prompt says which task / SQL format):
  rewrite, frame                                                       G3 / G2   training/build_sft_data.py
  nl2sql_base|cte|alias, question_inference, evidence_inference,
  self_refine, select                                                  G4 / G_select  training/build_sql_sft.py

    pip install -r requirements-train.txt
    # 80 GB GPU (A100/H100): BF16 LoRA        24-48 GB GPU (A100 40GB, L4, RTX 4090/5090): add --qlora
    python training/sft_lora.py --model /models/llm --data outputs/sft.jsonl outputs/sql_sft/sql_sft.jsonl \
        --out outputs/lora_all --qlora --max-per-task 6000 [--resume]
    # optional: one standalone model (only needed for Dockerfile.finetuned; ~30 GB)
    python training/sft_lora.py --merge --model /models/llm --adapter outputs/lora_all --out outputs/merged

Each example is a (prompt, completion) pair: the prompt is rendered with the model's chat template exactly as the
inference backends do (system + user, generation prompt, thinking disabled), the completion is the target plus
the end-of-turn token, and the loss is computed on the completion only.  After training a bf16 copy of the
adapter is written to <out>/adapter_bf16 (about 0.26 GB for rank 32): this is what is committed (Git LFS) and
loaded next to the base model at inference (--lora).  Base (14.77B) + adapter (0.13B) stays under 15B; a merged
model has exactly the base parameter count.  Train on TRAIN data only.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import shutil


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


def render_prompt(tok, system: str, user: str) -> str:
    """Same call as VLLMBackend.chat_prompt / HFBackend.chat_prompt (Qwen3: thinking disabled)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_example(r: dict, tok) -> dict:
    completion = r["target"]
    eos = getattr(tok, "eos_token", None) or ""
    if eos and not completion.endswith(eos):
        completion += eos
    return {"prompt": render_prompt(tok, r["system"], r["user"]), "completion": completion}


def export_bf16(adapter_dir: str, out_dir: str) -> str:
    """bf16 copy of a PEFT adapter (QLoRA keeps LoRA weights in fp32: half the size, same quality)."""
    import torch
    from safetensors.torch import load_file, save_file
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(adapter_dir, "adapter_model.safetensors")
    sd = {k: v.to(torch.bfloat16) for k, v in load_file(src).items()}
    save_file(sd, os.path.join(out_dir, "adapter_model.safetensors"), metadata={"format": "pt"})
    for f in ("adapter_config.json", "README.md"):
        if os.path.exists(os.path.join(adapter_dir, f)):
            shutil.copy(os.path.join(adapter_dir, f), out_dir)
    n = sum(v.numel() for v in sd.values())
    print(f"bf16 adapter: {n / 1e6:.1f}M parameters -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", default=["outputs/sft.jsonl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--qlora", action="store_true", help="4-bit NF4 base weights (fits a 24-48 GB GPU)")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out (Colab disconnects)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=8192, help="long enough for schema + rulebook + few-shot prompts")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-per-task", type=int, default=6000, help="cap examples per task (0 = all)")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.merge:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto",
                                                    low_cpu_mem_usage=True)          # BF16 base, also after QLoRA
        m = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
        m.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
        AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
        print("merged model ->", a.out)
        return

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("needs a GPU with bf16 (A100 / L4 / RTX 30xx-50xx / H100); a T4 or V100 cannot train this model")

    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    print("training examples per task:")
    rows = load_rows(a.data, a.max_per_task, a.seed)
    ds = Dataset.from_list([build_example(r, tok) for r in rows])
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
                    gradient_accumulation_steps=a.grad_accum, bf16=True, logging_steps=10, save_strategy="steps",
                    save_steps=a.save_steps, save_total_limit=2, max_length=a.max_len, completion_only_loss=True,
                    lr_scheduler_type="cosine", warmup_ratio=0.03, gradient_checkpointing=True, seed=a.seed,
                    report_to="none")
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    resume = a.resume and bool(glob.glob(os.path.join(a.out, "checkpoint-*")))
    if a.resume and not resume:
        print("no checkpoint in", a.out, "-> starting from scratch")
    tr.train(resume_from_checkpoint=True if resume else None)
    tr.save_model(a.out)
    export_bf16(a.out, os.path.join(a.out, "adapter_bf16"))
    print("adapter ->", a.out)


if __name__ == "__main__":
    main()
