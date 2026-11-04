"""One-minute check of the fine-tuning code path before a multi-hour run.

Builds a tiny random Qwen3 model that uses the REAL tokenizer / chat template of the base model, then runs the
real training/sft_lora.py on a few examples of the real SFT data: train -> checkpoint -> resume -> bf16 adapter
export -> merge.  It catches library-version problems (TRL / transformers / PEFT) in a minute instead of after
loading 30 GB of weights.  Works on GPU, and on CPU (DATAMATE_SFT_SMOKE=1 is set automatically).

    python training/smoke_test.py --base /content/models/qwen3-14b --data <sft.jsonl> <sql_sft.jsonl>
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args, env):
    print("$", " ".join(args[1:]))
    r = subprocess.run(args, env=dict(env, PYTHONIOENCODING="utf-8"), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-12:]
    print("\n".join("    " + l for l in tail))
    if r.returncode != 0:
        raise SystemExit("SMOKE TEST FAILED (see above)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model dir (only tokenizer + config are read)")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--keep", default=None, help="keep the work dir here instead of a temp dir")
    a = ap.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    work = a.keep or tempfile.mkdtemp(prefix="sft_smoke_")
    tiny = os.path.join(work, "tiny_model")
    tok = AutoTokenizer.from_pretrained(a.base)
    cfg = AutoConfig.from_pretrained(a.base)
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, max_position_embeddings=4096).items():
        setattr(cfg, k, v)
    if getattr(cfg, "layer_types", None):                      # per-layer attention types must match the layer count
        cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(tiny)
    tok.save_pretrained(tiny)
    env = dict(os.environ)
    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        env["DATAMATE_SFT_SMOKE"] = "1"
    out = os.path.join(work, "lora")
    common = [sys.executable, os.path.join(HERE, "sft_lora.py"), "--model", tiny, "--data", *a.data, "--out", out,
              "--max-per-task", "2", "--max-len", "1024", "--grad-accum", "1", "--rank", "8", "--save-steps", "4"]
    run(common + ["--epochs", "1"], env)
    assert os.path.exists(os.path.join(out, "adapter_bf16", "adapter_model.safetensors")), "no bf16 adapter"
    run(common + ["--epochs", "2", "--resume"], env)                     # continues from the saved checkpoint
    run([sys.executable, os.path.join(HERE, "sft_lora.py"), "--merge", "--model", tiny,
         "--adapter", os.path.join(out, "adapter_bf16"), "--out", os.path.join(work, "merged")], env)
    assert os.path.exists(os.path.join(work, "merged", "config.json")), "no merged model"
    print("SMOKE TEST OK: train, checkpoint/resume, bf16 adapter export and merge all work with these library versions")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
