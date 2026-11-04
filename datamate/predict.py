"""CLI entry point: python -m datamate.predict --data /data --out /result/submission.jsonl"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(description="DataMate-XiYan agent")
    ap.add_argument("--data", default="/data", help="split directory (data/, questions.jsonl, rulebook.md, calendar.csv)")
    ap.add_argument("--out", default="/result/submission.jsonl")
    ap.add_argument("--logs", default=None, help="per-question log dir (default: <out dir>/logs)")
    ap.add_argument("--llm", default=None, choices=[None, "none", "vllm", "hf"], help="override LLM backend")
    ap.add_argument("--model", default=None, help="model directory")
    ap.add_argument("--lora", default=None, help="fine-tuned LoRA adapter directory, served next to --model")
    ap.add_argument("--tau", type=float, default=None, help="abstention threshold")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N questions")
    ap.add_argument("--only", nargs="*", default=None, help="debug: only these business ids")
    ap.add_argument("--fold", type=int, default=None, help="only the shops of held-out fold K (see datamate/folds.py)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tp", type=int, default=None, help="tensor parallel size (GPUs), default $DATAMATE_TP or 2")
    ap.add_argument("--quant", default=None, help="vLLM quantization, e.g. fp8 (same model, smaller GPU)")
    ap.add_argument("--max-len", type=int, default=None, help="vLLM max_model_len")
    ap.add_argument("--gpu-util", type=float, default=None, help="vLLM gpu_memory_utilization")
    ap.add_argument("--dtype", default=None, help="vLLM dtype: 'half' on GPUs without bf16 (T4, V100); default auto")
    ap.add_argument("--enforce-eager", action="store_true", help="vLLM without CUDA graphs (less memory, slower)")
    ap.add_argument("--require-llm", action="store_true",
                    help="stop with an error if the LLM cannot be started, instead of silently running rule-only")
    ap.add_argument("--generators", default=None, help="LLM generators to run, e.g. g2,g3,g4 or g3 (G1 always runs)")
    ap.add_argument("--g4-scope", default=None, choices=[None, "unresolved", "all"], help="G4 on unresolved or all questions")
    a = ap.parse_args(argv)

    os.environ.setdefault("DATAMATE_CACHE", "/tmp/datamate_cache")
    from .agent import DataMateAgent, default_config

    cfg = default_config()
    if a.llm:
        cfg["llm"]["backend"] = a.llm
    if a.model:
        cfg["llm"]["model_path"] = a.model
    if a.lora:
        cfg["llm"]["lora_path"] = a.lora
    if a.tau is not None:
        cfg["tau"] = a.tau
    for k, v in (("tensor_parallel_size", a.tp), ("quantization", a.quant), ("max_model_len", a.max_len),
                 ("gpu_memory_utilization", a.gpu_util), ("dtype", a.dtype)):
        if v is not None:
            cfg["llm"][k] = v
    if a.enforce_eager:
        cfg["llm"]["enforce_eager"] = True
    if a.require_llm:
        cfg["llm"]["require_llm"] = True
    if a.generators is not None:
        gens = {g.strip().lower() for g in a.generators.split(",")}
        cfg["use_g2"], cfg["use_g3"], cfg["use_g4"] = "g2" in gens, "g3" in gens, "g4" in gens
    if a.g4_scope:
        cfg["g4_scope"] = a.g4_scope
    qpath = os.path.join(a.data, "questions.jsonl")
    with open(qpath, encoding="utf-8") as fh:
        qs = [json.loads(l) for l in fh if l.strip()]
    if a.only:
        qs = [q for q in qs if q["business_id"] in set(a.only)]
    if a.fold is not None:
        from .folds import fold_shops
        keep = fold_shops({q["business_id"] for q in qs}, a.fold, a.folds)
        qs = [q for q in qs if q["business_id"] in keep]
        print(f"[datamate] fold {a.fold}/{a.folds}: shops {sorted(keep)}", file=sys.stderr)
    if a.limit:
        qs = qs[: a.limit]
    log_dir = a.logs or os.path.join(os.path.dirname(os.path.abspath(a.out)), "logs")
    t0 = time.time()
    agent = DataMateAgent(a.data, cfg)
    if a.require_llm:
        agent._init_llm()                       # start the LLM now: a failure stops here, not one shop at a time
        if agent.backend is None:
            sys.exit("[datamate] --require-llm: no LLM backend could be started")
    res = agent.run(qs, a.out, log_dir)
    n_ans = sum(1 for r in res.values() if r["type"] not in ("abstain",))
    print(f"[datamate] {len(res)} questions, {n_ans} answered, {time.time() - t0:.1f}s -> {a.out}", file=sys.stderr)
    # optional local scoring
    apath = os.path.join(a.data, "answers.jsonl")
    if os.path.exists(apath):
        from .scorer import penalized_accuracy
        with open(apath, encoding="utf-8") as fh:
            refs = {json.loads(l)["id"]: json.loads(l) for l in fh if l.strip()}
        refs = {k: v for k, v in refs.items() if k in res}
        pa, c = penalized_accuracy(res, refs)
        print(f"[datamate] PA={pa:.4f} {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
