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
    ap.add_argument("--tau", type=float, default=None, help="abstention threshold")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N questions")
    ap.add_argument("--only", nargs="*", default=None, help="debug: only these business ids")
    a = ap.parse_args(argv)

    os.environ.setdefault("DATAMATE_CACHE", "/tmp/datamate_cache")
    from .agent import DataMateAgent, default_config

    cfg = default_config()
    if a.llm:
        cfg["llm"]["backend"] = a.llm
    if a.model:
        cfg["llm"]["model_path"] = a.model
    if a.tau is not None:
        cfg["tau"] = a.tau
    qpath = os.path.join(a.data, "questions.jsonl")
    with open(qpath, encoding="utf-8") as fh:
        qs = [json.loads(l) for l in fh if l.strip()]
    if a.only:
        qs = [q for q in qs if q["business_id"] in set(a.only)]
    if a.limit:
        qs = qs[: a.limit]
    log_dir = a.logs or os.path.join(os.path.dirname(os.path.abspath(a.out)), "logs")
    t0 = time.time()
    agent = DataMateAgent(a.data, cfg)
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
