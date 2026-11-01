#!/usr/bin/env bash
# DataMate-XiYan inference entry point (Zalo AI Challenge 2026).
# Reads the test split mounted read-only at /data and writes /result/submission.jsonl
# plus one log file per question in /result/logs/.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
RESULT_DIR="${RESULT_DIR:-/result}"
mkdir -p "${RESULT_DIR}/logs"

# all caches go to a writable location (the data mount is read-only)
export DATAMATE_CACHE="${DATAMATE_CACHE:-/tmp/datamate_cache}"
# LLM: vllm on 2 x RTX 5090 by default; set DATAMATE_LLM=none for the CPU rule-only mode
export DATAMATE_LLM="${DATAMATE_LLM:-vllm}"
export DATAMATE_MODEL="${DATAMATE_MODEL:-/models/llm}"
export DATAMATE_TP="${DATAMATE_TP:-2}"
# fully offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0

cd "$(dirname "$0")"
python3 -m datamate.predict --data "${DATA_DIR}" --out "${RESULT_DIR}/submission.jsonl" --logs "${RESULT_DIR}/logs"
