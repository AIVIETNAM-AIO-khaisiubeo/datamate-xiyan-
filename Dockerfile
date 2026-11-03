# DataMate-XiYan — Zalo AI Challenge 2026 (DataMate track)
# Follow the organisers' Docker guideline for the exact base image; BASE_IMAGE must provide
# CUDA >= 12.8 and a vLLM / PyTorch build for Blackwell (RTX 5090).
ARG BASE_IMAGE=vllm/vllm-openai:v0.12.0
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# ---- model weights are baked into the image (inference runs with networking disabled)
# Default: Qwen3-14B (14.8B parameters, Apache-2.0, released 04/2025) — the only model used.
ARG MODEL_ID=Qwen/Qwen3-14B
ARG MODEL_DIR=/models/llm
RUN python3 -m pip install "huggingface_hub[cli]>=0.24" && \
    huggingface-cli download ${MODEL_ID} --local-dir ${MODEL_DIR} --exclude "*.pth" "original/*" && \
    rm -rf /root/.cache/huggingface
# (Optional) fine-tuned LoRA merged weights: COPY outputs/merged ${MODEL_DIR}

COPY . /app
RUN chmod +x /app/predict.sh

ENV DATAMATE_LLM=vllm DATAMATE_MODEL=${MODEL_DIR} DATAMATE_TP=2 DATAMATE_CACHE=/tmp/datamate_cache \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1

ENTRYPOINT []
CMD ["/app/predict.sh"]
