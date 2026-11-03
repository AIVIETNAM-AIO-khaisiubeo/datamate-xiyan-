# models/

Fine-tuned weights that belong to the repository (stored with **Git LFS**, see `.gitattributes`).

| Path | Content | Size | Created by |
|---|---|---|---|
| `adapter/` | LoRA adapter of Qwen3-14B (bf16, rank 32): `adapter_config.json`, `adapter_model.safetensors` | ~0.26 GB | `training/sft_lora.py` → `<out>/adapter_bf16` |
| `adapter/TRAINING.md` | data, hyper-parameters, held-out results of the run that produced the adapter | small | you, after training |

The base model (Qwen/Qwen3-14B, ~30 GB) is **not** stored here: it is downloaded from the Hugging Face Hub
(`huggingface-cli download Qwen/Qwen3-14B`, or at `docker build` time). A merged model (~30 GB) does not fit
GitHub either; publish it to the Hub with `scripts/publish_hf.py` if you need `Dockerfile.finetuned`.

Use the adapter:

```bash
git lfs install && git lfs pull                     # fetch the real weights, not the LFS pointer files
python scripts/count_params.py /models/qwen3-14b models/adapter        # base + adapter must stay <= 15B
python -m datamate.predict --data data/public_test --llm vllm --model /models/qwen3-14b --lora models/adapter \
  --out runs/public_lora/submission.jsonl
docker build -f Dockerfile.lora -t datamate-xiyan:lora .                # image = base download + this adapter
```

The adapter is trained on the **train** split only (competition rule). The repository is public: check the
competition rules on publishing data and data-derived weights before pushing them.
