# LoRA adapter — phase A

- base: Qwen/Qwen3-14B; method: QLoRA 4-bit, rank 32, epochs 1, lr 0.0001, max-per-task 1500
- data: train split only; held-out fold 0 not used for training
- examples per task (before the cap): {"rewrite": 7341, "frame": 7341, "nl2sql_cte": 2766, "self_refine": 4444, "nl2sql_base": 2766, "select": 1377, "question_inference": 1383, "evidence_inference": 1383, "nl2sql_alias": 2766}
- this notebook run: GPU CPU, inference model None; date: 2026-11-04
- phase A evaluation on the held-out fold 0, noisy copy (scripts/make_noisy_split.py):
  PA end-to-end and LLM generators only (G1 removed): {"rule": {"PA_noisy": 0.9853, "llm_only": {}}, "base": {"PA_noisy": 0.9963, "llm_only": {"PA": 0.9467889908256881, "coverage": 0.998165137614679, "precision": 0.9742647058823529}}, "lora": {"PA_noisy": 0.9963, "llm_only": {"PA": 0.981651376146789, "coverage": 1.0, "precision": 0.9908256880733946}}}
