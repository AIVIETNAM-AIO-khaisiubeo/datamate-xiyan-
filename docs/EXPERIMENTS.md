# Experiments, evidence and reproducibility

[← Overview](../README.md) · [Architecture](ARCHITECTURE.md) · [XiYan algorithms](XIYAN_ADAPTATION.md) · [Detailed historical results](RESULTS.md)

**Scope:** the evidence available in the `colab-finetune-phasea-20261104-1534` branch on **2026-11-04**. Scores below are **reported by existing project records**; they are not a fresh independent execution. Always associate a score with its exact split, model, selector, input file and submission artifact.

## 1. What counts as evidence?

| Term | Meaning | Appropriate claim |
|---|---|---|
| Implemented | Code and tests for a feature exist | "The repository has a multi-schema SQL pipeline" |
| Unit/mock tested | A deterministic or mocked integration test exercised code | "Control flow works in a simulated setting" |
| In-sample train evaluation | Rule logic and artifacts were developed while inspecting training data | "Performance on the known training set" (not unseen generalization) |
| Held-out business fold | Certain business IDs excluded when training the adapter; evaluation on those businesses | "Local cross-business evidence for that checkpoint" |
| Public leaderboard | Official PA for a **specific uploaded submission file** | "Reported official public test score for that submission" |
| Private/organizer rerun | Offline frozen Docker image evaluated on private data | **No verified result present** |

The public test has **no released answers**. A public leaderboard score **does not reveal which questions are correct**. A comparison of two public submissions can explain aggregate score changes, but without per-question labels cannot prove which particular SQL or business rule is correct.

## 2. Reported results, properly separated

### 2.1 End-to-end train and public submissions

| Dataset / submitted version | Reported PA | Source | Interpretation |
|---|---:|---|---|
| Train, G1 + coverage + selector, 3,000 questions | **1.000** | `docs/RESULTS.md` `1 | **In-sample**; rule patterns were refined with the training set |
| Public test, initial G1 submission | **0.561** | `docs/RESULTS.md` `2 | Official public score, historic version |
| Public test, revised G1 | **0.912** | `docs/RESULTS.md` `2 | Official public score; heuristics were revised after reading unlabelled public input |
| Public test, earlier Phase-A Qwen3-14B LoRA ensemble | **0.777** | `docs/RESULTS.md` `2 | Official public score with an **older selector** and broken SQL extraction for adapter output |
| Public test, later T4/AWQ Phase-A LoRA run | **0.895** | `docs/RESULTS.md` `2 | Official public score for a **different** submission/configuration |

The `0.912` figure is **not** an unbiased estimate of private-test generalization: development involved examining the public questions and raw inputs (without labels), and the engine was tuned to the training data as well.

**Important distinction:** the current branch also includes `submissions/public_test_lora_phasea_final_de716e81.jsonl` and replay artifacts. The reviewed report **does not establish an official PA for that exact final-named file**. Do **not** copy the `0.777` or `0.895` result onto it, or claim `0.912` for an unsubmitted or differently generated artifact. Record an explicit official report plus hash if/when available.

### 2.2 Phase-A held-out-business, noisy-input evaluation

According to [models/adapter/TRAINING.md](../models/adapter/TRAINING.md), the adapter configuration is Qwen3-14B / QLoRA 4-bit / rank 32 / one epoch / learning rate 0.0001 / per-task cap 1,500, with **fold 0 excluded from adapter training**.

| Mode | Reported full-system PA on noisy held-out fold | Reported LLM-only PA |
|---|---:|---:|
| G1 rule | **0.9853** | — |
| Base model | **0.9963** | **0.9468** |
| Phase-A LoRA | **0.9963** | **0.9817** |

The original record also reports LLM-only coverage **0.9982 → 1.0000** and precision **0.9743 → 0.9908** (base → LoRA). These measurements support the narrow statement that the Phase-A adapter **improved LLM-only performance on the recorded local noisy fold**; they do **not** demonstrate improved public PA, private PA, or a higher score than the rule path on every business.

**Provenance warning:** `models/adapter/TRAINING.md` contains the auto-generated line `this notebook run: GPU CPU, inference model None` despite containing A100/T4-related training and held-out-run notes. Treat that metadata line as **internally inconsistent**, not proof of hardware identity. For a paper or challenge report, recover runtime details from the original training logs/checkpoint history and save the exact notebook commit, output hashes and device profile.

### 2.3 Component-level measurements — train only

From `docs/RESULTS.md` `5:

| Component | Reported metric | Scope |
|---|---|---|
| Silver SQL compiler + verifier | **1,671 / 1,689** compilable training cases had matching execution results | G1-derived **silver** SQL, not human gold SQL |
| Heuristic Schema Filter `S1` | **91.5%** of 1,671 cases preserved every required SQL column; mean **11.1** columns | In-sample component test, no LLM |
| Heuristic Schema Filter `S2` | **99.6%** preserved every required SQL column; mean **16.9** columns | In-sample component test, no LLM |
| Full schema | **100%**, mean **42.3** columns | Trivial reference coverage |
| SQL lint false alarms on verified SQL | **0 / 5,013** | Only tested valid examples and mutation suite |
| Synthetic SQL mutations | Reported high detection rates for several date/join/validity/finance error types | Synthetic negatives, not unseen model SQL |
| Multi-task/multi-format SFT examples | **20,379** examples at `--n-aug 1` | Dataset construction count, not model accuracy |
| G1 log-based 5-fold calibration | PA **0.9997**, Brier **0.0011**, ECE **0.022** | Optimistic: G1 rules were authored with the entire train split in view |

**Careful interpretation:** high column recall or lint mutation detection does not imply high **G4 downstream PA**. A correctly retrieved schema does not force a generated SQL to apply the correct business rule. No full schema-filter/G4/selector ablation on unseen businesses is established in the current materials.

## 3. Run the baseline locally

```bash
# Reproduce tests in the current Python environment
python -m pip install -r requirements.txt pytest
python -m pytest -q tests/

# Rule-only inference and local PA on the labeled train split
python -m datamate.predict --data data/train \
  --out runs/train/submission.jsonl --llm none
python scripts/check_submission.py \
  runs/train/submission.jsonl data/train/questions.jsonl data/train/answers.jsonl

# Public-test format validation (there are no public gold labels)
python -m datamate.predict --data data/public_test \
  --out runs/public/submission.jsonl --llm none
python scripts/check_submission.py \
  runs/public/submission.jsonl data/public_test/questions.jsonl
```

For component tests on training data:

```bash
python scripts/build_sql_silver.py data/train
python scripts/eval_schema_filter.py data/train
python scripts/eval_sql_checks.py data/train
```

Some scripts generate or overwrite files in `artifacts/`; use a disposable checkout if you need to compare against committed artifacts. Remember that these measure **components** and **known training examples**.

## 4. How to validate a real LLM experiment

Before each GPU run, save:

1. **Source commit and branch**; training data fold IDs; exact model ID, release/licence, parameter totals and adapter SHA.
2. **Model variant** (BF16/FP8/AWQ), vLLM/Transformers versions, GPU count and inference flags; `--require-llm` to disallow silent rule-only fallbacks.
3. **Prediction artifacts** and SHA-256; every per-question log including selected candidates, SQL, schema, execution errors and timing.
4. **Evaluation method**: train, noisy held-out fold, public leaderboard, or final offline Docker rerun. Never mix them in one table without labels.
5. **Paired ablations** on the **same examples**: rule-only; G1+G2; G1+G3; G1+G4; all generators; ablated selection; ablated schema filtering; fixed-vs-calibrated abstention.

Suggested local LLM sanity run:

```bash
python -m datamate.predict --data data/train \
  --out runs/gpu_smoke/submission.jsonl \
  --llm vllm --model /models/llm --lora models/adapter \
  --tp 2 --require-llm --limit 10
```

Then use `--fold 0` or a systematically held-out business fold to measure cross-business performance. **Do not** train or tune on public/private-test samples; use only allowed training data and legitimate validation protocols.

The repository provides `scripts/validate.py` for PA/Brier/ECE and reliability fitting from logs, and `scripts/replay_selector.py` for replaying a selector over an existing run. A replay confirms the **selector change on those logs**, not what an independently rerun model would generate.

## 5. Competition compliance checklist

- **Model eligibility:** inference models released before **1 Oct 2026** with compatible licenses; disclose all inference models; **sum of all model parameters ≤15B**, even if sequentially loaded; count adapters and auxiliary model weights. Quantization reduces memory but **not** the logical model parameter budget.
- **Offline:** model weights and software present in the final image; no network use, downloads or external APIs during inference.
- **Dataset integrity:** train and synthetic/allowed external data only for training; no public/private-test labels, inferred labels, answer harvesting, or question/business ID-specific hardcoding.
- **Runtime:** full private test within **6 hours** on the specified **2× RTX 5090** reference system, including model startup.
- **Determinism:** the challenge expects a close match between uploaded and organizer-rerun PA (absolute difference at most **0.03**); inspect logs and seeds; do not claim compliance based solely on temperature zero.
- **Output:** each input ID represented once with valid type/answer; `/result/submission.jsonl` and `/result/logs/` generated by `predict.sh`.

**The presence of Docker build recipes or a Phase-A adapter does not establish any of these end-to-end results.**

## 6. What still needs to be measured

| Question | Status in this branch | Required evidence |
|---|---|---|
| Is Phase-A LoRA a consistent net improvement over G1 on unseen businesses? | **Not established** | G1 vs LoRA on multiple independently held-out business folds with noisy and clean queries |
| Does each G4 format add complementary correct SQLs? | **Not established** | Oracle@k, unique-correct rate, final PA by format and cluster |
| Does SQL selection outperform majority execution consistency? | **Not established** | Controlled G4 selection ablation, same candidate pool |
| Is `tau=0.6` optimal? | **Not established** | Held-out business calibration and confidence/coverage/PA curve |
| Does GRPO improve abstention without over-abstaining? | **Not established** | Trained GRPO checkpoint, policy evaluation on untouched businesses |
| Does the final Docker image complete the private run offline within six hours? | **Not established** | Timed and hashed compatible-hardware, network-disabled rerun |
| What is the official score of the current final-named submission? | **Not established** | Exact uploaded file hash matched with official score record |

## 7. Document discipline

`docs/RESULTS.md` is retained as a **historical lab notebook** (including old submission debugging and experiments). Its dated descriptions should not be read as a single simultaneous deployment status. This page is the **interpretation and evidence guide**; `README.md` is the **public entry point**; technical algorithms belong in `XIYAN_ADAPTATION.md`, not in copied prose across multiple files.
