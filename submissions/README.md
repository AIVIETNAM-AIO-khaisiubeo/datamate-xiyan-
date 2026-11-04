# Public-test submission files and their leaderboard scores

Every file is a complete `submission.jsonl` for the 1,000 public questions. The 8-character suffix is the start of the
file's sha256 **as stored on GitHub** (LF line endings). A Windows checkout may rewrite the line endings and change the
local hash; compare the hash of the downloaded file, not of a working copy.

| File | sha256 | Leaderboard PA | How it was made | Run (tag) |
|---|---|---:|---|---|
| (not kept) | — | 0.561 | G1 rule-only, first submission | — |
| `public_test_rule_only.jsonl` | `080626d9` | **0.912** | G1 rule-only after label-free public error analysis | `main` |
| `public_test_lora_phasea_old_selector_8f3c7d3b.jsonl` | `8f3c7d3b` | **0.777** | G1 + Qwen3-14B + LoRA (A100), old selector, G4 broken | `run-0.777` |
| `public_test_lora_phasea_7b_8124e92b.jsonl` | `8124e92b` | **0.895** | Same adapter, Qwen3-14B-AWQ on T4 (notebook 7b), new selector, G4 fixed | `run-0.895` |
| `public_test_lora_phasea_7a_replay_28e7e73c.jsonl` | `28e7e73c` | not submitted | New selector replayed on the 0.777 logs (diagnostic) | `run-0.895` |
| `public_test_lora_phasea_final_de716e81.jsonl` | `de716e81` | **0.909** | Notebook 7c: calibrated dissent + guards, CPU replay of the 0.895 logs | `run-0.909` |

A new file never inherits an earlier score: two files differ in how many questions they answer and how. Before
submitting, compare a new file with `public_test_rule_only.jsonl` (`scripts/compare_submissions.py`); each differing
question can move the score by at most ±1 (±2 if the answer changed). The analysis of each score is in
[`docs/RESULTS.md`](../docs/RESULTS.md) §2.
