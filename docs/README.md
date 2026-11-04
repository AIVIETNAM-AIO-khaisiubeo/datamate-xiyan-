# Documentation

Start with the [root README](../README.md) for the scope, quick start, current state, model constraints and a high-level diagram.

## Read by goal

| Goal | Document |
|---|---|
| Understand what each runtime module does and where candidates flow | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Study schema retrieval, XiYan Algorithms 1–3, G1–G4 and training objectives | [XIYAN_ADAPTATION.md](XIYAN_ADAPTATION.md) |
| Interpret all reported scores, splits, leakage risks and reproduction status | [EXPERIMENTS.md](EXPERIMENTS.md) |
| Inspect the historical, detailed numerical log | [RESULTS.md](RESULTS.md) |
| Run a full local GPU experiment / train and calibrate | [GPU_FINETUNE_GUIDE.md](GPU_FINETUNE_GUIDE.md) |
| Use the notebook and fetch/publish adapter weights | [COLAB_GUIDE.md](COLAB_GUIDE.md) |

## Documentation conventions

- **Implemented** means the named code path exists, not that its accuracy was established.
- **Tested** identifies the actual unit/mock/CPU/GPU evaluation path.
- **Reported** refers to an existing project/leaderboard record, not an independent rerun.
- **Proposed** means no completed evidence is shown in the linked branch.
- Training data, public-test inputs, held-out fold results and official leaderboard submissions are **not interchangeable evaluation sets**.

The source of truth for execution is the [agent](../datamate/agent.py), the [CLI](../datamate/predict.py), the [final selector](../datamate/select/selector.py), and the [Docker entry point](../predict.sh). Read those files when in doubt; avoid treating older Colab troubleshooting notes as evidence of current end-to-end performance.
