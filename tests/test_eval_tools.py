"""Data-free tests of the evaluation helpers used by the Colab notebook (validate, noisy split, submission diff)."""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _fake_run(d):
    """Six shops, one question each: G1 right on all; G2 right on 4, wrong on 1, silent on 1; G3 alone on q5."""
    logs = os.path.join(d, "logs")
    os.makedirs(logs)
    answers = []
    for i in range(6):
        qid, gold = f"q{i}", 100 + i
        answers.append({"id": qid, "type": "number", "answer": gold, "alias": []})
        cands = [{"gen": "G1_rule", "intent": "revenue", "answer": {"type": "number", "answer": gold}}] if i < 5 else []
        if i < 4:
            cands.append({"gen": "G2_frame", "intent": "revenue", "answer": {"type": "number", "answer": gold}})
        elif i == 4:
            cands.append({"gen": "G2_frame", "intent": "revenue", "answer": {"type": "number", "answer": 1}})
        else:
            cands.append({"gen": "G3_rewrite", "intent": "revenue", "answer": {"type": "number", "answer": gold}})
        json.dump({"id": qid, "business_id": f"biz_{i}", "question": "?", "candidates": cands},
                  open(os.path.join(logs, qid + ".json"), "w", encoding="utf-8"))
    _write_jsonl(os.path.join(d, "answers.jsonl"), answers)
    return logs, os.path.join(d, "answers.jsonl")


def test_generator_report_separates_llm_from_g1():
    import validate
    with tempfile.TemporaryDirectory() as d:
        logs, ans = _fake_run(d)
        rows, n = validate.generator_report(validate.load(logs, ans))
    assert n == 6
    assert rows["G1_rule"]["n"] == 5 and rows["G1_rule"]["ok"] == 5
    assert rows["G2_frame"]["n"] == 5 and rows["G2_frame"]["ok"] == 4 and rows["G2_frame"]["agree_g1"] == 4
    assert rows["G3_rewrite"]["solo"] == 1 and rows["G3_rewrite"]["solo_ok"] == 1     # answered where G1 was silent


def test_write_reliability_keeps_g1_calibration():
    """Regression: a new --write-reliability file used to contain only the LLM generators (G1_rule was lost)."""
    with tempfile.TemporaryDirectory() as d:
        logs, ans = _fake_run(d)
        base = os.path.join(d, "base.json")
        json.dump({"G1_rule": {"__all__": 0.97, "revenue": 0.98}}, open(base, "w"))
        out = os.path.join(d, "rel.json")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "validate.py"), logs, ans, "--folds", "3",
                            "--out", os.path.join(d, "v"), "--write-reliability", out, "--base-reliability", base],
                           capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        assert r.returncode == 0, r.stderr
        rel = json.load(open(out, encoding="utf-8"))
        assert rel["G1_rule"] == {"__all__": 0.97, "revenue": 0.98}           # untouched without --include-g1
        assert "G2_frame" in rel and "G3_rewrite" in rel
        m = json.load(open(os.path.join(d, "v", "metrics.json"), encoding="utf-8"))
        assert "llm_only" in m and m["generators"]["G2_frame"]["n"] == 5
        assert "LLM generators only" in r.stdout


def test_noisy_split_keeps_labels_and_shops():
    import make_noisy_split
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "train"), os.path.join(d, "noisy")
        os.makedirs(os.path.join(src, "data", "biz_1"))
        open(os.path.join(src, "rulebook.md"), "w", encoding="utf-8").write("# rules\n")
        qs = [{"id": f"q{i}", "business_id": f"biz_{i % 3}", "question": "Tháng 5/2025 cửa hàng bán được bao nhiêu tiền?",
               "as_of_date": "2026-09-14"} for i in range(9)]
        _write_jsonl(os.path.join(src, "questions.jsonl"), qs)
        _write_jsonl(os.path.join(src, "answers.jsonl"),
                     [{"id": q["id"], "type": "number", "answer": i, "alias": []} for i, q in enumerate(qs)])
        st = make_noisy_split.make(src, out, fold=0, folds=3, n=2, level=1.0, keep_unchanged=True)
        nq = [json.loads(l) for l in open(os.path.join(out, "questions.jsonl"), encoding="utf-8")]
        na = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(out, "answers.jsonl"), encoding="utf-8")}
        assert st["shops"] == ["biz_0"] and len(nq) == 6                      # fold 0 of 3: biz_0 only, 2 variants each
        assert all(q["business_id"] == "biz_0" and q["id"].split("~")[0] in {"q0", "q3", "q6"} for q in nq)
        assert {k: v["answer"] for k, v in na.items()} == {q["id"]: int(q["id"].split("~")[0][1:]) for q in nq}
        assert os.path.exists(os.path.join(out, "data", "biz_1")) and os.path.exists(os.path.join(out, "rulebook.md"))
        assert any(q["question"] != qs[0]["question"] for q in nq)            # level 1.0 changes the wording


def test_compare_submissions_kinds():
    import compare_submissions as cs
    old = {"a": {"type": "abstain"}, "b": {"type": "number", "answer": 1}, "c": {"type": "number", "answer": 2},
           "d": {"type": "boolean", "answer": True}}
    new = {"a": {"type": "number", "answer": 5}, "b": {"type": "number", "answer": 1}, "c": {"type": "number", "answer": 3},
           "d": {"type": "abstain"}}
    cnt, rows = cs.compare(old, new)
    assert dict(cnt) == {"newly_answered": 1, "same": 1, "changed": 1, "newly_abstained": 1}
    assert [r["id"] for r in rows] == ["a", "c", "d"]


def test_notebook_push_cell_checks_lfs():
    nb = json.load(open(os.path.join(ROOT, "notebooks", "colab_finetune.ipynb"), encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "apt-get -qq install -y git-lfs" in src                        # Colab has no git-lfs
    assert "version https://git-lfs" in src and "ls-remote" in src         # pointer check before push, branch check after
    assert "DATAMATE_G4_SCOPE=all" in src and "make_noisy_split.py" in src


def test_require_llm_fails_loudly_instead_of_rule_only():
    import pytest
    from datamate.llm.backend import NullBackend, make_backend
    assert isinstance(make_backend({"backend": "vllm", "model_path": "/nonexistent/model"}), NullBackend)
    with pytest.raises(RuntimeError):
        make_backend({"backend": "vllm", "model_path": "/nonexistent/model", "require_llm": True})


def test_awq_checkpoint_counts_unpacked_parameters():
    import struct
    import count_params
    with tempfile.TemporaryDirectory() as d:
        hdr = {"l.qweight": {"dtype": "I32", "shape": [64, 8], "data_offsets": [0, 0]},
               "l.qzeros": {"dtype": "I32", "shape": [1, 8], "data_offsets": [0, 0]},
               "l.scales": {"dtype": "F16", "shape": [1, 64], "data_offsets": [0, 0]},
               "norm.weight": {"dtype": "F16", "shape": [10], "data_offsets": [0, 0]}}
        b = json.dumps(hdr).encode()
        open(os.path.join(d, "model.safetensors"), "wb").write(struct.pack("<Q", len(b)) + b)
        json.dump({"quantization_config": {"quant_method": "awq", "bits": 4}}, open(os.path.join(d, "config.json"), "w"))
        assert count_params.count_dir(d) == 64 * 64 + 10       # 4-bit packing undone, quantisation metadata skipped


def test_notebook_has_a_t4_profile():
    nb = json.load(open(os.path.join(ROOT, "notebooks", "colab_finetune.ipynb"), encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "--quant awq --dtype half --enforce-eager" in src and "Qwen/Qwen3-14B-AWQ" in src
    assert "--require-llm" in src and "CAN_TRAIN" in src
    from datamate.predict import main  # noqa: F401  (flags below must exist)
    import datamate.predict as pr
    text = open(pr.__file__, encoding="utf-8").read()
    assert all(f in text for f in ("--dtype", "--enforce-eager", "--require-llm"))
