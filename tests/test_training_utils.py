"""Data-free tests of the fine-tuning / adapter plumbing (no torch or GPU needed)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeTok:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + ("<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n"))


def test_prompt_completion_matches_inference_prompt():
    from training.sft_lora import build_example
    ex = build_example({"system": "S", "user": "U", "target": "SELECT 1\n```"}, FakeTok())
    assert ex["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")   # thinking disabled, as at inference
    assert ex["completion"] == "SELECT 1\n```<|im_end|>"


def test_lora_rank_and_missing_adapter():
    from datamate.llm.backend import _lora_rank, make_backend
    with tempfile.TemporaryDirectory() as d:
        json.dump({"r": 32}, open(os.path.join(d, "adapter_config.json"), "w"))
        assert _lora_rank(d) == 32
        json.dump({"r": 48}, open(os.path.join(d, "adapter_config.json"), "w"))
        assert _lora_rank(d) == 64
        be = make_backend({"backend": "vllm", "model_path": d, "lora_path": os.path.join(d, "missing")})
        assert not be.available                                                     # degrades to rule-only


def test_cli_lora_option():
    import datamate.predict as pr
    import argparse
    src = open(pr.__file__, encoding="utf-8").read()
    assert '"--lora"' in src and 'cfg["llm"]["lora_path"]' in src
