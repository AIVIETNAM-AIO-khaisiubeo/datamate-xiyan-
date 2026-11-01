"""LLM backends.

* ``VLLMBackend``  – offline vLLM engine (tensor parallel over both RTX 5090s), batched,
                     seeded, optional JSON-schema guided decoding.  Used inside Docker.
* ``HFBackend``    – plain transformers fallback (slow, single GPU / CPU).
* ``NullBackend``  – no model: the agent runs rule-only (CPU) mode.
* ``MockBackend``  – deterministic canned outputs for unit tests.

All backends expose ``generate(prompts, n=1, temperature=0.0, max_tokens=..., json_schema=None,
seed=0) -> List[List[str]]``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional


class LLMBackend:
    name = "base"
    available = False

    def generate(self, prompts: List[str], n: int = 1, temperature: float = 0.0, max_tokens: int = 512,
                 json_schema: Optional[dict] = None, seed: int = 0, stop: Optional[List[str]] = None) -> List[List[str]]:
        raise NotImplementedError

    def chat_prompt(self, system: str, user: str) -> str:
        return f"{system}\n\n{user}"


class NullBackend(LLMBackend):
    name = "none"
    available = False

    def generate(self, prompts, n=1, temperature=0.0, max_tokens=512, json_schema=None, seed=0, stop=None):
        return [[] for _ in prompts]


class MockBackend(LLMBackend):
    name = "mock"
    available = True

    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def generate(self, prompts, n=1, temperature=0.0, max_tokens=512, json_schema=None, seed=0, stop=None):
        return [[self.fn(p) for _ in range(n)] for p in prompts]


class VLLMBackend(LLMBackend):
    name = "vllm"

    def __init__(self, model_path: str, tensor_parallel_size: int = 2, max_model_len: int = 16384,
                 gpu_memory_utilization: float = 0.88, quantization: Optional[str] = None,
                 enable_thinking: bool = False, dtype: str = "auto"):
        from vllm import LLM  # noqa: import inside so CPU mode never needs vllm
        t0 = time.time()
        kw = dict(model=model_path, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len,
                  gpu_memory_utilization=gpu_memory_utilization, seed=0, trust_remote_code=True,
                  enforce_eager=False, dtype=dtype)
        if quantization:
            kw["quantization"] = quantization
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.enable_thinking = enable_thinking
        self.available = True
        self.load_seconds = time.time() - t0

    def chat_prompt(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=self.enable_thinking)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, n=1, temperature=0.0, max_tokens=512, json_schema=None, seed=0, stop=None):
        from vllm import SamplingParams
        kw = dict(n=n, temperature=temperature, top_p=1.0 if temperature == 0 else 0.95, max_tokens=max_tokens,
                  seed=seed, stop=stop)
        if json_schema is not None:
            try:
                from vllm.sampling_params import GuidedDecodingParams
                kw["guided_decoding"] = GuidedDecodingParams(json=json_schema)
            except Exception:
                pass
        sp = SamplingParams(**kw)
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [[c.text for c in o.outputs] for o in outs]


class HFBackend(LLMBackend):
    name = "hf"

    def __init__(self, model_path: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=device,
                                                          trust_remote_code=True)
        self.torch = torch
        self.available = True

    def chat_prompt(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, n=1, temperature=0.0, max_tokens=512, json_schema=None, seed=0, stop=None):
        out = []
        for p in prompts:
            ids = self.tok(p, return_tensors="pt").to(self.model.device)
            res = []
            for i in range(n):
                self.torch.manual_seed(seed + i)
                g = self.model.generate(**ids, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                        temperature=max(temperature, 1e-5), top_p=0.95)
                res.append(self.tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
            out.append(res)
        return out


def make_backend(cfg: Dict) -> LLMBackend:
    kind = (cfg or {}).get("backend", os.environ.get("DATAMATE_LLM", "none"))
    if kind in (None, "", "none"):
        return NullBackend()
    path = cfg.get("model_path") or os.environ.get("DATAMATE_MODEL", "/models/llm")
    if not os.path.exists(path):
        print(f"[llm] model path {path} not found -> rule-only mode")
        return NullBackend()
    try:
        if kind == "vllm":
            return VLLMBackend(path, tensor_parallel_size=int(cfg.get("tensor_parallel_size", 2)),
                               max_model_len=int(cfg.get("max_model_len", 16384)),
                               gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.88)),
                               quantization=cfg.get("quantization"),
                               enable_thinking=bool(cfg.get("enable_thinking", False)))
        if kind == "hf":
            return HFBackend(path)
    except Exception as e:  # never let a GPU problem kill the run: degrade to rule-only
        print(f"[llm] failed to start {kind}: {e} -> rule-only mode")
    return NullBackend()


def extract_json(text: str) -> Optional[dict]:
    """First JSON object in a model output (tolerates ```json fences and <think> blocks)."""
    if not text:
        return None
    t = text
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    t = t.replace("```json", "```")
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("{"):
                t = p
                break
    i = t.find("{")
    while i >= 0:
        depth = 0
        for j in range(i, len(t)):
            if t[j] == "{":
                depth += 1
            elif t[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[i:j + 1])
                    except json.JSONDecodeError:
                        break
        i = t.find("{", i + 1)
    return None
