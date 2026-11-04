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
import sys
import time
from typing import Callable, Dict, List, Optional


class LLMBackend:
    name = "base"
    available = False
    family = "none"              # model identity: candidates of one family are not independent votes
    last_usage: List = []         # [(prompt_tokens, [completion_tokens per sample])] of the last generate()

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

    family = "mock"

    def generate(self, prompts, n=1, temperature=0.0, max_tokens=512, json_schema=None, seed=0, stop=None):
        outs = [[self.fn(p) for _ in range(n)] for p in prompts]
        self.last_usage = [(len(p) // 4, [len(x) // 4 for x in o]) for p, o in zip(prompts, outs)]
        return outs


class VLLMBackend(LLMBackend):
    name = "vllm"

    def __init__(self, model_path: str, tensor_parallel_size: int = 2, max_model_len: int = 16384,
                 gpu_memory_utilization: float = 0.88, quantization: Optional[str] = None,
                 enable_thinking: bool = False, dtype: str = "auto", lora_path: Optional[str] = None,
                 enforce_eager: bool = False):
        from vllm import LLM  # noqa: import inside so CPU mode never needs vllm
        t0 = time.time()
        # T4 / V100 (no bf16): dtype="half"; enforce_eager=True saves the memory of CUDA graphs on a 16 GB card
        kw = dict(model=model_path, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len,
                  gpu_memory_utilization=gpu_memory_utilization, seed=0, trust_remote_code=True,
                  enforce_eager=enforce_eager, dtype=dtype)
        if quantization:
            kw["quantization"] = quantization
        self.lora = None
        if lora_path:
            # fine-tuned LoRA adapter served next to the base weights (no 30 GB merged copy needed)
            from vllm.lora.request import LoRARequest
            kw.update(enable_lora=True, max_lora_rank=_lora_rank(lora_path), max_loras=1)
            self.lora = LoRARequest("datamate", 1, os.path.abspath(lora_path))
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.enable_thinking = enable_thinking
        self.available = True
        self.family = os.path.basename(os.path.normpath(model_path)) + \
            (("+" + os.path.basename(os.path.normpath(lora_path))) if lora_path else "")
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
            kw.update(_json_constraint(json_schema))
        sp = SamplingParams(**kw)
        outs = self.llm.generate(prompts, sp, use_tqdm=False, lora_request=self.lora)
        self.last_usage = [(len(o.prompt_token_ids or []), [len(c.token_ids) for c in o.outputs]) for o in outs]
        return [[c.text for c in o.outputs] for o in outs]


class HFBackend(LLMBackend):
    name = "hf"

    def __init__(self, model_path: str, device: str = "auto", lora_path: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=device,
                                                          trust_remote_code=True)
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.torch = torch
        self.available = True
        self.family = os.path.basename(os.path.normpath(model_path)) + \
            (("+" + os.path.basename(os.path.normpath(lora_path))) if lora_path else "")

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
    # require_llm: an evaluation run must fail loudly instead of silently becoming rule-only (e.g. vLLM on a T4)
    strict = bool(cfg.get("require_llm")) or os.environ.get("DATAMATE_REQUIRE_LLM") == "1"

    def fallback(msg):
        if strict:
            raise RuntimeError(f"[llm] {msg} (require_llm is set, so the run stops instead of going rule-only)")
        print(f"[llm] {msg} -> rule-only mode", file=sys.stderr)
        return NullBackend()

    if not os.path.exists(path):
        return fallback(f"model path {path} not found")
    lora = cfg.get("lora_path") or None
    if lora and not os.path.exists(os.path.join(lora, "adapter_config.json")):
        return fallback(f"LoRA adapter {lora} not found (no adapter_config.json)")
    try:
        if kind == "vllm":
            be = VLLMBackend(path, tensor_parallel_size=int(cfg.get("tensor_parallel_size", 2)),
                             max_model_len=int(cfg.get("max_model_len", 16384)),
                             gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.88)),
                             quantization=cfg.get("quantization"), dtype=cfg.get("dtype") or "auto",
                             enable_thinking=bool(cfg.get("enable_thinking", False)), lora_path=lora,
                             enforce_eager=bool(cfg.get("enforce_eager", False)))
            print(f"[llm] vLLM ready: {be.family} in {be.load_seconds:.0f}s", file=sys.stderr)
            return be
        if kind == "hf":
            return HFBackend(path, lora_path=lora)
    except Exception as e:  # never let a GPU problem kill the run: degrade to rule-only
        return fallback(f"failed to start {kind}: {type(e).__name__}: {e}")
    return NullBackend()


def _json_constraint(schema: dict) -> dict:
    """SamplingParams kwargs that force JSON output: vLLM >= 0.11 'structured_outputs', older 'guided_decoding'."""
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(json=schema)}
    except ImportError:
        print("[llm] this vLLM has no JSON-constrained decoding: outputs are parsed without a schema")
        return {}


def _lora_rank(path: str) -> int:
    """vLLM needs max_lora_rank >= the adapter's rank (supported sizes 8/16/32/64/128/256)."""
    try:
        r = int(json.load(open(os.path.join(path, "adapter_config.json"), encoding="utf-8")).get("r", 16))
    except Exception:  # noqa: BLE001
        r = 64
    return next(s for s in (8, 16, 32, 64, 128, 256) if s >= r)


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
