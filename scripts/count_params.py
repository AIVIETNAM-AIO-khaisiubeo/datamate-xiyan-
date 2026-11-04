"""Count the parameters of every model used at inference (competition rule: <= 15B in total).

Reads only the safetensors headers (no torch, no GPU): a base model directory, merged models and LoRA
adapter directories can be passed together; the script prints each and fails if the sum exceeds the limit.

    python scripts/count_params.py /models/llm [outputs/lora_cte ...] [--limit 15e9]
"""
import argparse
import json
import os
import struct
import sys


QUANT_META = (".qzeros", ".scales", ".g_idx")     # AWQ / GPTQ bookkeeping, not model parameters


def count_file(path: str, bits: int = 0) -> int:
    """bits > 0: AWQ / GPTQ checkpoint whose int32 '*.qweight' tensors pack 32 // bits weights each, so a 4-bit copy of
    a 14.8B model still counts 14.8B (quantisation changes the storage, not the number of parameters)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    total = 0
    for k, v in header.items():
        if k == "__metadata__" or (bits and k.endswith(QUANT_META)):
            continue
        p = 1
        for d in v["shape"]:
            p *= d
        if bits and k.endswith(".qweight"):
            p *= 32 // bits
        total += p
    return total


def _quant_bits(path: str) -> int:
    try:
        q = json.load(open(os.path.join(path, "config.json"), encoding="utf-8")).get("quantization_config") or {}
    except (OSError, ValueError):
        return 0
    return int(q.get("bits") or q.get("w_bit") or 0) if q.get("quant_method") in ("awq", "gptq") else 0


def count_dir(path: str) -> int:
    if not os.path.isdir(path):
        return count_file(path)
    bits = _quant_bits(path)
    return sum(count_file(os.path.join(path, f), bits) for f in os.listdir(path) if f.endswith(".safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--limit", type=float, default=15e9)
    a = ap.parse_args()
    tot = 0
    for p in a.paths:
        n = count_dir(p)
        tot += n
        print(f"{n / 1e9:8.3f}B  {p}")
    print(f"{tot / 1e9:8.3f}B  total (limit {a.limit / 1e9:.1f}B)")
    if tot > a.limit:
        print("OVER THE PARAMETER LIMIT")
        sys.exit(1)


if __name__ == "__main__":
    main()
