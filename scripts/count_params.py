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


def count_file(path: str) -> int:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    total = 0
    for k, v in header.items():
        if k == "__metadata__":
            continue
        p = 1
        for d in v["shape"]:
            p *= d
        total += p
    return total


def count_dir(path: str) -> int:
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors")] if os.path.isdir(path) \
        else [path]
    return sum(count_file(f) for f in files)


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
