"""Local re-implementation of the official scoring (Penalized Accuracy)."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple


def _norm_str(s) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    return re.sub(r"[\s\W_]+", "", s)


def _num_eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))


def _str_eq(a, ref, aliases: Iterable = ()) -> bool:
    if not isinstance(a, str):
        return False
    cands = [ref] + [x for x in (aliases or []) if isinstance(x, str)]
    na = _norm_str(a)
    return any(na == _norm_str(c) for c in cands if c is not None)


def _elem_eq(a, ref, aliases=()) -> bool:
    if isinstance(ref, str):
        return _str_eq(a, ref, aliases)
    if isinstance(ref, bool):
        return a is ref
    if isinstance(ref, (int, float)):
        return _num_eq(a, ref)
    return a == ref


def score_one(pred: Optional[dict], ref: dict) -> int:
    """+1 correct, -1 wrong, 0 abstain."""
    if pred is None or pred.get("type") in (None, "abstain"):
        return 0
    pt, rt = pred.get("type"), ref.get("type")
    if pt != rt:
        return -1
    if rt == "unanswerable":
        return 1
    a, r = pred.get("answer"), ref.get("answer")
    al = ref.get("alias") or []
    if rt in ("number", "percent"):
        return 1 if _num_eq(a, r) else -1
    if rt == "boolean":
        return 1 if (isinstance(a, bool) and a == r) else -1
    if rt == "date":
        return 1 if (isinstance(a, str) and a == r) else -1
    if rt == "string":
        return 1 if _str_eq(a, r, al) else -1
    if rt == "list":
        if not isinstance(a, list) or len(a) != len(r):
            return -1
        for i, (x, y) in enumerate(zip(a, r)):
            ali = al[i] if i < len(al) and isinstance(al[i], list) else []
            if not _elem_eq(x, y, ali):
                return -1
        return 1
    if rt == "table":
        if not isinstance(a, dict) or "rows" not in a:
            return -1
        rr, ar = r.get("rows", []), a.get("rows", [])
        if len(rr) != len(ar):
            return -1
        used = [False] * len(rr)
        for row in ar:
            ok = False
            for j, ref_row in enumerate(rr):
                if used[j] or len(row) != len(ref_row):
                    continue
                ali = al[j] if j < len(al) and isinstance(al[j], list) else []
                if all(_elem_eq(x, y, ali if i == 0 else ()) for i, (x, y) in enumerate(zip(row, ref_row))):
                    used[j] = ok = True
                    break
            if not ok:
                return -1
        return 1
    return 1 if a == r else -1


def penalized_accuracy(preds: Dict[str, dict], refs: Dict[str, dict]) -> Tuple[float, Dict[str, int]]:
    c = {"correct": 0, "wrong": 0, "abstain": 0}
    for qid, ref in refs.items():
        s = score_one(preds.get(qid), ref)
        c["correct" if s == 1 else "wrong" if s == -1 else "abstain"] += 1
    n = max(1, len(refs))
    return (c["correct"] - c["wrong"]) / n, c


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
