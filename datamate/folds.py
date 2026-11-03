"""Business-level folds shared by validation, fine-tuning data and evaluation runs.

A shop is in exactly one fold (sorted shop ids, round robin), so "held-out fold k" means the same shops in
scripts/validate.py, training/build_*_sft.py --holdout-fold k and datamate.predict --fold k.
"""
from __future__ import annotations

from typing import Iterable, Set


def fold_shops(shops: Iterable[str], fold: int, folds: int = 5) -> Set[str]:
    return {b for i, b in enumerate(sorted(set(shops))) if i % folds == fold}


def fold_of(shops: Iterable[str], folds: int = 5) -> dict:
    return {b: i % folds for i, b in enumerate(sorted(set(shops)))}
