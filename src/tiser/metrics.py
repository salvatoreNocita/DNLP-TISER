# src/tiser/metrics.py

from __future__ import annotations

from typing import List, Tuple, Any
from collections import Counter


def _normalize_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return " ".join(s.strip().lower().split())


def exact_match(pred: str, gold: str) -> float:
    """
    Exact Match binario (0/1) dopo normalizzazione semplice.
    """
    return 1.0 if _normalize_text(pred) == _normalize_text(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """
    Token-level F1: standard per QA.
    """
    pred_tokens = _normalize_text(pred).split()
    gold_tokens = _normalize_text(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_em_f1(pairs: List[Tuple[str, str]]) -> Tuple[float, float]:
    """
    pairs: lista di (pred, gold).
    Restituisce (EM_avg, F1_avg).
    """
    if not pairs:
        return 0.0, 0.0

    em_sum = 0.0
    f1_sum = 0.0
    for pred, gold in pairs:
        em_sum += exact_match(pred, gold)
        f1_sum += token_f1(pred, gold)

    n = len(pairs)
    return em_sum / n, f1_sum / n