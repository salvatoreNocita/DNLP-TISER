# src/tiser/parsing.py

from __future__ import annotations

from typing import Optional


def _extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """
    Estrae la sottostringa tra start_tag ed end_tag, se presenti.
    Restituisce None se i tag non si trovano.
    """
    text_lower = text.lower()
    start_tag_lower = start_tag.lower()
    end_tag_lower = end_tag.lower()

    start_idx = text_lower.find(start_tag_lower)
    if start_idx == -1:
        return None
    start_idx += len(start_tag)

    end_idx = text_lower.find(end_tag_lower, start_idx)
    if end_idx == -1:
        return None

    content = text[start_idx:end_idx]
    return content.strip()


def extract_answer(text: str) -> str:
    """
    Estrae il contenuto di <answer>...</answer> dall'output del modello.
    Se i tag non sono presenti, fa un fallback rozzo (ultima riga non vuota).
    """
    ans = _extract_between(text, "<answer>", "</answer>")
    if ans is not None and ans != "":
        return ans

    # Fallback: prendiamo l'ultima riga non vuota del testo
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[-1]