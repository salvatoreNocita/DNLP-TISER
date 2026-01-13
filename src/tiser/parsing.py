# src/tiser/parsing.py
from __future__ import annotations

import re
from typing import Optional


_TAG_RE = re.compile(r"</?(reasoning|timeline|reflection|answer)\b[^>]*>", re.IGNORECASE)


def _extract_between_ci(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Case-insensitive extraction between tags. Returns None if not found."""
    lower = text.lower()
    s = start_tag.lower()
    e = end_tag.lower()

    start = lower.find(s)
    if start == -1:
        return None
    start += len(s)

    end = lower.find(e, start)
    if end == -1:
        return None

    return text[start:end].strip()


def _clean_answer(s: str) -> str:
    """Neutral cleanup: strip, remove surrounding quotes, remove leading bullets, collapse whitespace."""
    s = s.strip()

    # Remove common leading bullets
    s = re.sub(r"^\s*[-•*]+\s*", "", s)

    # Remove wrapping quotes if present
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        s = s[1:-1].strip()

    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s).strip()

    low = s.strip().lower()
    if low in {"none", "null", "n/a", "na"}:
        return ""
    
    return s


def extract_answer(text: str) -> str:
    """
    Robust but non-task-aware answer extraction.

    Priority:
    1) content inside <answer>...</answer>
    2) if <answer> exists without </answer>, take a bounded chunk after <answer>
    3) if </answer> exists without <answer>, take a bounded chunk before </answer>
    4) otherwise return "" (fail-closed)
    """
    # (1) strict
    strict = _extract_between_ci(text, "<answer>", "</answer>")
    if strict:
        return _clean_answer(strict)

    lower = text.lower()

    # (2) open <answer> without close: take bounded chunk
    a = lower.find("<answer>")
    if a != -1:
        chunk = text[a + len("<answer>") :]
        # stop at next tag if any (reasoning/timeline/reflection/answer)
        m = _TAG_RE.search(chunk)
        if m:
            chunk = chunk[: m.start()]
        # bound length to avoid swallowing prompt
        chunk = chunk.strip()
        chunk = chunk[:500]  # adjustable safety cap
        # take first non-empty line if multiline
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        return _clean_answer(lines[0] if lines else chunk)

    # (3) close </answer> without open: take bounded chunk before it
    b = lower.rfind("</answer>")
    if b != -1:
        before = text[:b].strip()
        # take last few lines, pick last non-empty
        lines = [ln.strip() for ln in before.splitlines() if ln.strip()]
        if not lines:
            return _clean_answer(before[:500])
        # bound to last line (usually the answer)
        return _clean_answer(lines[-1][:500])

    # (4) fail-closed
    return ""