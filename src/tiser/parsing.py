from __future__ import annotations

import re
from typing import Optional, Callable


# Regex to intercept any structural tag.
# Useful for determining where a section ends if the model forgets to close it.
_TAG_RE = re.compile(r"</?(reasoning|timeline|reflection|adjustments|answer)\b[^>]*>", re.IGNORECASE)


def _clean_text_neutral(s: str) -> str:
    """
    Neutral cleaning: removes only outer whitespace.
    Preserves internal bullet points, quotes, and structure.
    """
    return s.strip()


def _clean_answer(s: str) -> str:
    """
    Aggressive cleaning for the final answer.
    Removes leading bullet points, wrapping quotes, and normalizes internal whitespace.
    """
    s = s.strip()
    # Remove common leading bullets
    s = re.sub(r"^\s*[-•*]+\s*", "", s)
    
    # Remove wrapping quotes if present
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        s = s[1:-1].strip()

    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s).strip()

    low = s.lower()
    if low in {"none", "null", "n/a", "na"}:
        return ""
    
    return s


def _extract_between_ci(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Strict extraction: looks for the exact start-end pair case-insensitively."""
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

    return text[start:end]


def extract_section(text: str, tag_name: str) -> str:
    """
    Extracts long sections (reasoning, timeline, reflection) based on XML structure.
    Does NOT use arbitrary character limits.
    """
    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"
    lower = text.lower()

    # --- CASE 1: Perfect Pair (<tag>...</tag>) ---
    strict = _extract_between_ci(text, open_tag, close_tag)
    if strict is not None:
        return _clean_text_neutral(strict)

    # --- CASE 2: Missing closing tag (<tag>...) ---
    # Take from the opening tag until the NEXT structural tag.
    start_idx = lower.find(open_tag.lower())
    if start_idx != -1:
        content_start = start_idx + len(open_tag)
        chunk = text[content_start:]
        
        # Search for the start of the next tag (e.g., <timeline>, <answer>) to stop
        m = _TAG_RE.search(chunk)
        if m:
            chunk = chunk[:m.start()]
            
        return _clean_text_neutral(chunk)

    # --- CASE 3: Missing opening tag (...</tag>) ---
    # This is critical. Instead of taking arbitrary chars, we look for
    # the end of the previous block, identified by the '>' character.
    end_idx = lower.rfind(close_tag.lower())
    if end_idx != -1:
        before = text[:end_idx]
        
        # Look for the last '>' before this block.
        # Example: "... </reasoning> CONTENT </timeline>"
        # The last '>' belongs to </reasoning>.
        last_bracket = before.rfind(">")
        
        if last_bracket != -1:
            chunk = before[last_bracket + 1:]
        else:
            # If no '>' is found, it means we are at the start of the text
            # (or the prompt had no tags). Take everything in 'before'.
            chunk = before

        return _clean_text_neutral(chunk)

    return ""


def extract_answer(text: str) -> str:
    """
    Extracts the final Answer.
    Maintains heuristic logic (last lines/cap at 500 chars) 
    because answers are often unstructured or short.
    """
    # 1. Strict extraction
    strict = _extract_between_ci(text, "<answer>", "</answer>")
    if strict:
        return _clean_answer(strict)

    lower = text.lower()

    # 2. Open (<answer>...)
    a = lower.find("<answer>")
    if a != -1:
        chunk = text[a + len("<answer>") :]
        m = _TAG_RE.search(chunk)
        if m:
            chunk = chunk[: m.start()]
        
        # Cap is useful here: avoid dragging long comments or hallucinations
        chunk = chunk.strip()[:500] 
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        return _clean_answer(lines[0] if lines else chunk)

    # 3. Close (...</answer>)
    b = lower.rfind("</answer>")
    if b != -1:
        before = text[:b].strip()
        # Take the last non-empty line (standard heuristic for CoT)
        lines = [ln.strip() for ln in before.splitlines() if ln.strip()]
        if not lines:
            return ""
        # Limit length for safety
        return _clean_answer(lines[-1][:500])

    return ""