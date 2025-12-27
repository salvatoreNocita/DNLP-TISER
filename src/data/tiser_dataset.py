# src/data/tiser_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterable
import json


@dataclass
class TiserExample:
    dataset_name: str
    question_id: str
    question: str
    answer: str
    prompt: str
    output: Optional[str] = None  # usato solo per il train (SFT), di solito None per il test


def _normalize_item(raw: dict) -> TiserExample:
    """
    Converte un dict grezzo del JSON TISER (train o test) in un TiserExample.
    Alcuni campi potrebbero mancare nel test (es. 'output').
    """
    dataset_name = raw.get("dataset_name", "")
    qid = str(raw.get("question_id", ""))
    question = raw.get("question", "")
    answer = raw.get("answer", "")
    prompt = raw.get("prompt", "")
    output = raw.get("output")  # può essere None

    return TiserExample(
        dataset_name=dataset_name,
        question_id=qid,
        question=question,
        answer=answer,
        prompt=prompt,
        output=output,
    )


def load_tiser_file(
    path: Path,
    max_examples: Optional[int] = None,
    dataset_filter: Optional[Iterable[str]] = None,
) -> List[TiserExample]:
    """
    Carica un file JSON TISER (train/test/subset) e restituisce una lista di TiserExample.

    - path: path al file .json
    - max_examples: se specificato, tronca alla prima N righe
    - dataset_filter: lista di dataset_name da tenere (es. ["tgqa_split_test"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TISER file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of dicts in {path}, got {type(data)}")

    examples: List[TiserExample] = []
    ds_filter_set = set(dataset_filter) if dataset_filter is not None else None

    for raw in data:
        ex = _normalize_item(raw)
        if ds_filter_set is not None and ex.dataset_name not in ds_filter_set:
            continue
        examples.append(ex)
        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples