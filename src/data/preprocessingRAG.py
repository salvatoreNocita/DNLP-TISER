"""
RAG Benchmark Preprocessing Module

Currently supports:
- MultiHopRAG official format -> TISER-like JSONL

Output schema (TISER-like):
{
  "dataset_name": "multihoprag",
  "question_id": "...",
  "question": "...",
  "context": "...",
  "answer": "...",
  "question_type": "inference|comparison|temporal|unknown",
  "is_null_query": bool
}

This is designed to feed downstream evaluation scripts that expect:
- question + context -> prompt
- exact-match evaluation vs answer
- breakdown by question_type
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RAGPreprocessor:
    """
    Preprocessor/converter for RAG benchmarks into a TISER-like JSONL schema.
    """

    def __init__(
        self,
        dataset_name: str = "multihoprag",
        omit_null_queries: bool = True,
        verbose: bool = True,
    ):
        self.dataset_name = dataset_name
        self.omit_null_queries = omit_null_queries

        if verbose:
            root = logging.getLogger()
            if not root.handlers:
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def process(self, input_path: str, output_path: str) -> Tuple[int, int]:
        """
        Full pipeline: load -> convert -> (optional filter) -> save JSONL.

        Returns:
            (original_count, written_count)
        """
        in_path = Path(input_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading RAG benchmark from: {in_path}")
        raw = self._load_any_json(in_path)
        original = len(raw)
        logger.info(f"Loaded {original} raw samples")

        converted: List[Dict[str, Any]] = []
        skipped_null = 0
        unknown_types = 0

        for s in raw:
            ex = self.convert_multihoprag_sample(s)

            if self.omit_null_queries and ex.get("is_null_query", False):
                skipped_null += 1
                continue

            if ex.get("question_type") == "unknown":
                unknown_types += 1

            converted.append(ex)

        logger.info(
            f"Converted: {len(converted)} | skipped null: {skipped_null} | unknown types: {unknown_types}"
        )

        logger.info(f"Saving JSONL -> {out_path}")
        self._write_jsonl(out_path, converted)

        return original, len(converted)

    # -------------------------------------------------------------------------
    # MultiHopRAG conversion
    # -------------------------------------------------------------------------

    def convert_multihoprag_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert one MultiHopRAG sample into the TISER-like schema.
        Uses:
          - query / question
          - answer
          - question_type (e.g. inference_query)
          - evidence_list (list of evidence dicts) to build context
        """
        question = sample.get("query") or sample.get("question") or ""
        question = str(question).strip()

        answer = sample.get("answer") or sample.get("gold") or sample.get("gold_answer") or ""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = str(answer).strip()

        qtype_raw = sample.get("question_type") or sample.get("type")
        qtype = self._normalize_multihoprag_question_type(qtype_raw)

        evidence_list = sample.get("evidence_list")
        context = self._build_context_from_evidence_list(evidence_list)

        qid = sample.get("question_id") or sample.get("id") or sample.get("qid")
        qid = self._ensure_id(qid, question)

        is_null = self._detect_null_query(sample)

        return {
            "dataset_name": self.dataset_name,
            "question_id": qid,
            "question": question,
            "context": context,
            "answer": answer,
            "question_type": qtype,
            "is_null_query": is_null,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _normalize_multihoprag_question_type(self, qtype: Any) -> str:
        """
        MultiHopRAG uses:
          - inference_query
          - comparison_query
          - temporal_query
        Normalize to:
          inference | comparison | temporal
        """
        if qtype is None:
            return "unknown"
        t = str(qtype).strip().lower()
        if t in {"inference_query", "inference"}:
            return "inference"
        if t in {"comparison_query", "comparison"}:
            return "comparison"
        if t in {"temporal_query", "temporal"}:
            return "temporal"
        return "unknown"

    def _build_context_from_evidence_list(self, evidence_list: Any) -> str:
        """
        Deterministic grounding builder.

        Each evidence dict may contain:
          - source, title, published_at, fact
        We build a stable multi-block context with clear separators.
        """
        if evidence_list is None:
            return ""

        if not isinstance(evidence_list, list):
            return json.dumps(evidence_list, ensure_ascii=False)

        chunks: List[str] = []
        for i, ev in enumerate(evidence_list, start=1):
            if not isinstance(ev, dict):
                chunks.append(f"[{i}] {str(ev)}")
                continue

            source = str(ev.get("source", "")).strip()
            title = str(ev.get("title", "")).strip()
            published_at = str(ev.get("published_at", "")).strip()
            fact = str(ev.get("fact", "")).strip()

            header_parts: List[str] = []
            if source:
                header_parts.append(f"Source: {source}")
            if title:
                header_parts.append(f"Title: {title}")
            if published_at:
                header_parts.append(f"PublishedAt: {published_at}")

            header = " | ".join(header_parts) if header_parts else f"Evidence {i}"

            if fact:
                chunks.append(f"[{i}] {header}\nFact: {fact}")
            else:
                chunks.append(f"[{i}] {header}\n{json.dumps(ev, ensure_ascii=False)}")

        return "\n\n".join(chunks).strip()

    def _detect_null_query(self, sample: Dict[str, Any]) -> bool:
        """
        Prefer explicit flags if present.
        MultiHopRAG paper mentions omitting null queries; if your benchmark has a flag,
        this will catch it. Otherwise returns False by default.
        """
        for k in ["is_null_query", "null_query", "is_null", "unanswerable"]:
            if k in sample:
                try:
                    return bool(sample[k])
                except Exception:
                    pass
        return False

    def _ensure_id(self, qid: Any, question: str) -> str:
        """
        Ensure stable question_id. If missing, hash the question into a deterministic id.
        """
        if qid is not None and str(qid).strip() != "":
            return str(qid)

        import hashlib
        h = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        return f"{h}"

    def _load_any_json(self, input_path: Path) -> List[Dict[str, Any]]:
        """
        Load either:
        - .jsonl (one JSON object per line)
        - .json (a list, or a dict with keys: data/examples/items)
        """
        suffix = input_path.suffix.lower()

        if suffix == ".jsonl":
            data: List[Dict[str, Any]] = []
            with open(input_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSONL line {line_num}: {e}")
                        raise
            return data

        if suffix == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            if isinstance(obj, list):
                return obj

            if isinstance(obj, dict):
                for k in ["data", "examples", "items"]:
                    if k in obj and isinstance(obj[k], list):
                        return obj[k]

            raise ValueError("Unsupported JSON format: expected list or dict with data/examples/items.")

        raise ValueError(f"Unsupported file extension: {suffix}")

    def _write_jsonl(self, output_path: Path, items: List[Dict[str, Any]]) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for it in items:
                json.dump(it, f, ensure_ascii=False)
                f.write("\n")

def preprocess_multihoprag_to_tiserlike_jsonl(
    input_path: str,
    output_path: str,
    dataset_name: str = "multihoprag",
    omit_null_queries: bool = True,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Wrapper function for MultiHopRAG -> TISER-like preprocessing.
    Returns (original_count, written_count).
    """
    pre = RAGPreprocessor(
        dataset_name=dataset_name,
        omit_null_queries=omit_null_queries,
        verbose=verbose,
    )
    return pre.process(input_path, output_path)
