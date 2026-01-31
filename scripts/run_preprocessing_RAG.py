#!/usr/bin/env python3
"""
CLI Script for MultiHopRAG Preprocessing (RAG -> TISER-like)

This script converts the official MultiHopRAG benchmark samples into a
TISER-like schema to be used by the evaluation scripts.

Key behaviors (paper-aligned):
- Omits null queries (empty/None)
- Builds a grounding "context" from evidence_list
- Outputs JSONL (default) or JSON list

Examples:
  # Basic conversion JSON -> JSONL
  python scripts/run_rag_preprocessing.py \
    --input data/raw/multihoprag.json \
    --output data/processed/multihoprag_tiserlike.jsonl

  # Limit to first 200 examples for quick debug
  python scripts/run_rag_preprocessing.py \
    --input data/raw/multihoprag.json \
    --output data/processed/multihoprag_tiserlike_200.jsonl \
    --max-samples 200

  # Keep evidences in the output
  python scripts/run_rag_preprocessing.py \
    --input data/raw/multihoprag.json \
    --output data/processed/multihoprag_tiserlike_with_ev.jsonl \
    --keep-evidence

  # Output as a single JSON list instead of JSONL
  python scripts/run_rag_preprocessing.py \
    --input data/raw/multihoprag.json \
    --output data/processed/multihoprag_tiserlike.json \
    --out-format json
"""

import argparse
import sys
import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import PROJECT_ROOT, RAW_DIR, PROCESSED_DIR
from src.data.preprocessingRAG import preprocess_multihoprag_to_tiserlike_jsonl

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")


def _resolve_input_path(p: Path) -> Path:
    if p.is_absolute() or p.parent != Path("."):
        return p
    return RAW_DIR / p


def _resolve_output_path(p: Path) -> Path:
    if p.is_absolute() or p.parent != Path("."):
        return p
    return PROCESSED_DIR / p


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MultiHopRAG into a TISER-like JSONL/JSON schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to MultiHopRAG raw file (.json or .jsonl)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output path (.jsonl or .json)")

    parser.add_argument("--dataset-name", type=str, default="multihoprag", help="dataset_name to store in each sample")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for debugging")

    parser.add_argument("--keep-null", action="store_true", help="If set, keeps null queries (default: omitted).")


    parser.add_argument("--out-format", choices=["jsonl", "json"], default=None,
                        help="Force output format. If omitted, inferred from output extension.")

    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed logging")

    args = parser.parse_args()
    setup_logging(verbose=not args.quiet)

    inp = _resolve_input_path(args.input)
    out = _resolve_output_path(args.output)

    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    out.parent.mkdir(parents=True, exist_ok=True)

    # Infer out format if not forced
    out_format = args.out_format
    if out_format is None:
        suf = out.suffix.lower()
        if suf == ".jsonl":
            out_format = "jsonl"
        elif suf == ".json":
            out_format = "json"
        else:
            raise ValueError("Output must end with .jsonl or .json (or set --out-format).")

    logger.info(f"\n{'='*80}")
    logger.info("MULTIHOPRAG PREPROCESSING")
    logger.info(f"Input:  {inp}")
    logger.info(f"Output: {out} ({out_format})")
    logger.info(f"Dataset name: {args.dataset_name}")
    if args.max_samples is not None:
        logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"{'='*80}\n")

    original, kept = preprocess_multihoprag_to_tiserlike_jsonl(
    input_path=str(inp),
    output_path=str(out),
    dataset_name=args.dataset_name,
    omit_null_queries=not args.keep_null, 
    verbose=not args.quiet,
    )
    print(f"Original: {original} | Kept: {kept}")


if __name__ == "__main__":
    main()
