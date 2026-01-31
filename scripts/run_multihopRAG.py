# scripts/run_multihoprag_table3_eval.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from src.config import (
    RESULTS_DIR,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    get_model_name,
)
from src.models.base_model import LLMWrapper
from src.tiser.metrics import compute_em_f1
from src.tiser.parsing import extract_answer

from src.tiser.prompts import (
    STANDARD_PROMPT_TEMPLATE,
    TISER_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def load_multihoprag_tiserlike(path: Path) -> List[Dict[str, Any]]:
    """
    Load the processed MultiHopRAG file in TISER-like schema.
    Supported:
      - .jsonl (one sample per line)
      - .json (list of samples)
    """
    suf = path.suffix.lower()

    if suf == ".jsonl":
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_num}: {e}")
        return out

    if suf == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError("Expected a JSON list of samples.")
        return obj

    raise ValueError(f"Unsupported extension: {suf}")


def flatten_text(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ")


# ---------------------------------------------------------------------
# Generation (same robust “retry until </answer>” approach you used)
# ---------------------------------------------------------------------

def generate_until_answer(
    llm: LLMWrapper,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int = 2,
    growth: float = 2.0,
    hard_cap: int = 2048,
) -> str:
    cur = max_new_tokens
    out = llm.generate(prompt=prompt, max_new_tokens=cur, temperature=temperature, top_p=top_p)

    for r in range(max_retries):
        if "</answer>" in out.lower():
            return out
        cur = min(int(cur * growth), hard_cap)
        print(f"    [WARN] Missing </answer>. Retrying with max_new_tokens={cur} (retry {r+1}/{max_retries})")
        out = llm.generate(prompt=prompt, max_new_tokens=cur, temperature=temperature, top_p=top_p)

    return out


# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------

def compute_metrics_by_group(rows: List[Dict[str, Any]], group_key: str) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      {
        group_value: {"em": float, "f1": float, "n": int}
      }
    """
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)

    for r in rows:
        g = r.get(group_key, "unknown") or "unknown"
        grouped[g].append((r["pred_answer"], r["gold_answer"]))
        counts[g] += 1

    out: Dict[str, Dict[str, float]] = {}
    for g, pairs in grouped.items():
        em, f1 = compute_em_f1(pairs)
        out[g] = {"em": float(em), "f1": float(f1), "n": int(counts[g])}

    return out


def build_model(mode: str = "dev", lora_path: str | None = None) -> LLMWrapper:
    model_name = get_model_name(mode=mode, lang="en", role="actor")
    return LLMWrapper(model_name=model_name, lora_path=lora_path)




# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

PROMPT_MODES = {
    "standard": STANDARD_PROMPT_TEMPLATE,
    "tiser": TISER_PROMPT_TEMPLATE,
}


def main():
    parser = argparse.ArgumentParser(description="MultiHopRAG Table-3 style eval (Standard vs TISER prompting)")
    parser.add_argument("--mode", type=str, default="dev", choices=["dev", "train"])
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter (optional). If set, loads fine-tuned adapters.")

    parser.add_argument("--data-file", type=str, required=True, help="Path to processed MultiHopRAG TISER-like JSON/JSONL")
    parser.add_argument("--tag", type=str, default="multihoprag")
    parser.add_argument("--max-examples", type=int, default=None)

    parser.add_argument("--temp", type=float, default=GEN_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=GEN_TOP_P)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=2)

    parser.add_argument("--prompt-modes", type=str, default="standard,tiser",
                        help="Comma-separated prompt modes to run: standard,tiser")

    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.prompt_modes.split(",") if m.strip()]
    for m in modes:
        if m not in PROMPT_MODES:
            raise ValueError(f"Unknown prompt mode '{m}'. Available: {list(PROMPT_MODES.keys())}")

    print(f"[INFO] Loading dataset: {data_path}")
    samples = load_multihoprag_tiserlike(data_path)
    if args.max_examples is not None:
        samples = samples[:args.max_examples]
    print(f"[INFO] Loaded {len(samples)} samples")

    print(f"[INFO] Initializing model (mode={args.mode})")
    llm = build_model(mode=args.mode, lora_path=args.lora)


    # Collect summary rows (one per prompt mode × group)
    summary_rows: List[Dict[str, Any]] = []

    for mode in modes:
        print(f"\n==============================")
        print(f"[RUN] Prompt mode: {mode}")
        print(f"==============================")

        template = PROMPT_MODES[mode]
        rows: List[Dict[str, Any]] = []
        preds_gold_all: List[Tuple[str, str]] = []

        for i, ex in enumerate(samples, start=1):
            qid = ex.get("question_id", f"ex_{i}")
            qtype = ex.get("question_type", "unknown") or "unknown"
            question = ex.get("question", "")
            context = ex.get("context", "")
            gold = ex.get("answer", "")

            print(f"  [{i}/{len(samples)}] qid={qid} ({qtype})")

            prompt = template.format(question=question, context=context)

            raw = generate_until_answer(
                llm=llm,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temp,
                top_p=args.top_p,
                max_retries=args.max_retries,
                growth=2.0,
                hard_cap=2048,
            )

            pred = extract_answer(raw)  
            preds_gold_all.append((pred, gold))

            rows.append({
                "idx": i,
                "lora": args.lora if args.lora else "",
                "prompt_mode": mode,
                "question_type": qtype,
                "question_id": qid,
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred,
                "raw_output": flatten_text(raw),
                "has_answer_tag": False if pred == "" else True,
            })

        # Save per-mode logs
        out_csv = RESULTS_DIR / f"multihoprag_{args.tag}_{mode}.csv"
        print(f"[INFO] Saving logs -> {out_csv}")
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "idx", "lora", "prompt_mode", "question_type", "question_id",
                "question", "gold_answer", "pred_answer",
                "raw_output", "has_answer_tag",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        # Overall metrics
        overall_em, overall_f1 = compute_em_f1(preds_gold_all)
        print(f"[RESULTS:{mode}] overall EM={overall_em:.4f} | F1={overall_f1:.4f}")

        # Per-question_type metrics
        by_type = compute_metrics_by_group(rows, group_key="question_type")
        for qt, mtr in sorted(by_type.items()):
            print(f"  - {qt}: EM={mtr['em']:.4f} | F1={mtr['f1']:.4f} | n={mtr['n']}")

        # Add summary rows (overall + each type)
        summary_rows.append({
            "tag": args.tag,
            "lora": args.lora if args.lora else "",
            "prompt_mode": mode,
            "group": "overall",
            "n": len(rows),
            "em": float(overall_em),
            "f1": float(overall_f1),
        })
        for qt, mtr in by_type.items():
            summary_rows.append({
                "tag": args.tag,
                "lora": args.lora if args.lora else "",
                "prompt_mode": mode,
                "group": qt,
                "n": int(mtr["n"]),
                "em": float(mtr["em"]),
                "f1": float(mtr["f1"]),
            })

    # Write summary CSV
    summary_csv = RESULTS_DIR / f"multihoprag_summary_{args.tag}.csv"
    print(f"\n[INFO] Saving summary -> {summary_csv}")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["tag", "lora", "prompt_mode", "group", "n", "em", "f1"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print("[DONE] MultiHopRAG evaluation completed.")


if __name__ == "__main__":
    main()
