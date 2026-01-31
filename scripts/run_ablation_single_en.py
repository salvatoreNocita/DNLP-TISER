"""
CLI Script for TISER Prompt Ablation Study (Single-Prompt Only)

- Same LLM for all variants
- Each ablation variant corresponds to ONE prompt template variable from src.tiser.prompts
- Each template only needs .format(question=..., context=ex.context)
- Computes EM per dataset + macro average
- Saves per-variant CSV logs + one summary CSV

Example:
    python scripts/run_ablation_single_prompt.py \
        --test-file data/processed/TISER_test.json \
        --tag table4_replica \
        --variants standard,only_reasoning,only_timeline,no_reflection,no_timeline,no_reasoning,all_stages
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict

from src.config import (
    RESULTS_DIR,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    get_model_name,
)
from src.models.base_model import LLMWrapper
from src.data.tiser_dataset import load_tiser_file
from src.tiser.metrics import compute_em_f1
from src.tiser.parsing import extract_answer


# ----------------------------------------------------------------------
# IMPORT PROMPT VARIABLES (edit names to match your prompts.py)
# Each is a FULL prompt (single prompt) that contains its own instructions.
# ----------------------------------------------------------------------
from src.tiser.prompts import (
    STANDARD_PROMPT_TEMPLATE,               # standard prompt
    ABLATION_ONLY_REASONING_PROMPT_TEMPLATE,         # only reasoning
    ABLATION_ONLY_TIMELINE_PROMPT_TEMPLATE,          # only timeline construction
    ABLATION_NO_REFLECTION_PROMPT_TEMPLATE,          # ablation: no reflection
    ABLATION_NO_TIMELINE_PROMPT_TEMPLATE,            # ablation: no timeline construction
    ABLATION_NO_REASONING_PROMPT_TEMPLATE,           # ablation: no reasoning
    TISER_PROMPT_TEMPLATE,             # full TISER prompt
)

# Map: variant_name -> template_variable
VARIANT_PROMPTS: Dict[str, str] = {
    "standard": STANDARD_PROMPT_TEMPLATE,
    "only_reasoning": ABLATION_ONLY_REASONING_PROMPT_TEMPLATE,
    "only_timeline": ABLATION_ONLY_TIMELINE_PROMPT_TEMPLATE,
    "no_reflection": ABLATION_NO_REFLECTION_PROMPT_TEMPLATE,
    "no_timeline": ABLATION_NO_TIMELINE_PROMPT_TEMPLATE,
    "no_reasoning": ABLATION_NO_REASONING_PROMPT_TEMPLATE,
    "all_stages": TISER_PROMPT_TEMPLATE,
}

def flatten_text(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ")


# ======================================================================
# GENERATION HELPERS
# ======================================================================

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


# ======================================================================
# METRICS
# ======================================================================

def compute_em_by_dataset(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, float], float]:
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[r["dataset_name"]].append((r["pred_answer"], r["gold_answer"]))

    em_per_ds: Dict[str, float] = {}
    for ds, pairs in grouped.items():
        em, _ = compute_em_f1(pairs)
        em_per_ds[ds] = float(em)

    macro_avg = sum(em_per_ds.values()) / len(em_per_ds) if em_per_ds else 0.0
    return em_per_ds, float(macro_avg)


# ======================================================================
# MODEL
# ======================================================================

def build_model(mode: str = "dev", lora_path: Optional[str] = None) -> LLMWrapper:
    model_name = get_model_name(mode=mode, lang="en", role="actor")
    return LLMWrapper(model_name=model_name)


# ======================================================================
# MAIN
# ======================================================================

DEFAULT_VARIANTS = ["standard", "only_reasoning", "only_timeline", "no_reflection", "no_timeline", "no_reasoning", "all_stages"]

def main():
    parser = argparse.ArgumentParser(description="TISER Ablation Runner (Single Prompt per Variant)")
    parser.add_argument("--mode", type=str, default="dev", choices=["dev", "train"])
    parser.add_argument("--test-file", type=str, required=True, help="Path to JSON/JSONL test file")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--tag", type=str, default="ablation")
    parser.add_argument("--temp", type=float, default=GEN_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=GEN_TOP_P)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))

    args = parser.parse_args()

    test_path = Path(args.test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANT_PROMPTS:
            raise ValueError(f"Unknown variant '{v}'. Available: {list(VARIANT_PROMPTS.keys())}")

    print(f"[INFO] Loading test set: {test_path}")
    examples = load_tiser_file(test_path, max_examples=args.max_examples)
    print(f"[INFO] Loaded {len(examples)} examples")

    print(f"[INFO] Initializing model (mode={args.mode})")
    llm = build_model(mode=args.mode)

    summary_rows: List[Dict[str, Any]] = []

    for variant in variants:
        print(f"\n==============================")
        print(f"[RUN] Variant: {variant}")
        print(f"==============================")

        prompt_template = VARIANT_PROMPTS[variant]
        variant_rows: List[Dict[str, Any]] = []
        preds_gold_all: List[Tuple[str, str]] = []

        for i, ex in enumerate(examples, start=1):
            print(f"  [{i}/{len(examples)}] qid={ex.question_id} ({ex.dataset_name})")

            question = ex.question
            context = ex.context

            template_to_use = prompt_template

            prompt = template_to_use.format(question=question, context=context)

            raw = generate_until_answer(
                llm=llm,
                prompt=prompt,
                max_new_tokens=256,
                temperature=args.temp,
                top_p=args.top_p,
                max_retries=args.max_retries,
                growth=2.0,
                hard_cap=2048,
            )

            pred_answer = extract_answer(raw)
            has_answer_tag = False if pred_answer == "" else True
            gold = ex.answer

            preds_gold_all.append((pred_answer, gold))

            variant_rows.append({
                "idx": i,
                "variant": variant,
                "dataset_name": ex.dataset_name,
                "question_id": ex.question_id,
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred_answer,
                "raw_output": flatten_text(raw),
                "has_answer_tag": has_answer_tag,
            })

        em_all, f1_all = compute_em_f1(preds_gold_all)
        em_per_ds, macro_avg = compute_em_by_dataset(variant_rows)

        print(f"\n[RESULTS:{variant}] overall EM={em_all:.4f} | F1={f1_all:.4f}")
        print(f"[RESULTS:{variant}] macro avg (per-dataset EM avg) = {macro_avg:.4f}")
        for ds, emv in sorted(em_per_ds.items()):
            print(f"  - {ds}: EM={emv:.4f}")

        out_csv = RESULTS_DIR / f"ablation_{args.tag}_{variant}.csv"
        print(f"[INFO] Saving logs -> {out_csv}")
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "idx", "variant", "dataset_name", "question_id",
                "question", "gold_answer", "pred_answer",
                "raw_output", "has_answer_tag",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(variant_rows)

        row = {
            "tag": args.tag,
            "variant": variant,
            "overall_em": float(em_all),
            "overall_f1": float(f1_all),
            "macro_avg_em": float(macro_avg),
        }
        for ds, emv in em_per_ds.items():
            row[f"em__{ds}"] = float(emv)
        summary_rows.append(row)

    summary_csv = RESULTS_DIR / f"ablation_summary_{args.tag}.csv"
    print(f"\n[INFO] Saving summary -> {summary_csv}")

    all_keys = set()
    for r in summary_rows:
        all_keys |= set(r.keys())
    base_keys = ["tag", "variant", "overall_em", "overall_f1", "macro_avg_em"]
    extra_keys = sorted([k for k in all_keys if k not in base_keys])
    fieldnames = base_keys + extra_keys

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print("[DONE] Ablation completed.")


if __name__ == "__main__":
    main()
