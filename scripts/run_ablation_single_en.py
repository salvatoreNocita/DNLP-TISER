"""
CLI Script for TISER Prompt Ablation Study (Single-Prompt Only)
MODIFIED: Vertical Summary Format (Row per Dataset per Variant) + Flattened Raw Output
MODIFIED: Added --lora-path support for Fine-Tuned models

- Same LLM for all variants
- Computes EM & F1 per dataset
- Saves per-variant CSV logs (flattened)
- Saves ONE summary CSV with vertical structure:
  [tag, variant, dataset_name, n, em, f1]

Example:
    python scripts/run_ablation_single_prompt.py \
        --test-file data/processed/TISER_test.json \
        --tag v2_vertical \
        --variants standard,no_reasoning \
        --lora-path /path/to/checkpoints/checkpoint-100
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
# IMPORT PROMPT VARIABLES
# ----------------------------------------------------------------------
from src.tiser.prompts import (
    STANDARD_PROMPT_TEMPLATE,               
    ABLATION_ONLY_REASONING_PROMPT_TEMPLATE,         
    ABLATION_ONLY_TIMELINE_PROMPT_TEMPLATE,          
    ABLATION_NO_REFLECTION_PROMPT_TEMPLATE,          
    ABLATION_NO_TIMELINE_PROMPT_TEMPLATE,            
    ABLATION_NO_REASONING_PROMPT_TEMPLATE,           
    TISER_PROMPT_TEMPLATE,             
)

VARIANT_PROMPTS: Dict[str, str] = {
    "standard": STANDARD_PROMPT_TEMPLATE,
    "only_reasoning": ABLATION_ONLY_REASONING_PROMPT_TEMPLATE,
    "only_timeline": ABLATION_ONLY_TIMELINE_PROMPT_TEMPLATE,
    "no_reflection": ABLATION_NO_REFLECTION_PROMPT_TEMPLATE,
    "no_timeline": ABLATION_NO_TIMELINE_PROMPT_TEMPLATE,
    "no_reasoning": ABLATION_NO_REASONING_PROMPT_TEMPLATE,
    "all_stages": TISER_PROMPT_TEMPLATE,
}

# ======================================================================
# UTILITIES
# ======================================================================

def flatten_text(text: str) -> str:
    """Flatten newlines for single-line CSV logging."""
    if not text: return ""
    return text.replace("\n", " ").replace("\r", " ")

def compute_detailed_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes EM and F1 for each dataset AND an overall aggregate.
    Returns a list of dictionaries ready for the summary CSV.
    """
    # 1. Group by dataset
    grouped = defaultdict(list)
    all_pairs = []
    
    for r in rows:
        pair = (r["pred_answer"], r["gold_answer"])
        grouped[r["dataset_name"]].append(pair)
        all_pairs.append(pair)

    metrics_list = []

    # 2. Calculate per-dataset metrics
    for ds_name, pairs in grouped.items():
        em, f1 = compute_em_f1(pairs)
        metrics_list.append({
            "dataset_name": ds_name,
            "n": len(pairs),
            "em": em,
            "f1": f1
        })
    
    # 3. Calculate Overall metrics (Micro-average essentially)
    if all_pairs:
        ov_em, ov_f1 = compute_em_f1(all_pairs)
        metrics_list.append({
            "dataset_name": "__OVERALL__",
            "n": len(all_pairs),
            "em": ov_em,
            "f1": ov_f1
        })

    # Sort so datasets appear alphabetically
    metrics_list.sort(key=lambda x: x["dataset_name"])
    
    return metrics_list

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
# MODEL
# ======================================================================

def build_model(mode: str = "dev", lora_path: Optional[str] = None) -> LLMWrapper:
    """
    Builds the model wrapper.
    If lora_path is provided, it is passed to the LLMWrapper.
    """
    model_name = get_model_name(mode=mode, lang="en", role="actor")
    
    print(f"[MODEL] Base model: {model_name}")
    if lora_path:
        print(f"[MODEL] Loading LoRA adapter from: {lora_path}")
        # NOTE: Ensure your LLMWrapper accepts 'lora_path' or modify the key below 
        # to match your implementation (e.g., peft_model_id, adapter_path).
        return LLMWrapper(model_name=model_name, lora_path=lora_path)
    else:
        return LLMWrapper(model_name=model_name)

# ======================================================================
# MAIN
# ======================================================================

DEFAULT_VARIANTS = ["standard", "only_reasoning", "only_timeline", "no_reflection", "no_timeline", "no_reasoning", "all_stages"]

def main():
    parser = argparse.ArgumentParser(description="TISER Ablation Runner (Vertical Summary)")
    parser.add_argument("--mode", type=str, default="dev", choices=["dev", "train"])
    parser.add_argument("--test-file", type=str, required=True, help="Path to JSON/JSONL test file")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--tag", type=str, default="ablation")
    parser.add_argument("--temp", type=float, default=GEN_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=GEN_TOP_P)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--lora-path", type=str, default=None, help="Path to fine-tuned LoRA adapter (optional)")

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
    # MODIFIED CALL
    llm = build_model(mode=args.mode, lora_path=args.lora_path)

    # This list will hold rows for the final SUMMARY csv
    global_summary_rows: List[Dict[str, Any]] = []

    for variant in variants:
        print(f"\n==============================")
        print(f"[RUN] Variant: {variant}")
        print(f"==============================")

        prompt_template = VARIANT_PROMPTS[variant]
        variant_rows: List[Dict[str, Any]] = []
        
        for i, ex in enumerate(examples, start=1):
            print(f"  [{i}/{len(examples)}] qid={ex.question_id} ({ex.dataset_name})")

            question = ex.question
            context = ex.context
            prompt = prompt_template.format(question=question, context=context)

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

            # Store log row
            variant_rows.append({
                "idx": i,
                "variant": variant,
                "dataset_name": ex.dataset_name,
                "question_id": ex.question_id,
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred_answer,
                "raw_output": flatten_text(raw), # FLATTENED as requested
                "has_answer_tag": has_answer_tag,
            })

        # --- 1. Save detailed logs for this variant ---
        # If lora is used, maybe reflect that in filename or just trust the tag
        out_csv = RESULTS_DIR / f"ablation_{args.tag}_{variant}.csv"
        print(f"[INFO] Saving detailed logs -> {out_csv}")
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "idx", "variant", "dataset_name", "question_id",
                "question", "gold_answer", "pred_answer",
                "raw_output", "has_answer_tag",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(variant_rows)

        # --- 2. Compute Metrics (Dataset-wise + Overall) ---
        stats_list = compute_detailed_metrics(variant_rows)
        
        # Add these stats to the global summary list
        print(f"[METRICS] Summary for {variant}:")
        for stat in stats_list:
            print(f"  - {stat['dataset_name']:20s} | N={stat['n']:3d} | EM={stat['em']:.4f} | F1={stat['f1']:.4f}")
            
            # Construct row for summary CSV
            global_summary_rows.append({
                "tag": args.tag,
                "variant": variant,
                "dataset_name": stat["dataset_name"],
                "n": stat["n"],
                "em": f"{stat['em']:.4f}",
                "f1": f"{stat['f1']:.4f}"
            })

    # --- 3. Save Global Summary CSV ---
    summary_csv = RESULTS_DIR / f"ablation_summary_{args.tag}.csv"
    print(f"\n[INFO] Saving global vertical summary -> {summary_csv}")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["tag", "variant", "dataset_name", "n", "em", "f1"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(global_summary_rows)

    print("[DONE] Ablation completed.")

if __name__ == "__main__":
    main()