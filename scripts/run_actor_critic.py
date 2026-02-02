"""
CLI Script for TISER Actor-Critic Inference Pipeline
MODIFIED: Uniformed Format (Vertical Summary + Flattened Raw Output)

- Supports Actor -> Critic -> Solver loop
- Flattened raw outputs for consistent CSV logging
- Vertical summary format (Row per Dataset, plus __OVERALL__)

Examples:
    python scripts/run_pipeline.py --test-file data/processed/TISER_test.json --tag base_run
    python scripts/run_pipeline.py --test-file data/processed/TISER_test.json --lora checkpoints/tiser_lora_v1 --tag ft_run
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from collections import defaultdict

from src.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    get_model_name,
)
from src.models.base_model import LLMWrapper
from src.data.tiser_dataset import load_tiser_file
from src.tiser.metrics import compute_em_f1
from src.tiser.parsing import extract_answer, extract_section
from src.tiser.prompts import (
    TISER_PROMPT_TEMPLATE,
    CRITIC_PROMPT_TEMPLATE,
    FINAL_SOLVER_PROMPT_TEMPLATE
)

# ==============================================================================
# UTILITIES
# ==============================================================================

def flatten_text(text: str) -> str:
    """Flatten newlines for single-line CSV logging."""
    if not text: return ""
    return text.replace("\n", " ").replace("\r", " ")

def compute_detailed_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes EM and F1 for each dataset AND an overall aggregate.
    Returns a list of dictionaries ready for the summary CSV.
    Matches the format of the ablation script.
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
    
    # 3. Calculate Overall metrics (Micro-average)
    if all_pairs:
        ov_em, ov_f1 = compute_em_f1(all_pairs)
        metrics_list.append({
            "dataset_name": "__OVERALL__",
            "n": len(all_pairs),
            "em": ov_em,
            "f1": ov_f1
        })

    # Sort alphabetically
    metrics_list.sort(key=lambda x: x["dataset_name"])
    
    return metrics_list

# ==============================================================================
# HELPER GENERATION FUNCTION
# ==============================================================================

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
    """
    Generates output and, if </answer> is missing, retries by increasing max_new_tokens.
    """
    cur = max_new_tokens
    out = llm.generate(prompt=prompt, max_new_tokens=cur, temperature=temperature, top_p=top_p)

    for r in range(max_retries):
        if "</answer>" in out.lower():
            return out
        
        cur = min(int(cur * growth), hard_cap)
        print(f"    [WARN] Missing </answer>. Retrying with max_new_tokens={cur} (retry {r+1}/{max_retries})")
        out = llm.generate(prompt=prompt, max_new_tokens=cur, temperature=temperature, top_p=top_p)

    return out

# ==============================================================================
# CORE LOGIC
# ==============================================================================

def generate_with_actor_critic_loop(
    llm: LLMWrapper,
    original_prompt: str,
    question: str,
    context: str,
    temperature: float,
    top_p: float,
    max_retries: int = 2, 
) -> Dict[str, str]:
    """
    Executes the 3-stage pipeline.
    """

    # === STAGE 1: THE ACTOR ===
    raw_stage_1 = llm.generate(
        prompt=original_prompt,
        max_new_tokens=1024, 
        temperature=temperature, 
        top_p=top_p
    )

    draft_reasoning = extract_section(raw_stage_1, "reasoning")
    draft_timeline = extract_section(raw_stage_1, "timeline")

    if not draft_reasoning: draft_reasoning = raw_stage_1
    if not draft_timeline: draft_timeline = "Timeline tag missing in draft."

    # === STAGE 2: THE CRITIC ===
    critic_prompt = CRITIC_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        draft_reasoning=draft_reasoning,
        draft_timeline=draft_timeline
    )

    raw_stage_2 = llm.generate(
        prompt=critic_prompt,
        max_new_tokens=512, 
        temperature=temperature, 
        top_p=top_p
    )

    critic_reflection = extract_section(raw_stage_2, "reflection")
    if not critic_reflection: critic_reflection = raw_stage_2 

    # === STAGE 3: THE FINAL SOLVER ===
    final_prompt = FINAL_SOLVER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        draft_reasoning=draft_reasoning,
        draft_timeline=draft_timeline,
        critic_reflection=critic_reflection
    )
    
    raw_stage_3 = generate_until_answer(
        llm=llm,
        prompt=final_prompt,
        max_new_tokens=256,
        temperature=temperature,
        top_p=top_p,
        max_retries=max_retries,
        growth=2.0,
        hard_cap=2048
    )

    final_answer_text = extract_answer(raw_stage_3)
    adjustments_text = extract_section(raw_stage_3, "adjustments")

    return {
        "final_answer": final_answer_text,
        "adjustments": adjustments_text,
        "stage1_raw": raw_stage_1,
        "stage2_raw": raw_stage_2,
        "stage3_raw": raw_stage_3,
    }


def build_model(mode: str = "dev", lora_path: Optional[str] = None) -> LLMWrapper:
    model_name = get_model_name(mode=mode, lang="en", role="actor")
    llm = LLMWrapper(
        model_name=model_name,
        lora_path=lora_path,
    )
    return llm


def main():
    parser = argparse.ArgumentParser(description="TISER Multi-Stage Inference (Actor-Critic Loop)")
    parser.add_argument("--mode", type=str, default="dev", choices=["dev", "train"])
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--test-file", type=str, required=True, help="Path to JSON/JSONL test file")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--temp", type=float, default=GEN_TEMPERATURE)
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries for Stage 3 token expansion")

    args = parser.parse_args()

    test_path = Path(args.test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading test set: {test_path}")
    examples = load_tiser_file(test_path, max_examples=args.max_examples)
    
    print(f"[INFO] Initializing Unified Model (Actor+Critic)...")
    llm = build_model(mode=args.mode, lora_path=args.lora)

    csv_rows = []
    
    # Determine default tag if not provided
    run_tag = args.tag if args.tag else ("ft_critic_loop" if args.lora else "base_critic_loop")

    for i, ex in enumerate(examples, start=1):
        print(f"\n--- [{i}/{len(examples)}] Processing qid={ex.question_id} ---")

        actor_prompt = TISER_PROMPT_TEMPLATE.format(
                question=ex.question,
                context=ex.context
            )

        # Execute Pipeline
        result = generate_with_actor_critic_loop(
            llm=llm,
            original_prompt=actor_prompt, 
            question=ex.question,
            context=ex.context,
            temperature=args.temp,
            top_p=GEN_TOP_P,
            max_retries=args.max_retries
        )
        
        gold = ex.answer
        pred_answer = result["final_answer"]
        
        print(f"  Gold: {gold} | Pred: {pred_answer}")

        # Combine Raw outputs and Flatten
        combined_raw_output = (
            f"[STAGE 1: ACTOR] {result['stage1_raw']} "
            f"[STAGE 2: CRITIC] {result['stage2_raw']} "
            f"[STAGE 3: SOLVER] {result['stage3_raw']}"
        )
        
        has_answer_tag = "<answer>" in result["stage3_raw"].lower()

        csv_rows.append({
            "idx": i,
            "dataset_name": ex.dataset_name,
            "question_id": ex.question_id,
            "question": ex.question,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "raw_output": flatten_text(combined_raw_output), # FLATTENED HERE
            "has_answer_tag": has_answer_tag
        })

    # --- Metrics Computation (Aligned with Ablation Script) ---
    stats_list = compute_detailed_metrics(csv_rows)

    print("\n[RESULTS SUMMARY]")
    for stat in stats_list:
        print(f"  - {stat['dataset_name']:20s} | N={stat['n']:3d} | EM={stat['em']:.4f} | F1={stat['f1']:.4f}")

    # --- Save Detailed Logs (Flattened) ---
    out_csv = RESULTS_DIR / f"actor_critic_results_{run_tag}.csv"
    print(f"\n[INFO] Saving logs to: {out_csv}")
    
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "idx", "dataset_name", "question_id",
            "question", "gold_answer", "pred_answer",
            "raw_output", "has_answer_tag"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # --- Save Summary (Vertical Format) ---
    summary_csv = RESULTS_DIR / f"actor_critic_summary_{run_tag}.csv"
    print(f"[INFO] Saving per-dataset summary to: {summary_csv}")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        # Matches the ablation script structure (minus 'variant', using 'tag' instead)
        fieldnames = ["tag", "dataset_name", "n", "em", "f1"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        
        for stat in stats_list:
            w.writerow({
                "tag": run_tag,
                "dataset_name": stat["dataset_name"],
                "n": stat["n"],
                "em": f"{stat['em']:.4f}",
                "f1": f"{stat['f1']:.4f}",
            })

if __name__ == "__main__":
    main()