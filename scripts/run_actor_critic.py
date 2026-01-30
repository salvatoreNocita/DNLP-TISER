"""
CLI Script for TISER Actor-Critic Inference Pipeline

This script executes the multi-stage reasoning pipeline (Actor -> Critic -> Solver)
for the TISER dataset, supporting both base models and fine-tuned LoRA adapters.

Key Features:
- Multi-Stage Inference: Implements the full Actor (Reasoning), Critic (Reflection),
  and Solver (Adjustment) loop.
- Dynamic Prompting: Automatically switches between instruction-heavy prompts for
  base models and minimal inputs for fine-tuned (LoRA) models.
- Robust Generation: Handles XML tag validation and retries for malformed outputs.
- Comprehensive Logging: Exports detailed CSV results including raw outputs from
  all stages for error analysis.

Examples:
    # Run standard inference with a base model (Zero-Shot/Few-Shot)
    python scripts/run_pipeline.py --test-file data/processed/TISER_test.json --tag base_run
    
    # Run inference with a Fine-Tuned LoRA adapter (activates minimal prompts)
    python scripts/run_pipeline.py --test-file data/processed/TISER_test.json --lora checkpoints/tiser_lora_v1 --tag ft_run
    
    # Quick debug run on first 5 examples
    python scripts/run_pipeline.py --test-file data/processed/TISER_test.json --max-examples 5 --tag debug_quick
    
    # Run with higher temperature for creative critique and increased robustness
    python scripts/run_pipeline.py --test-file data/processed/TISER_test.json --temp 0.7 --max-retries 5
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import defaultdict
from typing import List


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
    ACTOR_FINETUNED_TEMPLATE,
    CRITIC_PROMPT_TEMPLATE,
    FINAL_SOLVER_PROMPT_TEMPLATE
)

def compute_metrics_by_dataset(rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    rows must contain: dataset_name, pred_answer, gold_answer
    Returns:
      - per_dataset: {dataset_name: {"em": em, "f1": f1, "n": n}}
      - macro_avg_em: mean of per-dataset EM (unweighted)
    """
    grouped = defaultdict(list)  # dataset_name -> list[(pred, gold)]
    for r in rows:
        grouped[r["dataset_name"]].append((r["pred_answer"], r["gold_answer"]))

    per_dataset = {}
    for ds, pairs in grouped.items():
        em, f1 = compute_em_f1(pairs)
        per_dataset[ds] = {"em": float(em), "f1": float(f1), "n": len(pairs)}

    macro_avg_em = 0.0
    if per_dataset:
        macro_avg_em = sum(v["em"] for v in per_dataset.values()) / len(per_dataset)

    return per_dataset, float(macro_avg_em)


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
    If it still fails after retries, returns the last output (which might be incomplete).
    """
    cur = max_new_tokens
    # First attempt
    out = llm.generate(prompt=prompt, max_new_tokens=cur, temperature=temperature, top_p=top_p)

    for r in range(max_retries):
        # Check for closing tag (case-insensitive)
        if "</answer>" in out.lower():
            return out
        
        # Increase token limit
        cur = min(int(cur * growth), hard_cap)
        print(f"    [WARN] Missing </answer>. Retrying with max_new_tokens={cur} (retry {r+1}/{max_retries})")
        
        # Retry generation with increased tokens
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
    Stage 3 uses `generate_until_answer` to ensure the model has enough tokens to finish.
    """

    # === STAGE 1: THE ACTOR (Original Prompt) ===
    # Using standard generation here as we parse roughly
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

    # === STAGE 2: THE CRITIC (Reflection) ===
    critic_prompt = CRITIC_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        draft_reasoning=draft_reasoning,
        draft_timeline=draft_timeline
    )

    # Reflection is usually short, 512 is plenty
    raw_stage_2 = llm.generate(
        prompt=critic_prompt,
        max_new_tokens=512, 
        temperature=temperature, 
        top_p=top_p
    )

    critic_reflection = extract_section(raw_stage_2, "reflection")
    if not critic_reflection: critic_reflection = raw_stage_2 

    # === STAGE 3: THE FINAL SOLVER (With Adaptive Retry) ===
    final_prompt = FINAL_SOLVER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        draft_reasoning=draft_reasoning,
        draft_timeline=draft_timeline,
        critic_reflection=critic_reflection
    )
    
    # Use the robust generation function specifically for the final answer
    raw_stage_3 = generate_until_answer(
        llm=llm,
        prompt=final_prompt,
        max_new_tokens=256, # Start small
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
        "draft_reasoning": draft_reasoning,
        "draft_timeline": draft_timeline,
        "critic_reflection": critic_reflection
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

    preds_gold = []
    csv_rows = []

    for i, ex in enumerate(examples, start=1):
        print(f"\n--- [{i}/{len(examples)}] Processing qid={ex.question_id} ---")

        # --- SELEZIONE PROMPT DINAMICA ---
        if args.lora:
            
            # Costruiamo il prompt corto
            actor_prompt = ACTOR_FINETUNED_TEMPLATE.format(
                question=ex.question,
                context=ex.context
            )
            
        else:
            actor_prompt = ex.prompt

        # Esegui la pipeline
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
        
        preds_gold.append((pred_answer, gold))

        combined_raw_output = (
            f"--- [STAGE 1: GENERATOR] ---\n{result['stage1_raw']}\n\n"
            f"--- [STAGE 2: CRITIC] ---\n{result['stage2_raw']}\n\n"
            f"--- [STAGE 3: SOLVER] ---\n{result['stage3_raw']}"
        )

        has_answer_tag = "<answer>" in result["stage3_raw"].lower()

        csv_rows.append({
            "idx": i,
            "dataset_name": ex.dataset_name,
            "question_id": ex.question_id,
            "question": ex.question,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "raw_output": combined_raw_output,
            "has_answer_tag": has_answer_tag
        })

    em, f1 = compute_em_f1(preds_gold)
    print(f"\n[RESULTS] EM = {em:.4f}, F1 = {f1:.4f}")

        # ---- Per-dataset metrics + Macro Avg (like Table 4) ----
    per_ds, macro_avg_em = compute_metrics_by_dataset(csv_rows)

    print("\n[PER-DATASET RESULTS]")
    for ds in sorted(per_ds.keys()):
        m = per_ds[ds]
        print(f"  - {ds:20s} | EM={m['em']:.4f} | F1={m['f1']:.4f} | N={m['n']}")

    print(f"\n[MACRO AVG] MacroAvg EM (unweighted) = {macro_avg_em:.4f}")

    tag = args.tag if args.tag else ("ft_critic_loop" if args.lora else "base_critic_loop")
    out_csv = RESULTS_DIR / f"actor_critic_results_{tag}.csv"
    
    print(f"[INFO] Saving results to: {out_csv}")
    
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "idx",
            "dataset_name",
            "question_id",
            "question",
            "gold_answer",
            "pred_answer",
            "raw_output",
            "has_answer_tag"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    summary_csv = RESULTS_DIR / f"actor_critic_summary_{tag}.csv"
    print(f"[INFO] Saving per-dataset summary to: {summary_csv}")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["tag", "dataset_name", "n", "em", "f1"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ds in sorted(per_ds.keys()):
            m = per_ds[ds]
            w.writerow({
                "tag": tag,
                "dataset_name": ds,
                "n": m["n"],
                "em": m["em"],
                "f1": m["f1"],
            })
        # add macro row
        w.writerow({
            "tag": tag,
            "dataset_name": "__MACRO_AVG__",
            "n": sum(m["n"] for m in per_ds.values()),
            "em": macro_avg_em,
            "f1": "",  # macro F1 isn't defined the same way; keep blank
        })


if __name__ == "__main__":
    main()