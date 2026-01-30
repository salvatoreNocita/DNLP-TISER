# scripts/run_baseline_single_en.py

import argparse
import csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple
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
from src.tiser.parsing import extract_answer


def generate_until_answer(
    llm,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int = 2,
    growth: float = 2.0,
    hard_cap: int = 2048,
) -> str:
    """
    Genera output e, se manca </answer>, ritenta aumentando max_new_tokens.
    Se fallisce comunque, restituisce l'ultimo output (che sarà incompleto).
    """
    cur = max_new_tokens
    out = llm.generate(prompt=prompt, max_new_tokens=cur, temperature=temperature, top_p=top_p)

    for r in range(max_retries):
        if "</answer>" in out.lower():
            return out
        cur = min(int(cur * growth), hard_cap)
        print(f"[WARN] Missing </answer>. Retrying with max_new_tokens={cur} (retry {r+1}/{max_retries})")
        out = llm.generate(prompt=prompt, max_new_tokens=cur, temperature=temperature, top_p=top_p)

    return out


def compute_per_dataset_metrics(rows_for_csv: List[Dict]) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    rows_for_csv must contain: dataset_name, pred_answer, gold_answer
    Returns:
      per_ds: {dataset_name: {"em": em, "f1": f1, "n": n}}
      macro_avg_em: mean of EM across datasets (unweighted)
    """
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r in rows_for_csv:
        grouped[r["dataset_name"]].append((r["pred_answer"], r["gold_answer"]))

    per_ds: Dict[str, Dict[str, float]] = {}
    for ds, pairs in grouped.items():
        em, f1 = compute_em_f1(pairs)
        per_ds[ds] = {"em": float(em), "f1": float(f1), "n": float(len(pairs))}

    macro_avg_em = 0.0
    if per_ds:
        macro_avg_em = sum(v["em"] for v in per_ds.values()) / len(per_ds)

    return per_ds, float(macro_avg_em)


def build_model(mode: str = "dev", lora_path: Optional[str] = None) -> LLMWrapper:
    """
    Crea un LLMWrapper con il modello giusto:
    - mode = "dev" -> modello piccolo (per sviluppo locale)
    - mode = "train" -> modello grande (per esperimenti seri su GPU)
    - lora_path: se non None, carica gli adapter LoRA (modello fine-tuned)
    """
    model_name = get_model_name(mode=mode, lang="en", role="actor")
    llm = LLMWrapper(
        model_name=model_name,
        lora_path=lora_path,
    )
    return llm


def main():
    parser = argparse.ArgumentParser(
        description="Baseline TISER EN: off-the-shelf o fine-tuned su stesso test set."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dev",
        choices=["dev", "train"],
        help="dev = modello piccolo per sviluppo; train = modello grande per esperimenti seri.",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Path all'adapter LoRA (se fornito, il modello è fine-tuned).",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to the JSON/JSONL test file.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Se specificato, limita il numero di esempi valutati.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag per il nome del file di risultato (es. 'base' o 'ft').",
    )

    # Retry policy knobs
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--growth", type=float, default=2.0)
    parser.add_argument("--hard-cap", type=int, default=2048)

    args = parser.parse_args()

    test_path = Path(args.test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Carico test set da: {test_path}")
    examples = load_tiser_file(test_path, max_examples=args.max_examples)
    print(f"[INFO] Numero di esempi: {len(examples)}")

    print(f"[INFO] Creo modello (mode={args.mode}, lora={args.lora})...")
    llm = build_model(mode=args.mode, lora_path=args.lora)

    preds_gold = []
    rows_for_csv = []

    for i, ex in enumerate(examples, start=1):
        prompt = ex.prompt
        gold = ex.answer

        print(f"[{i}/{len(examples)}] question_id={ex.question_id}")

        output = generate_until_answer(
            llm=llm,
            prompt=prompt,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            max_retries=args.max_retries,
            growth=args.growth,
            hard_cap=args.hard_cap,
        )

        if "</answer>" not in output.lower():
            print(f"[WARN] Still missing </answer> after retries. Marking prediction as empty. qid={ex.question_id}")
            pred_answer = ""   # fail-closed
        else:
            pred_answer = extract_answer(output)

        preds_gold.append((pred_answer, gold))

        rows_for_csv.append(
            {
                "idx": i,
                "dataset_name": ex.dataset_name,
                "question_id": ex.question_id,
                "question": ex.question,
                "gold_answer": gold,
                "pred_answer": pred_answer,
                "raw_output": output,
                "has_answer_tag": ("</answer>" in output.lower()),
            }
        )

    # Overall metrics (micro-ish, across all examples)
    em, f1 = compute_em_f1(preds_gold)
    print(f"\n[RESULTS - OVERALL] EM = {em:.4f}, F1 = {f1:.4f}")

    # Per-dataset metrics + macro avg (unweighted mean across datasets)
    per_ds, macro_avg_em = compute_per_dataset_metrics(rows_for_csv)

    print("\n[RESULTS - PER DATASET]")
    for ds in sorted(per_ds.keys()):
        m = per_ds[ds]
        print(f"  - {ds:20s} | EM={m['em']:.4f} | F1={m['f1']:.4f} | N={int(m['n'])}")

    print(f"\n[RESULTS - MACRO AVG] MacroAvg EM (unweighted) = {macro_avg_em:.4f}")

    tag = args.tag
    if tag is None:
        tag = "ft" if args.lora is not None else "base"

    out_csv = RESULTS_DIR / f"baseline_single_en_{tag}.csv"
    print(f"[INFO] Save results in: {out_csv}")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "dataset_name",
                "question_id",
                "question",
                "gold_answer",
                "pred_answer",
                "raw_output",
                "has_answer_tag",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_for_csv)

    # Optional: per-dataset summary CSV
    summary_csv = RESULTS_DIR / f"baseline_single_en_{tag}_summary.csv"
    print(f"[INFO] Salvo summary per-dataset in: {summary_csv}")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["tag", "dataset_name", "n", "em", "f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ds in sorted(per_ds.keys()):
            m = per_ds[ds]
            writer.writerow(
                {
                    "tag": tag,
                    "dataset_name": ds,
                    "n": int(m["n"]),
                    "em": m["em"],
                    "f1": m["f1"],
                }
            )

        writer.writerow(
            {
                "tag": tag,
                "dataset_name": "__MACRO_AVG__",
                "n": sum(int(per_ds[ds]["n"]) for ds in per_ds),
                "em": macro_avg_em,
                "f1": "",
            }
        )


if __name__ == "__main__":
    main()
