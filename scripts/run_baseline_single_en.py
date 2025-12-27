# scripts/run_baseline_single_en.py

import argparse
import csv
from pathlib import Path
from typing import Optional

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
from src.tiser.parsing import extract_answer
from src.tiser.metrics import compute_em_f1


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
        default=str(PROCESSED_DIR / "tiser_test_subset_en.json"),
        help="Path al file JSON di test (subset EN).",
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

    args = parser.parse_args()

    test_path = Path(args.test_file)
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

        output = llm.generate(
            prompt=prompt,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
        )

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
            }
        )

    em, f1 = compute_em_f1(preds_gold)
    print(f"[RESULTS] EM = {em:.4f}, F1 = {f1:.4f}")

    tag = args.tag
    if tag is None:
        tag = "ft" if args.lora is not None else "base"

    out_csv = RESULTS_DIR / f"baseline_single_en_{tag}.csv"
    print(f"[INFO] Salvo risultati in: {out_csv}")

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
            ],
        )
        writer.writeheader()
        writer.writerows(rows_for_csv)


if __name__ == "__main__":
    main()