# src/config.py

from __future__ import annotations

from pathlib import Path

# === Path di progetto ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_EN_ACTOR = CHECKPOINTS_DIR / "en_actor_lora"
CHECKPOINTS_IT_ACTOR = CHECKPOINTS_DIR / "it_actor_lora"
CHECKPOINTS_CRITIC = CHECKPOINTS_DIR / "critic"

# Directory per i checkpoint durante l'addestramento
CHECKPOINTS_TRAINING_LORA = CHECKPOINTS_DIR / "training_lora"


# === Modelli ===
# Idea:
# - DEV_*: modelli piccoli per sviluppo locale (Mac / CPU o MPS)
# - TRAIN_*: modelli più grandi per Colab / GPU

# Modello piccolo per sviluppo (puoi cambiarlo con quello che riesci a far girare):
DEV_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# Alternative:
#   - da 3.8B parameters microsoft/Phi-3-mini-4k-instruct
#   - da 3B parameters: Qwen/Qwen-3B-Instruct
#   - da 3B parameters:meta-llama/Llama-3.2-3B-Instruct

# Modello “serio” per esperimenti EN su GPU (Colab)
TRAIN_MODEL_NAME_EN = "Qwen/Qwen2.5-7B-Instruct"
# Alternativa: da 7B parameters mistralai/Mistral-7B-v0.1

# Per semplicità, usiamo lo stesso modello anche per IT (multilingual)
TRAIN_MODEL_NAME_IT = TRAIN_MODEL_NAME_EN

# Critic: per ora lo facciamo uguale all'actor off-the-shelf
CRITIC_MODEL_NAME_EN = TRAIN_MODEL_NAME_EN


# === Default generazione ===

GEN_MAX_NEW_TOKENS = 256  # set to 32 for faster testing & sanity checks
GEN_TEMPERATURE = 0.2
GEN_TOP_P = 0.9


def get_model_name(mode: str = "dev", lang: str = "en", role: str = "actor") -> str:
    """
    Restituisce il nome del modello HF a seconda di:
      - mode: "dev" (sviluppo locale) o "train" (esperimenti seri)
      - lang: "en" / "it"
      - role: "actor" / "critic" (per futura estensione multi-agent)

    Uso tipico:
        model_name = get_model_name(mode="dev", lang="en", role="actor")
    """
    mode = mode.lower()
    lang = lang.lower()
    role = role.lower()

    if mode == "dev":
        # in dev usiamo sempre il modello piccolo
        return DEV_MODEL_NAME

    # mode == "train": usiamo i modelli “seri”
    if role == "critic":
        # se mai vorrete un critic diverso, lo cambiate qui
        return CRITIC_MODEL_NAME_EN

    # actor
    if lang == "it":
        return TRAIN_MODEL_NAME_IT
    return TRAIN_MODEL_NAME_EN