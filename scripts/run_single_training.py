#!/usr/bin/env python3
"""
CLI Script for TISER Single Model Fine-tuning

This script provides a command-line interface for fine-tuning language models
on the TISER dataset using LoRA (Low-Rank Adaptation) for efficient training.

Features:
- Automatic device detection (CUDA, MPS, CPU)
- LoRA-based efficient fine-tuning
- Configurable training hyperparameters
- Support for custom datasets and model checkpoints
- Comprehensive logging and checkpoint saving
- Memory-efficient gradient checkpointing

Examples:
    # Basic training with default settings
    python scripts/run_single_training.py --data src/data/processed/TISER_train_10pct.json --output experiments/qwen_finetuned
    
    # Custom model and training parameters
    python scripts/run_single_training.py --data src/data/processed/TISER_train_20pct.json \\
        --output experiments/qwen_custom \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --epochs 5 \\
        --batch-size 2 \\
        --learning-rate 3e-4
    
    # Advanced LoRA configuration
    python scripts/run_single_training.py \\
        --data src/data/processed/TISER_train_10pct.json \\
        --output experiments/qwen_lora \\
        --lora-r 32 \\
        --lora-alpha 64 \\
        --max-seq-length 4096
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from datasets import Dataset as HFDataset
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tiser_dataset import TiserDataset, load_tiser_file

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True):
    """Configure logging for the script."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def detect_device() -> str:
    """
    Detect the best available device for training.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Device detected: NVIDIA GPU ({gpu_name})")
    elif torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Device detected: Apple Silicon (MPS)")
    else:
        device = 'cpu'
        logger.warning("Device detected: CPU (Training will be very slow!)")
    
    return device


def setup_tokenizer(model_id: str):
    """
    Load and configure tokenizer for the model.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Configured tokenizer
    """
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    return tokenizer


def load_dataset(data_path: Path, tokenizer, max_examples: Optional[int] = None):
    """
    Load and prepare TISER dataset for training.
    
    Args:
        data_path: Path to JSONL dataset file
        tokenizer: Tokenizer to use for the dataset
        max_examples: Optional limit on number of examples
        
    Returns:
        TiserDataset ready for training
    """
    logger.info(f"Loading dataset from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    # Load raw examples
    raw_examples = load_tiser_file(
        data_path,
        max_examples=max_examples,
        skip_invalid=True,
        verbose=True
    )
    
    # Create PyTorch dataset
    train_dataset = TiserDataset(
        raw_examples,
        tokenizer,
        validate_training=True
    )
    
    logger.info(f"Dataset loaded: {len(train_dataset)} examples")
    
    return train_dataset


def setup_data_collator(tokenizer, response_template: str = "<|im_start|>assistant\n"):
    """
    Create data collator for completion-only training. The collator is responsible for:
    Padding sequences: Different text sequences might have different lengths. The data colltor pads these sequences to the same length so that they can be processed in parallel in a batch
    Creating Attention Masks: When padding sequences, the data collator also creates attention masks to inform the model which tokens are real and which are just padding.
    Handling Special Tokens: Some models require special tokens (like start and end tokens) in the input. The data collator can add these as needed.
    
    Args:
        tokenizer: Tokenizer to use
        response_template: Template marking the start of assistant response
        
    Returns:
        DataCollatorForCompletionOnlyLM
    """
    #We use this collator because forces the model to try to learn only the response, not the question. The colletor
    #is indeed responsible to associate -inf labels to the corresponding pad token inserted, but DataCollatorForCompletionOnlyLM
    #is used to force the model to learn only how to minimize the loss on the response, avoiding useless mnemonic work
    #to learn the question. For example:
    #A classic collator would do the following:
    #input_ids: [User] [Prompt] [Assist] [Resp] [PAD]
    #labels: [User] [Prompt] [Assist] [Resp] [-100]
    #DataCollatorForCompletionOnlyLM would do this instead:
    #input_ids: [User] [Prompt] [Assist] [Resp] [PAD]
    #labels: [-100] [-100] [-100] [Resp] [-100]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )
    
    logger.info("Data collator configured for completion-only training")
    
    return collator


def load_model(model_id: str, device: str, use_flash_attention: bool = False):
    """
    Load and configure the language model.
    
    Args:
        model_id: HuggingFace model identifier
        device: Device type ('cuda', 'mps', or 'cpu')
        use_flash_attention: Whether to use Flash Attention 2 (CUDA only)
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model: {model_id} (this may take a while...)")
    
    # Configure model loading parameters
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,  # bfloat16 works well on M1/M2/M3 and modern GPUs
    }
    
    # Flash Attention 2 only works on CUDA
    if device == "cuda" and use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2 for faster training")
        except Exception as e:
            logger.warning(f"Flash Attention 2 not available: {e}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )
    
    # Enable gradient checkpointing to save memory (essential for 7B+ models)
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")
    model.config.use_cache = False
    return model


def create_lora_config(
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05
) -> LoraConfig:
    """
    Create LoRA configuration for efficient fine-tuning.
    
    Args:
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout probability
        
    Returns:
        LoraConfig object
    """
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Target modules for Qwen architecture (attention + feed forward)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    
    logger.info(
        f"LoRA config: r={lora_r}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}"
    )
    
    return peft_config


def create_training_args(
    output_dir: Path,
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_grad_norm: float = 0.3,
    warmup_ratio: float = 0.03,
    logging_steps: int = 5,
    save_strategy: str = "epoch",
    device: str = "cuda",
    max_seq_length: int = 2048
) -> SFTConfig:
    """
    Create training arguments configuration.
    
    Args:
        output_dir: Directory for saving checkpoints
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Steps for gradient accumulation
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm for clipping
        warmup_ratio: Warmup ratio
        logging_steps: Logging frequency
        save_strategy: Strategy for saving checkpoints
        device: Device type
        
    Returns:
        TrainingArguments object
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        fp16=False,  # Disable fp16 (not compatible with MPS)
        bf16=True,   # Use bfloat16 (works on Apple Silicon and modern GPUs)
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",  # Standard PyTorch optimizer (stable on MPS)
        report_to="none",     # Disable WandB/Tensorboard by default
        remove_unused_columns=True,
        logging_first_step=True,
    )
    
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(
        f"Training config: {num_epochs} epochs, "
        f"effective batch size {effective_batch_size} "
        f"({batch_size} × {gradient_accumulation_steps})"
    )
    logger.info(f"Learning rate: {learning_rate}, warmup: {warmup_ratio}")
    
    return training_args


def train_model(
    data_path: Path,
    output_dir: Path,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_length: int = 2048,
    max_examples: Optional[int] = None,
    use_flash_attention: bool = False,
    response_template: str = "<|im_start|>assistant\n"
):
    """
    Main training function.
    
    Args:
        data_path: Path to training data
        output_dir: Directory for saving model
        model_id: HuggingFace model identifier
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        max_seq_length: Maximum sequence length
        max_examples: Optional limit on training examples
        use_flash_attention: Use Flash Attention 2 (CUDA only)
        response_template: Template for completion-only training
    """
    logger.info("="*80)
    logger.info("TISER SINGLE MODEL FINE-TUNING PIPELINE")
    logger.info("="*80)
    
    # Disable tokenizer parallelism to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Detect device
    device = detect_device()
    
    # Load tokenizer
    tokenizer = setup_tokenizer(model_id)
    
    # Load dataset
    train_dataset = load_dataset(data_path, tokenizer, max_examples)
    hf_train_dataset = HFDataset.from_list([item for item in train_dataset])
    
    # Setup data collator
    data_collator = setup_data_collator(tokenizer, response_template)
    
    # Load model
    model = load_model(model_id, device, use_flash_attention)
    
    # Create LoRA config
    peft_config = create_lora_config(lora_r, lora_alpha, lora_dropout)
    
    # Create training arguments
    training_args = create_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        device=device
    )
    
    # Initialize trainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_train_dataset,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        peft_config=peft_config,
        packing=False
    )
    
    # Start training
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80 + "\n")
    
    trainer.train()
    
    # Save model
    logger.info("\nSaving model...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    logger.info("="*80)
    logger.info(f"TRAINING COMPLETE! Model saved to: {output_dir}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune language models on TISER dataset with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  %(prog)s --data src/data/processed/TISER_train_10pct.json \\
           --output experiments/qwen_finetuned

  # Custom training parameters
  %(prog)s --data src/data/processed/TISER_train_20pct.json \\
           --output experiments/qwen_custom \\
           --epochs 5 \\
           --batch-size 2 \\
           --learning-rate 3e-4

  # Advanced LoRA configuration
  %(prog)s --data src/data/processed/TISER_train_10pct.json \\
           --output experiments/qwen_lora \\
           --lora-r 32 \\
           --lora-alpha 64 \\
           --max-seq-length 4096
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data', '-d',
        type=Path,
        required=True,
        help='Path to training data (JSONL format)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output directory for saving model checkpoints'
    )
    
    # Model configuration
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='HuggingFace model identifier (default: Qwen/Qwen2.5-7B-Instruct)'
    )
    parser.add_argument(
        '--response-template',
        type=str,
        default='<|im_start|>assistant\n',
        help='Response template for completion-only training (default: <|im_start|>assistant\\n)'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='Per-device batch size (default: 1)'
    )
    parser.add_argument(
        '--gradient-accumulation-steps', '-g',
        type=int,
        default=8,
        help='Gradient accumulation steps (default: 8)'
    )
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=2e-4,
        help='Learning rate (default: 2e-4)'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=2048,
        help='Maximum sequence length (default: 2048)'
    )
    
    # LoRA parameters
    parser.add_argument(
        '--lora-r',
        type=int,
        default=16,
        help='LoRA rank (default: 16)'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha parameter (default: 32)'
    )
    parser.add_argument(
        '--lora-dropout',
        type=float,
        default=0.05,
        help='LoRA dropout (default: 0.05)'
    )
    
    # Advanced options
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Limit number of training examples (default: all)'
    )
    parser.add_argument(
        '--flash-attention',
        action='store_true',
        help='Use Flash Attention 2 (CUDA only)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=not args.quiet)
    
    # Run training
    try:
        train_model(
            data_path=args.data,
            output_dir=args.output,
            model_id=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            max_seq_length=args.max_seq_length,
            max_examples=args.max_examples,
            use_flash_attention=args.flash_attention,
            response_template=args.response_template
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

