"""
TISER Dataset Loader and PyTorch Dataset Classes

This module provides utilities for loading and handling the TISER
(Temporal Information Semantic Extraction and Reasoning) dataset in various formats.

Key Features:
- JSONL file format support with robust parsing
- Dataclass-based example representation
- Filtering by dataset type
- PyTorch Dataset integration
- Automatic chat template formatting for LLMs
- Comprehensive error handling and validation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Any

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TiserExample:
    """
    Represents a single example from the TISER dataset.
    
    This dataclass encapsulates all the information for one temporal reasoning
    question-answer pair, including metadata, prompt, and optional output.
    
    Attributes:
        dataset_name: Source dataset identifier (e.g., 'tgqa_train', 'timeqa_easy_test')
        question_id: Unique identifier for the question (may not be globally unique)
        question: The natural language question text
        answer: Ground truth answer
        prompt: Full prompt template for the model including instructions
        output: Model's expected output (reasoning + answer). None for test/inference.
        
    Note:
        The output field is only present in training data. Test sets do not include
        this field and it will be None during inference.
    """
    dataset_name: str
    question_id: str
    question: str
    context: str
    answer: str
    prompt: str
    output: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        # Ensure question_id is always a string
        self.question_id = str(self.question_id)
        
        # Validate required fields are not empty
        if not self.dataset_name:
            logger.warning(f"TiserExample created with empty dataset_name (qid: {self.question_id})")
        if not self.question:
            logger.warning(f"TiserExample created with empty question (qid: {self.question_id})")
        if not self.prompt:
            logger.warning(f"TiserExample created with empty prompt (qid: {self.question_id})")
    
    def is_training_example(self) -> bool:
        """
        Check if this example can be used for training.
        
        Returns:
            True if output is available (training), False otherwise (inference)
        """
        return self.output is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the example back to a dictionary format.
        
        Returns:
            Dictionary representation matching the original JSONL format
        """
        result = {
            'dataset_name': self.dataset_name,
            'question_id': self.question_id,
            'question': self.question,
            'context': self.context,
            'answer': self.answer,
            'prompt': self.prompt,
        }
        
        if self.output is not None:
            result['output'] = self.output
        
        return result
    
    def __repr__(self) -> str:
        """Pretty string representation for debugging."""
        output_status = "with output" if self.output else "no output"
        return (
            f"TiserExample(dataset={self.dataset_name}, "
            f"qid={self.question_id[:20]}..., {output_status})"
        )


class TiserFileLoader:
    """
    Handles loading and parsing of TISER dataset files.
    
    This class provides robust loading of JSONL format files with error handling,
    validation, and optional filtering capabilities.
    """
    
    @staticmethod
    def load_file(
        path: Path | str,
        max_examples: Optional[int] = None,
        dataset_filter: Optional[Iterable[str]] = None,
        skip_invalid: bool = True,
        verbose: bool = True
    ) -> List[TiserExample]:
        """
        Load TISER examples from a JSONL file.
        
        Args:
            path: Path to the JSONL file
            max_examples: Maximum number of examples to load (None for all)
            dataset_filter: Iterable of dataset names to include (None for all)
            skip_invalid: If True, skip invalid lines; if False, raise errors
            verbose: If True, log detailed loading information
            
        Returns:
            List of TiserExample objects
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is invalid and skip_invalid is False
            
        Example:
            >>> loader = TiserFileLoader()
            >>> examples = loader.load_file('data/TISER_train.json', max_examples=1000)
            >>> print(f"Loaded {len(examples)} examples")
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"TISER file not found: {path}")
        
        if verbose:
            logger.info(f"Loading TISER data from {path}")
        
        # Parse dataset filter
        dataset_filter_set = set(dataset_filter) if dataset_filter is not None else None
        
        # Load raw data from JSONL
        raw_data = TiserFileLoader._load_jsonl(path, skip_invalid, verbose)
        
        # Convert to TiserExample objects with filtering
        examples = TiserFileLoader._convert_to_examples(
            raw_data,
            dataset_filter_set,
            max_examples,
            verbose
        )
        
        if verbose:
            logger.info(f"Successfully loaded {len(examples)} examples")
            TiserFileLoader._log_dataset_distribution(examples)
        
        return examples


    @staticmethod
    def _load_jsonl(
        path: Path,
        skip_invalid: bool,
        verbose: bool
    ) -> List[Dict[str, Any]]:
        """
        Load JSONL (one JSON object per line) OR a JSON list (single JSON array).
        Auto-detects by looking at the first non-whitespace character.
        """
        # Peek first non-whitespace char
        with path.open("r", encoding="utf-8") as f:
            start = f.read(4096)

        first_non_ws = None
        for ch in start:
            if not ch.isspace():
                first_non_ws = ch
                break

        # Case A: JSON list
        if first_non_ws == "[":
            try:
                obj = json.loads(start + path.open("r", encoding="utf-8").read()[4096:])
            except Exception:
                # fallback: simpler, just json.load
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)

            if not isinstance(obj, list):
                raise ValueError(f"{path} looks like JSON list but is not a list at top-level.")
            # ensure dicts
            out = []
            for i, item in enumerate(obj):
                if isinstance(item, dict):
                    out.append(item)
                elif not skip_invalid:
                    raise ValueError(f"Invalid item at index {i} in JSON list: expected dict, got {type(item)}")
            if verbose:
                logger.info(f"Loaded {len(out)} examples from JSON list file: {path.name}")
            return out

        # Case B: JSONL (current behavior)
        data: List[Dict[str, Any]] = []
        invalid_count = 0

        with path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    invalid_count += 1
                    if skip_invalid:
                        if verbose:
                            logger.warning(f"Skipping invalid line {line_num} in {path.name}: {e}")
                    else:
                        raise ValueError(f"Invalid JSON on line {line_num} in {path}: {e}") from e

        if invalid_count > 0 and verbose:
            logger.warning(f"Skipped {invalid_count} invalid lines in {path.name}")

        return data

    @staticmethod
    def _convert_to_examples(
        raw_data: List[Dict[str, Any]],
        dataset_filter_set: Optional[set],
        max_examples: Optional[int],
        verbose: bool
    ) -> List[TiserExample]:
        """
        Convert raw dictionaries to TiserExample objects with filtering.
        
        Args:
            raw_data: List of raw JSON objects
            dataset_filter_set: Set of dataset names to include (None for all)
            max_examples: Maximum number of examples to return
            verbose: Whether to log filtering information
            
        Returns:
            List of TiserExample objects
        """
        examples = []
        filtered_count = 0
        
        for raw in raw_data:
            # Create example
            example = TiserFileLoader._parse_example(raw)
            
            # Apply dataset filter
            if dataset_filter_set is not None:
                if example.dataset_name not in dataset_filter_set:
                    filtered_count += 1
                    continue
            
            examples.append(example)
            
            # Check max examples limit
            if max_examples is not None and len(examples) >= max_examples:
                break
        
        if filtered_count > 0 and verbose:
            logger.info(f"Filtered out {filtered_count} examples not matching dataset filter")
        
        return examples
    
    @staticmethod
    def _parse_example(raw: Dict[str, Any]) -> TiserExample:
        """
        Parse a raw dictionary into a TiserExample object.
        
        Args:
            raw: Raw JSON object from JSONL file
            
        Returns:
            TiserExample object
        """
        prompt = raw.get('prompt', '')
        context_text = raw.get('context', '') # First, check if it exists explicitly
        
        # --- LOGIC EXTRACTION CONTEXT ---
        if not context_text and prompt:
            # Marker identified in the prompt structure
            marker = "Temporal context:"
            if marker in prompt:
                # Extract everything after the marker
                try:
                    context_text = prompt.split(marker)[-1].strip()
                except Exception:
                    # In case of splitting errors (rare), leave empty
                    pass

        return TiserExample(
            dataset_name=raw.get('dataset_name', ''),
            question_id=str(raw.get('question_id', '')),
            question=raw.get('question', ''),
            context=context_text,
            answer=raw.get('answer', ''),
            prompt=raw.get('prompt', ''),
            output=raw.get('output'),  # May be None for test data
        )
    
    @staticmethod
    def _log_dataset_distribution(examples: List[TiserExample]):
        """Log the distribution of examples across different datasets."""
        from collections import Counter
        
        dataset_counts = Counter(ex.dataset_name for ex in examples)
        
        logger.info("\nDataset Distribution:")
        logger.info("-" * 60)
        for dataset_name, count in sorted(dataset_counts.items()):
            percentage = (count / len(examples)) * 100
            logger.info(f"{dataset_name:<40} {count:>6} ({percentage:>5.1f}%)")
        logger.info("-" * 60)


class TiserDataset(Dataset):
    """
    PyTorch Dataset wrapper for TISER examples.
    
    This dataset class handles:
    - Conversion of TiserExample objects to tokenized inputs
    - Chat template formatting for instruction-tuned models
    - Validation of training examples (ensuring output is present)
    - Integration with PyTorch DataLoader
    
    The dataset returns formatted text suitable for autoregressive language models,
    with proper chat template formatting applied.
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
        >>> examples = load_tiser_file('data/TISER_train_10pct.json')
        >>> dataset = TiserDataset(examples, tokenizer)
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(sample['text'][:100])
    """
    
    def __init__(
        self,
        examples: List[TiserExample],
        tokenizer,
        validate_training: bool = True
    ):
        """
        Initialize the PyTorch Dataset.
        
        Args:
            examples: List of TiserExample objects
            tokenizer: HuggingFace tokenizer with chat template support
            validate_training: If True, validate that all examples have output field
            
        Raises:
            ValueError: If validate_training is True and any example lacks output
        """
        self.examples = examples
        self.tokenizer = tokenizer
        
        # Validate examples if required
        if validate_training:
            self._validate_examples()
        
        logger.info(f"Initialized TiserDataset with {len(examples)} examples")
    
    def _validate_examples(self):
        """
        Validate that all examples are suitable for training.
        
        Raises:
            ValueError: If any example lacks the output field
        """
        invalid_examples = [
            ex for ex in self.examples
            if not ex.is_training_example()
        ]
        
        if invalid_examples:
            error_msg = (
                f"Found {len(invalid_examples)} examples without output field. "
                f"Cannot use for training. First invalid: {invalid_examples[0].question_id}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Get a formatted training example.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Dictionary with 'text' key containing the formatted chat template
            
        Raises:
            ValueError: If the example lacks output field
        """
        example = self.examples[idx]
        
        # Safety check for training examples
        if not example.is_training_example():
            raise ValueError(
                f"Example {example.question_id} lacks output field. "
                f"Cannot be used for training."
            )
        
        # Build chat messages in the format expected by chat templates
        messages = [
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": example.output}
        ]
        
        # Apply the tokenizer's chat template without tokenizing
        # This produces a formatted string ready for the model
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Use this alternative formatting if the tokenizer lacks chat template support
        # and eliminate the above block.
        """if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            else:
                formatted_text = f"{example.prompt}\n{example.output}"
                """
        
        return {"text": formatted_text}
    
    def get_example_metadata(self, idx: int) -> Dict[str, str]:
        """
        Get metadata for an example without formatting.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with metadata fields (dataset_name, question_id, etc.)
        """
        example = self.examples[idx]
        return {
            'dataset_name': example.dataset_name,
            'question_id': example.question_id,
            'question': example.question,
            'context': example.context,
            'answer': example.answer,
        }


# Convenience function that maintains backward compatibility
def load_tiser_file(
    path: Path | str,
    max_examples: Optional[int] = None,
    dataset_filter: Optional[Iterable[str]] = None,
    skip_invalid: bool = True,
    verbose: bool = True
) -> List[TiserExample]:
    """
    Load TISER examples from a JSONL file.
    
    This is a convenience function that wraps TiserFileLoader.load_file()
    for backward compatibility and ease of use.
    
    Args:
        path: Path to the JSONL file
        max_examples: Maximum number of examples to load (None for all)
        dataset_filter: Iterable of dataset names to include (None for all)
        skip_invalid: If True, skip invalid lines; if False, raise errors
        verbose: If True, log detailed loading information
        
    Returns:
        List of TiserExample objects
        
    Example:
        >>> examples = load_tiser_file('data/TISER_train.json', max_examples=100)
        >>> print(f"Loaded {len(examples)} examples")
        >>> 
        >>> # Filter specific datasets
        >>> examples = load_tiser_file(
        ...     'data/TISER_test.json',
        ...     dataset_filter=['tgqa_test', 'timeqa_easy_test']
        ... )
    """
    loader = TiserFileLoader()
    return loader.load_file(
        path=path,
        max_examples=max_examples,
        dataset_filter=dataset_filter,
        skip_invalid=skip_invalid,
        verbose=verbose
    )
