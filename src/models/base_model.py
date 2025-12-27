# src/models/base_model.py

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional


class LLMWrapper:
    """
    Wrapper semplice per un modello causale HF + eventuale adapter LoRA.

    Uso tipico:
        llm = LLMWrapper(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            lora_path=None,  # oppure "checkpoints/en_actor_lora"
        )
        output = llm.generate("My prompt...", max_new_tokens=256)
    """

    def __init__(
        self,
        model_name: str,
        lora_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_name = model_name
        self.lora_path = lora_path

        # === Selezione device: cuda > mps > cpu ===
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # === Selezione dtype in base al device ===
        if dtype is None:
            if self.device in ("cuda", "mps"):
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        # === Tokenizer ===
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # === Modello base ===
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,   # usiamo `dtype` al posto di `torch_dtype`
        )

        # === Adapter LoRA opzionale ===
        if self.lora_path is not None:
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        else:
            self.model = base_model

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: Optional[bool] = None,
    ) -> str:
        """
        Genera testo a partire da un prompt.

        Restituisce l'intero testo generato (prompt incluso).
        Il parsing di <answer> lo facciamo a livello di pipeline TISER.
        """
        if do_sample is None:
            do_sample = temperature > 0.0

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        full_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return full_text

    @torch.inference_mode()
    def generate_answer_only(
        self,
        prompt: str,
        extractor_fn,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        """
        Variante comoda: genera output completo e poi applica una funzione
        che estrae la parte desiderata (es. il contenuto di <answer>).
        """
        full_text = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        answer = extractor_fn(full_text)
        return answer