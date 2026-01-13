# src/models/base_model.py

from __future__ import annotations

from typing import Optional, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMWrapper:
    """
    Wrapper semplice per un modello HF causal LM, con supporto:
    - modelli chat/instruct via tokenizer.chat_template (apply_chat_template)
    - LoRA opzionale (PEFT) se lora_path è fornito
    - device selection: cuda > mps > cpu
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

        # === Selezione dtype ===
        # Nota: su MPS float16 può essere ok, ma bf16 spesso è più stabile dove supportato.
        # Restiamo conservativi: fp16 su cuda/mps, fp32 su cpu.
        if dtype is None:
            if self.device in ("cuda", "mps"):
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        # === Tokenizer ===
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # pad_token: necessario per generation
        if self.tokenizer.pad_token is None:
            # per molti causal LM si usa eos come pad
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # === Modello base ===
        # device_map="auto" SOLO su CUDA (su MPS può fare mapping strani)
        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        try:
            # transformers “nuovi” (come il tuo) preferiscono dtype=
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                **({k: v for k, v in model_kwargs.items() if k != "torch_dtype"}),
            )
        except TypeError:
            # fallback per transformers “vecchi”
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                **model_kwargs,
            )

        # Se NON CUDA, spostiamo esplicitamente sul device
        # (con CUDA+device_map auto spesso è già sharded)
        if self.device != "cuda":
            base_model.to(self.device)

        # === Adapter LoRA opzionale ===
        if self.lora_path is not None:
            # Import lazy: baseline non richiede peft installato
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        else:
            self.model = base_model

        self.model.eval()

    def _build_inputs(self, prompt: str) -> tuple[torch.Tensor, int]:
        """
        Costruisce input_ids e ritorna anche input_len (per tagliare l'output generato).

        Se esiste una chat_template nel tokenizer, usa apply_chat_template con
        add_generation_prompt=True (fondamentale per molti Instruct model).
        Altrimenti fallback: tokenizzazione diretta del prompt.
        """
        # Chat-template path (Qwen Instruct, Mistral Instruct, ecc.)
        has_chat_template = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
            and str(self.tokenizer.chat_template).strip() != ""
        )

        if has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        else:
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"]

        input_ids = input_ids.to(self.device)
        input_len = input_ids.shape[-1]
        return input_ids, input_len

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
        Ritorna SOLO il testo generato (continuation), non include il prompt.

        Nota: il parsing di <answer> lo fate a livello di pipeline TISER.
        """
        if do_sample is None:
            do_sample = temperature > 0.0

        input_ids, input_len = self._build_inputs(prompt)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Taglia via il prompt: teniamo solo la continuation
        gen_only = generated_ids[0, input_len:]
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True)

        return text.strip()

    @torch.inference_mode()
    def generate_answer_only(
        self,
        prompt: str,
        extractor_fn: Callable[[str], str],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        """
        Variante comoda: genera output e poi applica una funzione
        che estrae la parte desiderata (es. il contenuto di <answer>).
        """
        text = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return extractor_fn(text)