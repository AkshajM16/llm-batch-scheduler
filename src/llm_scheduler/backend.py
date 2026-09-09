from __future__ import annotations

import asyncio
import random
import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from llm_scheduler.config import Settings
from llm_scheduler.types import BackendOutput, BatchOutput


class GenerationBackend(ABC):
    @abstractmethod
    def estimate_tokens(self, text: str) -> int: ...

    @abstractmethod
    async def generate(self, prompts: list[str]) -> BatchOutput: ...


class FakeBackend(GenerationBackend):
    """Deterministic scheduler-test backend; never used for inference claims."""

    def __init__(self, delay_ms: float = 5.0) -> None:
        self.delay_s = delay_ms / 1_000

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    async def generate(self, prompts: list[str]) -> BatchOutput:
        await asyncio.sleep(self.delay_s)
        lengths = [self.estimate_tokens(prompt) for prompt in prompts]
        padded = max(lengths) * len(lengths)
        return BatchOutput(
            outputs=[
                BackendOutput(
                    generated_text=f"response:{prompt}",
                    prompt_tokens=length,
                    generated_tokens=1,
                )
                for prompt, length in zip(prompts, lengths, strict=True)
            ],
            padded_prompt_tokens=padded,
            useful_prompt_tokens=sum(lengths),
        )


class TransformersBackend(GenerationBackend):
    def __init__(self, settings: Settings) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.settings = settings
        self._tokenizer_lock = threading.Lock()
        self.device = self._select_device(settings.device)
        self.dtype = self._select_dtype(settings.dtype, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested != "auto":
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _select_dtype(requested: str, device: torch.device) -> torch.dtype:
        if requested != "auto":
            return getattr(torch, requested)
        return torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    def estimate_tokens(self, text: str) -> int:
        with self._tokenizer_lock:
            count = len(self.tokenizer.encode(text, add_special_tokens=True))
        return min(count, self.settings.max_input_tokens)

    async def generate(self, prompts: list[str]) -> BatchOutput:
        return await asyncio.to_thread(self._generate_sync, prompts)

    def _generated_length(self, tokens: torch.Tensor) -> int:
        token_ids = tokens.tolist()
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and eos_id in token_ids:
            return token_ids.index(eos_id) + 1
        return len(token_ids)

    def _generate_sync(self, prompts: list[str]) -> BatchOutput:
        with self._tokenizer_lock:
            encoded: dict[str, Any] = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.settings.max_input_tokens,
            )
        attention = encoded["attention_mask"]
        prompt_lengths = attention.sum(dim=1).tolist()
        padded = int(attention.numel())
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "min_new_tokens": self.settings.min_new_tokens,
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": self.settings.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.settings.temperature > 0:
            generation_kwargs["temperature"] = self.settings.temperature
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        input_width = inputs["input_ids"].shape[1]
        new_tokens = generated[:, input_width:]
        with self._tokenizer_lock:
            outputs = [
                BackendOutput(
                    generated_text=self.tokenizer.decode(tokens, skip_special_tokens=True),
                    prompt_tokens=int(prompt_length),
                    generated_tokens=self._generated_length(tokens),
                )
                for tokens, prompt_length in zip(new_tokens.cpu(), prompt_lengths, strict=True)
            ]
        return BatchOutput(
            outputs=outputs,
            padded_prompt_tokens=padded,
            useful_prompt_tokens=int(sum(prompt_lengths)),
        )


def create_backend(settings: Settings) -> GenerationBackend:
    if settings.backend == "fake":
        return FakeBackend(settings.fake_delay_ms)
    return TransformersBackend(settings)
