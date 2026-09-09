from __future__ import annotations

from types import SimpleNamespace

import torch

from llm_scheduler.backend import TransformersBackend


def test_generated_length_stops_at_first_eos_padding() -> None:
    backend = object.__new__(TransformersBackend)
    backend.tokenizer = SimpleNamespace(eos_token_id=2)

    assert backend._generated_length(torch.tensor([8, 9, 2, 2, 2])) == 3
    assert backend._generated_length(torch.tensor([8, 9, 10])) == 3
