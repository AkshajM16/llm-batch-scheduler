from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    scheduler_mode: Literal["sequential", "dynamic"] = "dynamic"
    backend: Literal["transformers", "fake"] = "transformers"
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    max_batch_size: int = Field(default=4, ge=1, le=128)
    max_wait_ms: float = Field(default=10.0, ge=0, le=10_000)
    max_queue_size: int = Field(default=128, ge=1)
    request_timeout_s: float = Field(default=30.0, gt=0)
    length_buckets: tuple[int, ...] = (32, 128, 512)
    max_input_tokens: int = Field(default=512, ge=1)
    min_new_tokens: int = Field(default=0, ge=0, le=512)
    max_new_tokens: int = Field(default=16, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0)
    seed: int = 7
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    fake_delay_ms: float = Field(default=5.0, ge=0)

    @field_validator("length_buckets", mode="before")
    @classmethod
    def parse_buckets(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(int(part.strip()) for part in value.split(",") if part.strip())
        return value

    @field_validator("length_buckets")
    @classmethod
    def validate_buckets(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(item < 1 for item in value):
            raise ValueError("length buckets must contain positive integers")
        return tuple(sorted(set(value)))

    @model_validator(mode="after")
    def validate_generation_lengths(self) -> Settings:
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
