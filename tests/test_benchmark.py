from pathlib import Path

from llm_scheduler.benchmark import configs_for_preset, portable_path
from llm_scheduler.loadgen import ROOT


def test_full_preset_contains_each_sweep_point_once() -> None:
    configs = configs_for_preset("full")
    bucketed = [config for config in configs if config.label == "dynamic_bucketed"]

    assert len(configs) == 26
    assert len(bucketed) == 20
    assert len({(config.max_batch_size, config.max_wait_ms) for config in bucketed}) == 20


def test_corpus_metadata_path_is_portable() -> None:
    assert portable_path(ROOT / "data" / "prompts.jsonl") == "data/prompts.jsonl"
    assert portable_path(Path("/outside/private/corpus.jsonl")) == "corpus.jsonl"
