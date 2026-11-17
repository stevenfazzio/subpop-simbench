"""Fixtures shared by the inference and training tests.

The model is built locally, so no weights are downloaded. The tokenizer comes from the
gated meta-llama/Llama-2-7b-hf repo; without access to it the tests that need it are
skipped.
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


@pytest.fixture(scope="session")
def llama2_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    except Exception as exc:  # no network, or no access to the gated repo
        pytest.skip(f"Llama-2 tokenizer unavailable: {type(exc).__name__}")


@pytest.fixture(scope="session")
def tiny_paths(tmp_path_factory, llama2_tokenizer):
    """A saved tiny base model and a LoRA adapter for it that changes its outputs."""
    root = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(llama2_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=512,
    )
    base = LlamaForCausalLM(config)
    base.save_pretrained(root / "base")
    llama2_tokenizer.save_pretrained(root / "base")

    # init_lora_weights=False gives a random, non-zero update. Same shape as SubPOP's
    # adapters: rank-8 LoRA on the attention query and value projections.
    lora = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], init_lora_weights=False
    )
    get_peft_model(base, lora).save_pretrained(root / "adapter")
    return root / "base", root / "adapter"
