"""Harness mechanics on a tiny randomly-initialised Llama with the real Llama-2 tokenizer.

The model is built locally, so no weights are downloaded. The tokenizer comes from the
gated meta-llama/Llama-2-7b-hf repo; without access to it these tests are skipped.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from subpop_simbench import runner
from subpop_simbench.infer import (
    ModelSpec,
    iter_first_token_probs,
    load_model,
    plan_batches,
    tokenize,
)
from subpop_simbench.readout import letter_ids
from subpop_simbench.results import load_run

PROMPTS = [
    "Question: Is the sky blue?\nA. Yes\nB. No\nAnswer as a choice between A.,B.\n\nAnswer:",
    "You are from Finland.\n\nQuestion: Tea or coffee, and why would anyone pick the other "
    "one given the choice?\nA. Tea\nB. Coffee\nC. Neither\nAnswer as a choice between "
    "A.,B.,C.\n\nAnswer:",
    "**Question**: Pick one.\n\nOptions:\n(A): Left\n(B): Right\n**Answer**: (",
    "Short\n\nAnswer:",
]


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


def _all_probs(model, tokenizer, prompts, max_batch_tokens):
    input_ids = tokenize(tokenizer, prompts)
    out = {rule: np.zeros((len(prompts), 26)) for rule in ("simbench", "subpop")}
    for batch in iter_first_token_probs(model, tokenizer, input_ids, max_batch_tokens):
        for rule in out:
            out[rule][batch.indices] = batch.letter_probs[rule]
    return out


def test_readout_ids_on_llama2(llama2_tokenizer):
    spaced_a, bare_a, underscore = 319, 29909, 903  # "▁A", "A", "▁_"
    assert letter_ids(llama2_tokenizer, "simbench")[0] == [spaced_a, underscore, bare_a]
    assert letter_ids(llama2_tokenizer, "subpop")[0] == [bare_a, spaced_a]


def test_plan_batches_respects_budget_and_covers_everything():
    lengths = [5, 50, 7, 49, 6, 51]
    batches = plan_batches(lengths, max_batch_tokens=110)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) == 1 or len(batch) * max(lengths[i] for i in batch) <= 110


def test_batching_does_not_change_probabilities(tiny_paths):
    model, tokenizer = load_model(ModelSpec(base=str(tiny_paths[0])), device="cpu")
    together = _all_probs(model, tokenizer, PROMPTS, max_batch_tokens=10_000)
    one_by_one = _all_probs(model, tokenizer, PROMPTS, max_batch_tokens=1)
    for rule in together:
        np.testing.assert_allclose(together[rule], one_by_one[rule], rtol=1e-5, atol=1e-9)


def test_probabilities_match_a_plain_forward_pass(tiny_paths, llama2_tokenizer):
    model, tokenizer = load_model(ModelSpec(base=str(tiny_paths[0])), device="cpu")
    ours = _all_probs(model, tokenizer, PROMPTS, max_batch_tokens=10_000)["subpop"]
    reference = AutoModelForCausalLM.from_pretrained(tiny_paths[0]).eval()
    for i, prompt in enumerate(PROMPTS):
        ids = llama2_tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            probs = torch.softmax(reference(ids).logits[0, -1].double(), dim=-1)
        expected = [
            probs[319].item() + probs[29909].item(),
            probs[350].item() + probs[29933].item(),
        ]
        np.testing.assert_allclose(ours[i, :2], expected, rtol=1e-5)


def test_adapter_is_applied_and_equals_merged_weights(tiny_paths):
    base_path, adapter_path = (str(p) for p in tiny_paths)
    base_model, tokenizer = load_model(ModelSpec(base=base_path), device="cpu")
    tuned_model, _ = load_model(ModelSpec(base=base_path, adapter=adapter_path), device="cpu")
    base = _all_probs(base_model, tokenizer, PROMPTS, 10_000)["subpop"]
    tuned = _all_probs(tuned_model, tokenizer, PROMPTS, 10_000)["subpop"]
    assert not np.allclose(base, tuned, rtol=1e-3), "adapter had no effect"

    merged_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_path), adapter_path
    ).merge_and_unload()
    merged = _all_probs(merged_model.eval(), tokenizer, PROMPTS, 10_000)["subpop"]
    np.testing.assert_allclose(tuned, merged, rtol=1e-4, atol=1e-9)


@pytest.fixture
def tiny_run_inputs(tmp_path):
    """Prompt and item tables in the pipeline's on-disk layout."""
    n = 23
    items = pd.DataFrame(
        {"item_id": [f"t-{i:03d}" for i in range(n)], "n_options": [2 + i % 4 for i in range(n)]}
    )
    prompts = pd.DataFrame(
        [
            (item_id, condition, f"{PROMPTS[i % len(PROMPTS)]} [{i}]\n\nAnswer:")
            for i, item_id in enumerate(items["item_id"])
            for condition in ("simbench", "subpop_skeleton")
        ],
        columns=["item_id", "condition", "prompt"],
    )
    items.to_parquet(tmp_path / "items.parquet", index=False)
    prompts.to_parquet(tmp_path / "prompts.parquet", index=False)
    return tmp_path


def _run(tiny_paths, inputs, results_dir, tag="run", **kwargs):
    return runner.run_inference(
        ModelSpec(base=str(tiny_paths[0]), adapter=str(tiny_paths[1])),
        tag=tag,
        shard_rows=10,
        max_batch_tokens=400,
        prompts_path=inputs / "prompts.parquet",
        items_path=inputs / "items.parquet",
        results_dir=results_dir,
        **kwargs,
    )


def test_run_writes_one_normalisable_prediction_per_prompt(tiny_paths, tiny_run_inputs, tmp_path):
    final = _run(tiny_paths, tiny_run_inputs, tmp_path / "results")
    df = load_run(final.parent)
    assert len(df) == 46 and not df.duplicated(["item_id", "condition"]).any()
    n_options = pd.read_parquet(tiny_run_inputs / "items.parquet").set_index("item_id")["n_options"]
    assert [len(p) for p in df["pred"]] == n_options.loc[df["item_id"]].tolist()
    assert np.allclose([p.sum() for p in df["pred"]], 1.0)
    assert not (final.parent / "shards").exists()


def test_interrupted_run_resumes_to_the_same_result(
    tiny_paths, tiny_run_inputs, tmp_path, monkeypatch
):
    clean = pd.read_parquet(_run(tiny_paths, tiny_run_inputs, tmp_path / "clean"))

    real_write = runner.write_parquet_safely
    calls = {"n": 0}

    def crash_on_third_shard(df, path):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        real_write(df, path)

    monkeypatch.setattr(runner, "write_parquet_safely", crash_on_third_shard)
    with pytest.raises(KeyboardInterrupt):
        _run(tiny_paths, tiny_run_inputs, tmp_path / "crashy")
    run_dir = tmp_path / "crashy" / "run"
    assert len(list((run_dir / "shards").glob("part-*.parquet"))) == 2
    assert not (run_dir / "predictions.parquet").exists()

    monkeypatch.setattr(runner, "write_parquet_safely", real_write)
    resumed = pd.read_parquet(_run(tiny_paths, tiny_run_inputs, tmp_path / "crashy"))
    assert resumed[["item_id", "condition"]].equals(clean[["item_id", "condition"]])
    for col in ("p_simbench", "p_subpop"):
        for a, b in zip(resumed[col], clean[col], strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-12)


def test_tag_cannot_be_reused_for_a_different_run(tiny_paths, tiny_run_inputs, tmp_path):
    _run(tiny_paths, tiny_run_inputs, tmp_path / "results")
    with pytest.raises(SystemExit):
        _run(tiny_paths, tiny_run_inputs, tmp_path / "results", conditions=["simbench"])
