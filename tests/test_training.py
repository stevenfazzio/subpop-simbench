"""The adapter trainer: SubPOP's loss and schedule, the training examples, and whole runs on
a tiny randomly-initialised model."""

import json
from dataclasses import replace

import pandas as pd
import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from subpop_simbench import train as train_module
from subpop_simbench.data import load_subpop_steering
from subpop_simbench.infer import ModelSpec, load_model
from subpop_simbench.prompts import render_subpop_exact
from subpop_simbench.train import (
    TrainConfig,
    distribution_loss,
    encode_examples,
    evaluate,
    label_token_ids,
    lr_multiplier,
    pad_token_id,
    tokenize_for_training,
    train,
)
from subpop_simbench.train_data import build_examples, split_questions

PROMPT = "Question: Is the sky blue?\nA. Yes\nB. No\nAnswer as a choice between A.,B.\n\nAnswer:"


def test_loss_equals_subpops_formula():
    torch.manual_seed(0)
    logits = torch.randn(5, 50) * 3
    label_ids = [3, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    target = torch.rand(5, 10)
    target[:, 4:] = 0  # four-option questions, padded to ten
    target /= target.sum(dim=-1, keepdim=True)
    target[0] *= 1.3  # SubPOP's targets sum to more than one when people refuse

    # Transcribed from SubPOP's subpop/train/utils/train_utils.py.
    probs = torch.softmax(logits, dim=-1)[:, label_ids]
    probs /= probs.sum(dim=-1, keepdim=True)
    expected = -torch.sum(target * torch.log(probs + 1e-8), dim=-1) + torch.sum(
        target * torch.log(target + 1e-8), dim=-1
    )
    assert torch.allclose(distribution_loss(logits, label_ids, target), expected, atol=1e-6)


def test_a_target_summing_above_one_is_minimised_by_its_normalised_form():
    shares = torch.tensor([0.5, 0.3, 0.4])  # sums to 1.2
    target = torch.cat([shares, torch.zeros(7)])[None]
    logits = torch.cat([torch.log(shares / shares.sum()), torch.full((7,), -30.0)])[None]
    logits.requires_grad_()
    distribution_loss(logits, list(range(10)), target).sum().backward()
    assert logits.grad.abs().max() < 1e-4


def test_schedule_warms_up_linearly_then_decays_to_zero():
    assert lr_multiplier(0, 10, 100) == 0.0
    assert lr_multiplier(5, 10, 100) == pytest.approx(0.5)
    assert lr_multiplier(10, 10, 100) == pytest.approx(1.0)
    assert lr_multiplier(55, 10, 100) == pytest.approx(0.5)
    assert lr_multiplier(100, 10, 100) == pytest.approx(0.0, abs=1e-12)
    decay = [lr_multiplier(step, 10, 100) for step in range(10, 101)]
    assert decay == sorted(decay, reverse=True)


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "qkey": ["Q1_W70", "Q1_W70", "Q2_W71", "Q2_W71"],
            "attribute": ["SEX", "RACE", "SEX", "RACE"],
            "group": ["Female", "Asian", "Female", "Asian"],
            "question": ["Is tea better than coffee?"] * 2 + ["How often do you vote?"] * 2,
            "options": [["Yes", "No", "Refused"]] * 2
            + [["Often", "Sometimes", "Never", "Refused"]] * 2,
            "responses": [[0.6, 0.4], [0.3, 0.7], [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]],
            "refusal_rate": [0.02, 0.0, 0.1, 0.05],
        }
    )


def test_examples_carry_subpops_prompt_and_unnormalised_target():
    steering = load_subpop_steering()
    examples = build_examples(_records(), steering)
    assert len(examples) == 4
    row = examples.set_index("example_id").loc["Q1_W70|SEX|Female"]
    assert row["prompt"] == render_subpop_exact(
        "SEX", "Female", "Is tea better than coffee?", ["Yes", "No", "Refused"], steering
    )
    assert row["prompt"].endswith("Answer as a choice between A.,B.,C.\n\nAnswer:")
    assert list(row["target"]) == pytest.approx([0.6, 0.4, 0.02])  # as SubPOP: not renormalised
    assert row["n_options"] == 3
    # Every subgroup's row of a question lands in the same split.
    assert (examples.groupby("qkey")["split"].nunique() == 1).all()


def test_options_and_shares_must_line_up():
    records = _records()
    records.at[0, "options"] = ["Yes", "No"]
    with pytest.raises(ValueError, match="2 options but 3 shares"):
        build_examples(records, load_subpop_steering())


def test_question_split_is_stable_and_sized():
    qkeys = [f"Q{i}" for i in range(50)]
    split = split_questions(qkeys + qkeys[:10])
    assert sum(value == "val" for value in split.values()) == 5
    assert split == split_questions(reversed(qkeys))
    assert split != split_questions(qkeys, seed=1)


def test_label_tokens_are_the_spaced_letters(llama2_tokenizer):
    tokens = llama2_tokenizer.convert_ids_to_tokens(label_token_ids(llama2_tokenizer))
    assert tokens == [f"▁{letter}" for letter in "ABCDEFGHIJ"]


def test_bos_string_tokenisation_is_subpops_training_code(llama2_tokenizer):
    ids = tokenize_for_training(llama2_tokenizer, [PROMPT], "bos_string")[0]
    assert ids == llama2_tokenizer.encode(
        llama2_tokenizer.bos_token + PROMPT, add_special_tokens=False
    )
    default = tokenize_for_training(llama2_tokenizer, [PROMPT], "default")[0]
    # Tripwire. Which tokenisation reproduces SubPOP's training for which model (see
    # train.TOKENIZATIONS) rests on the two differing only in the first word's marker.
    assert llama2_tokenizer.convert_ids_to_tokens([ids[1], default[1]]) == [
        "Question",
        "▁Question",
    ]
    assert ids[0] == default[0] == llama2_tokenizer.bos_token_id
    assert ids[2:] == default[2:]


def test_mistral_prompts_start_the_way_subpop_trained_them():
    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    except Exception as exc:  # no network
        pytest.skip(f"Mistral tokenizer unavailable: {type(exc).__name__}")
    ids = tokenize_for_training(tokenizer, [PROMPT], "bos_string")[0]
    assert tokenizer.convert_ids_to_tokens(ids[:2]) == ["<s>", "Question"]


@pytest.fixture
def tiny_examples(tmp_path):
    """Two kinds of question, each with one answer distribution a model can learn."""
    kinds = [
        ("Is tea better than coffee?", ["Yes", "No", "Refused"], [0.75, 0.2, 0.05]),
        (
            "How often do you vote?",
            ["Often", "Sometimes", "Never", "Refused"],
            [0.1, 0.2, 0.6, 0.2],
        ),
    ]
    rows = []
    for i in range(24):
        question, options, target = kinds[i % 2]
        listed = "\n".join(f"{chr(ord('A') + j)}. {option}" for j, option in enumerate(options))
        rows.append(
            {
                "example_id": f"e{i:02d}",
                "split": "val" if i >= 20 else "train",
                "qkey": f"Q{i}",
                "attribute": "SEX",
                "group": "Female",
                "prompt": f"Respondent {i}.\n\nQuestion: {question}\n{listed}\n\nAnswer:",
                "target": target,
                "n_options": len(options),
            }
        )
    path = tmp_path / "examples.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _config(tiny_paths, **overrides) -> TrainConfig:
    settings = {"lr": 2e-2, "batch_size": 8, "epochs": 3, "tokenization": "bos_string"}
    return TrainConfig(base=str(tiny_paths[0]), **{**settings, **overrides})


def _train(config, tag, examples, tmp_path):
    return train(
        config, tag, examples, tmp_path / "adapters", micro_batch_size=3, device="cpu", log_every=0
    )


def _validation_loss(tiny_paths, examples_path, adapter=None) -> float:
    spec = ModelSpec(base=str(tiny_paths[0]), adapter=str(adapter) if adapter else None)
    model, tokenizer = load_model(spec, device="cpu")
    examples = pd.read_parquet(examples_path)
    encoded = encode_examples(tokenizer, examples[examples["split"] == "val"], "bos_string")
    return evaluate(model, encoded, label_token_ids(tokenizer), pad_token_id(tokenizer))


def test_training_lowers_the_loss_and_saves_a_loadable_adapter(tiny_paths, tiny_examples, tmp_path):
    run_dir = _train(_config(tiny_paths), "run", tiny_examples, tmp_path)
    meta = json.loads((run_dir / "meta.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert meta["status"] == "complete"
    assert [record["epoch"] for record in metrics] == [1, 2, 3]
    assert all(record["steps"] == 3 for record in metrics)  # 20 examples in batches of 8

    before = _validation_loss(tiny_paths, tiny_examples)
    assert metrics[-1]["val_loss"] < before
    assert metrics[-1]["train_loss"] < metrics[0]["train_loss"]

    # Each epoch's adapter loads through the inference path and reproduces its recorded loss.
    for record in metrics:
        adapter = run_dir / "epochs" / f"epoch-{record['epoch']:03d}"
        reloaded = _validation_loss(tiny_paths, tiny_examples, adapter)
        assert reloaded == pytest.approx(record["val_loss"], abs=1e-5)
    best = min(metrics, key=lambda record: record["val_loss"])
    assert meta["best_epoch"] == best["epoch"]
    assert (run_dir / meta["best_adapter"] / "adapter_model.safetensors").exists()


def test_interrupted_training_resumes_to_the_same_adapter(
    tiny_paths, tiny_examples, tmp_path, monkeypatch
):
    config = _config(tiny_paths, epochs=2)
    clean = _train(config, "clean", tiny_examples, tmp_path)

    save_state = train_module._save_state

    def crash_after_saving(state, path):
        save_state(state, path)
        raise KeyboardInterrupt

    monkeypatch.setattr(train_module, "_save_state", crash_after_saving)
    with pytest.raises(KeyboardInterrupt):
        _train(config, "crashy", tiny_examples, tmp_path)
    monkeypatch.setattr(train_module, "_save_state", save_state)
    crashy = tmp_path / "adapters" / "crashy"
    assert json.loads((crashy / "meta.json").read_text())["status"] == "running"
    assert not (crashy / "epochs" / "epoch-002").exists()

    resumed = _train(config, "crashy", tiny_examples, tmp_path)
    weights = [
        load_file(run / "epochs" / "epoch-002" / "adapter_model.safetensors")
        for run in (clean, resumed)
    ]
    assert weights[0].keys() == weights[1].keys()
    for name, tensor in weights[0].items():
        assert torch.allclose(tensor, weights[1][name], atol=1e-6), name
    losses = [
        [record["val_loss"] for record in json.loads((run / "metrics.json").read_text())]
        for run in (clean, resumed)
    ]
    assert losses[0] == pytest.approx(losses[1], abs=1e-6)


def test_max_steps_stops_early_and_keeps_nothing_resumable(tiny_paths, tiny_examples, tmp_path):
    run_dir = _train(_config(tiny_paths, max_steps=2), "smoke", tiny_examples, tmp_path)
    meta = json.loads((run_dir / "meta.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert meta["status"] == "stopped_at_max_steps"
    assert [(record["steps"], record["partial"]) for record in metrics] == [(2, True)]
    assert metrics[0]["tokens_per_second"] > 0
    assert not (run_dir / "state.pt").exists()
    assert not (run_dir / "epochs").exists()


def test_a_tag_cannot_hold_two_different_runs(tiny_paths, tiny_examples, tmp_path):
    config = _config(tiny_paths, epochs=1)
    _train(config, "run", tiny_examples, tmp_path)
    with pytest.raises(SystemExit, match="different run"):
        _train(replace(config, lr=1e-3), "run", tiny_examples, tmp_path)
