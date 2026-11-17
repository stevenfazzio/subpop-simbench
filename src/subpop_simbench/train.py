"""Train a LoRA adapter the way SubPOP trained its released ones.

Follows SubPOP's fork of llama-recipes (`subpop/train/` in its repository).

Copied from it:

- The loss: forward KL between the target distribution and the model's next-token
  probabilities over the label tokens " A" to " J", renormalised over those ten, read at
  the last prompt position (`distribution_loss`).
- The adapter: rank-8 LoRA on the attention query and value projections, alpha 32,
  dropout 0.05, PEFT's default initialisation.
- AdamW without weight decay, 256 examples per step, and a cosine schedule with linear
  warm-up (`lr_multiplier`).
- Validation after every epoch, with the best epoch recorded.

Different here:

- One process with gradient accumulation, where SubPOP ran two with FSDP.
- Adapter weights are float32 under bfloat16 autocast. SubPOP trained in pure bfloat16 with
  its any-precision AdamW.
- Only the prompt is fed to the model. SubPOP appends a placeholder answer after the
  position it reads, which a causal model cannot see from that position.
- Every epoch's adapter is kept, so any of them can be scored later. SubPOP keeps the best.

Not recorded by SubPOP, so chosen here: the number of epochs, the warm-up ratio, and which
questions are held out for validation.

A run lives in data/adapters/<tag>/:

    meta.json          what was run, where, and how it ended (including the best epoch)
    metrics.json       one record per epoch: losses, learning rate, speed
    epochs/epoch-NNN/  the adapter after each epoch, loadable with PeftModel.from_pretrained
    state.pt           adapter and optimiser state after the last finished epoch

Killing a run loses at most the epoch in progress; re-running the same command continues
from `state.pt`. A different configuration under the same tag is refused.
"""

import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from subpop_simbench.infer import _resolve_dtype, pick_device, tokenize
from subpop_simbench.io import ADAPTERS_DIR, TRAIN_EXAMPLES_PATH, sha256_file, write_json_safely
from subpop_simbench.train_data import MAX_OPTIONS

# How a prompt becomes token ids.
#
# default     the tokenizer adds its own BOS, as this repo's inference does.
# bos_string  the BOS token is written into the text and no special tokens are added, as
#             SubPOP's training code does: encode(bos_token + prompt).
#
# On SentencePiece vocabularies the two can differ in the first word: "▁Question" under
# `default`, "Question" under `bos_string`. Whether they differ depends on the tokenizer and
# on the transformers version. With the version SubPOP pinned (4.48.2) its training code
# gave "Question" for Mistral-7B and "▁Question" for Llama-2. With the version used here
# (5.23) `bos_string` gives "Question" for both. So reproducing SubPOP's Mistral-7B training
# takes `bos_string`, and reproducing its Llama-2 training takes `default`.
TOKENIZATIONS = ("default", "bos_string")


@dataclass(frozen=True)
class TrainConfig:
    """Everything that decides what gets trained. Defaults are SubPOP's where it states them."""

    base: str
    base_revision: str | None = None
    dtype: str = "auto"  # auto: bfloat16 on CUDA, float32 elsewhere
    tokenization: str = "default"
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lr: float = 2e-4
    weight_decay: float = 0.0
    batch_size: int = 256  # examples per optimiser step
    epochs: int = 20  # not recorded by SubPOP; its example configuration says 50
    warmup_ratio: float = 0.05  # not recorded by SubPOP
    seed: int = 0
    max_steps: int | None = None  # stop early after this many steps, to measure a run


@dataclass
class Encoded:
    """Examples ready for the model: token ids and zero-padded target distributions."""

    input_ids: list[list[int]]
    targets: np.ndarray  # (n, MAX_OPTIONS)


def _say(message: str) -> None:
    """Print a progress line at once, so a log shows it even when output is piped."""
    print(message, flush=True)


def label_token_ids(tokenizer) -> list[int]:
    """SubPOP's training labels: the last token id of " A", " B", ... " J"."""
    return [
        tokenizer.encode(" " + chr(ord("A") + i), add_special_tokens=False)[-1]
        for i in range(MAX_OPTIONS)
    ]


def tokenize_for_training(tokenizer, prompts: list[str], tokenization: str) -> list[list[int]]:
    """Token ids per prompt under one of `TOKENIZATIONS`."""
    if tokenization == "default":
        return tokenize(tokenizer, prompts)
    if tokenization == "bos_string":
        if tokenizer.bos_token is None:
            raise ValueError("this tokenizer has no BOS token; use tokenization='default'")
        return [
            tokenizer.encode(tokenizer.bos_token + prompt, add_special_tokens=False)
            for prompt in prompts
        ]
    raise ValueError(f"unknown tokenization {tokenization!r}; expected one of {TOKENIZATIONS}")


def encode_examples(tokenizer, examples: pd.DataFrame, tokenization: str) -> Encoded:
    targets = np.zeros((len(examples), MAX_OPTIONS), dtype=np.float32)
    for row, target in enumerate(examples["target"]):
        targets[row, : len(target)] = target
    input_ids = tokenize_for_training(tokenizer, examples["prompt"].tolist(), tokenization)
    return Encoded(input_ids=input_ids, targets=targets)


def distribution_loss(
    logits_last: torch.Tensor, label_ids: list[int], target: torch.Tensor
) -> torch.Tensor:
    """SubPOP's forward-KL loss, one value per example.

    logits_last  (batch, vocab) logits at the last prompt position.
    target       (batch, MAX_OPTIONS) target shares, zero for options a question lacks.

    A softmax over the ten label logits equals SubPOP's softmax over the vocabulary
    renormalised over the ten labels, and does not underflow. The 1e-8 terms are SubPOP's.

    SubPOP's targets sum to one plus the refusal share. Such a target is still minimised by
    predicting it renormalised; the excess only weights the example more heavily.
    """
    probs = torch.softmax(logits_last[:, label_ids].float(), dim=-1)
    cross_entropy = -(target * torch.log(probs + 1e-8)).sum(dim=-1)
    negative_entropy = (target * torch.log(target + 1e-8)).sum(dim=-1)
    return cross_entropy + negative_entropy


def lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    """SubPOP's schedule: linear warm-up from zero, then cosine decay to zero."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def load_for_training(config: TrainConfig, device: str):
    """(model, tokenizer): the base model with a fresh, trainable LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(config.base, revision=config.base_revision)
    kwargs: dict = {"revision": config.base_revision, "dtype": _resolve_dtype(config.dtype, device)}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    base = AutoModelForCausalLM.from_pretrained(config.base, **kwargs)
    if "device_map" not in kwargs:
        base.to(device)
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    torch.manual_seed(config.seed)  # the adapter's A matrices start random
    return get_peft_model(base, lora), tokenizer


def pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0


def _losses(
    model, encoded: Encoded, indices: list[int], label_ids: list[int], pad_id: int, amp: bool
) -> torch.Tensor:
    """Per-example loss for `indices`, from one right-padded forward pass."""
    lengths = [len(encoded.input_ids[i]) for i in indices]
    width = max(lengths)
    ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(indices), width), dtype=torch.long)
    for row, i in enumerate(indices):
        ids[row, : lengths[row]] = torch.tensor(encoded.input_ids[i], dtype=torch.long)
        mask[row, : lengths[row]] = 1
    device = model.get_input_embeddings().weight.device
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    with autocast:
        logits = model(
            input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False
        ).logits
    rows = torch.arange(len(indices), device=logits.device)
    last = torch.tensor(lengths, device=logits.device) - 1
    target = torch.from_numpy(encoded.targets[indices]).to(logits.device)
    return distribution_loss(logits[rows, last], label_ids, target)


@torch.no_grad()
def evaluate(
    model,
    encoded: Encoded,
    label_ids: list[int],
    pad_id: int,
    micro_batch_size: int = 32,
    amp: bool = False,
) -> float:
    """Mean loss per example, with dropout off.

    SubPOP averages per-batch means instead, which differs only through a smaller last batch.
    """
    was_training = model.training
    model.eval()
    order = sorted(range(len(encoded.input_ids)), key=lambda i: -len(encoded.input_ids[i]))
    total = 0.0
    for start in range(0, len(order), micro_batch_size):
        chunk = order[start : start + micro_batch_size]
        total += _losses(model, encoded, chunk, label_ids, pad_id, amp).sum().item()
    model.train(was_training)
    return total / len(order)


def _save_adapter(model, path: Path) -> None:
    """Write the adapter so `path` only ever holds a complete one."""
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    if path.exists():  # an epoch being redone after a crash
        shutil.rmtree(path)
    os.replace(tmp, path)


def _save_state(state: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def train(
    config: TrainConfig,
    tag: str,
    examples_path: Path = TRAIN_EXAMPLES_PATH,
    adapters_dir: Path = ADAPTERS_DIR,
    micro_batch_size: int = 32,
    device: str | None = None,
    log_every: int = 20,
) -> Path:
    """Train an adapter and return its run directory.

    `micro_batch_size` (examples per forward pass) and `device` say how this session
    computes; they are not part of the run's identity.
    """
    if config.tokenization not in TOKENIZATIONS:
        raise ValueError(f"unknown tokenization {config.tokenization!r}")
    run_dir = Path(adapters_dir) / tag
    meta_path, state_path = run_dir / "meta.json", run_dir / "state.pt"
    # Through JSON, so a stored identity compares equal to a fresh one (tuples become lists).
    identity = json.loads(
        json.dumps({"config": asdict(config), "examples_sha256": sha256_file(examples_path)})
    )
    earlier_sessions: list[dict] = []
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous["identity"] != identity:
            raise SystemExit(
                f"{run_dir} holds a different run.\n  there: {previous['identity']}\n"
                f"  asked: {identity}\n"
                "Use another --tag, or delete that directory if it is no longer wanted."
            )
        if previous["status"] != "running":
            _say(f"{tag}: already finished ({previous['status']}) at {run_dir}")
            return run_dir
        earlier_sessions = previous.get("sessions", [])

    examples = pd.read_parquet(examples_path)
    device = device or pick_device()
    model, tokenizer = load_for_training(config, device)
    amp = device == "cuda" and next(model.parameters()).dtype == torch.bfloat16
    train_set = encode_examples(
        tokenizer, examples[examples["split"] == "train"], config.tokenization
    )
    val_set = encode_examples(tokenizer, examples[examples["split"] == "val"], config.tokenization)
    n_train = len(train_set.input_ids)
    if not n_train or not val_set.input_ids:
        raise SystemExit(f"{examples_path} needs both training and validation examples")
    label_ids, pad_id = label_token_ids(tokenizer), pad_token_id(tokenizer)
    first = tokenizer.convert_ids_to_tokens(train_set.input_ids[0][:4])
    _say(f"{tag}: {n_train:,} training and {len(val_set.input_ids):,} validation examples")
    _say(f"{tag}: prompts start {first} under tokenization={config.tokenization!r}")

    steps_per_epoch = math.ceil(n_train / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(config.warmup_ratio * total_steps)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)

    history: list[dict] = []
    start_epoch = 0
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        history, start_epoch = state["history"], state["epoch"]
        _say(f"{tag}: resuming after epoch {start_epoch} of {config.epochs}")

    session = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "from_epoch": start_epoch,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "micro_batch_size": micro_batch_size,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    meta = {"identity": identity, "status": "running", "sessions": [*earlier_sessions, session]}
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_safely(meta, meta_path)

    stopped_early = False
    for epoch in range(start_epoch, config.epochs):
        if config.max_steps is not None and epoch * steps_per_epoch >= config.max_steps:
            stopped_early = True
            break
        # Seeded per epoch, so a resumed run shuffles and drops out exactly as a fresh one.
        torch.manual_seed(config.seed * 100_003 + epoch)
        order = np.random.default_rng([config.seed, epoch]).permutation(n_train)
        model.train()
        started = time.monotonic()
        loss_sum, steps, tokens = 0.0, 0, 0
        for step in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + step
            if config.max_steps is not None and global_step >= config.max_steps:
                stopped_early = True
                break
            lr = config.lr * lr_multiplier(global_step, warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = order[step * config.batch_size : (step + 1) * config.batch_size].tolist()
            batch.sort(key=lambda i: -len(train_set.input_ids[i]))
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(batch), micro_batch_size):
                chunk = batch[start : start + micro_batch_size]
                losses = _losses(model, train_set, chunk, label_ids, pad_id, amp)
                loss = losses.sum() / len(batch)  # so the step's gradient is the batch mean
                loss.backward()
                loss_sum += loss.item()
                tokens += sum(len(train_set.input_ids[i]) for i in chunk)
            optimizer.step()
            steps += 1
            if log_every and steps % log_every == 0:
                rate = tokens / (time.monotonic() - started)
                _say(
                    f"{tag}: epoch {epoch + 1}, step {steps}/{steps_per_epoch}, "
                    f"loss {loss_sum / steps:.4f}, lr {lr:.2e}, {rate:,.0f} tok/s",
                )
        train_seconds = time.monotonic() - started
        val_loss = evaluate(model, val_set, label_ids, pad_id, micro_batch_size, amp)
        history.append(
            {
                "epoch": epoch + 1,
                "steps": steps,
                "partial": stopped_early,
                "train_loss": loss_sum / max(1, steps),
                "val_loss": val_loss,
                "lr_at_end": lr,
                "train_seconds": round(train_seconds, 1),
                "train_tokens": tokens,
                "tokens_per_second": round(tokens / train_seconds, 1),
            }
        )
        _say(
            f"{tag}: epoch {epoch + 1} done, train loss {history[-1]['train_loss']:.4f}, "
            f"validation loss {val_loss:.4f}, {train_seconds / 60:.1f} min",
        )
        if not stopped_early:
            # Adapter first, then the state that says the epoch is finished.
            _save_adapter(model, run_dir / "epochs" / f"epoch-{epoch + 1:03d}")
            _save_state(
                {
                    "epoch": epoch + 1,
                    "adapter": {
                        k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()
                    },
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                },
                state_path,
            )
        write_json_safely(history, run_dir / "metrics.json")
        if stopped_early:
            break

    finished = [record for record in history if not record["partial"]]
    if finished:
        best = min(finished, key=lambda record: record["val_loss"])
        meta["best_epoch"] = best["epoch"]
        meta["best_val_loss"] = best["val_loss"]
        meta["best_adapter"] = f"epochs/epoch-{best['epoch']:03d}"
    meta["status"] = "stopped_at_max_steps" if stopped_early else "complete"
    meta["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    write_json_safely(meta, meta_path)
    _say(f"{tag}: {meta['status']}; results in {run_dir}")
    return run_dir
