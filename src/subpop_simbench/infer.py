"""Load a base model (optionally with a LoRA adapter) and read first-token answer probabilities.

Each prompt costs one forward pass. Nothing is generated: we take the next-token
distribution at the end of the prompt over the full vocabulary and sum it into per-letter
probabilities under each readout rule.
"""

from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from subpop_simbench.readout import RULES, readout_matrix


@dataclass(frozen=True)
class ModelSpec:
    base: str
    adapter: str | None = None
    base_revision: str | None = None
    adapter_revision: str | None = None
    dtype: str = "auto"  # auto: bfloat16 on CUDA, float32 elsewhere
    quantization: str | None = None  # None, "8bit" or "4bit" (bitsandbytes, CUDA only)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchResult:
    indices: list[int]  # positions in the prompt list passed in
    n_tokens: list[int]
    letter_probs: dict[str, np.ndarray]  # rule -> (batch, 26), unnormalised
    top_token: list[str]
    top_prob: list[float]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    return getattr(torch, name)


def load_model(spec: ModelSpec, device: str | None = None):
    """Return (model, tokenizer) in eval mode. The adapter is applied unmerged."""
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(spec.base, revision=spec.base_revision)
    kwargs: dict = {"revision": spec.base_revision, "dtype": _resolve_dtype(spec.dtype, device)}
    if spec.quantization is not None:
        from transformers import BitsAndBytesConfig

        if spec.quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif spec.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            raise ValueError(f"unknown quantization {spec.quantization!r}")
        kwargs["device_map"] = "auto"
    elif device == "cuda":
        kwargs["device_map"] = "auto"  # spreads a large model over every visible GPU
    model = AutoModelForCausalLM.from_pretrained(spec.base, **kwargs)
    if "device_map" not in kwargs:
        model.to(device)
    if spec.adapter is not None:
        model = PeftModel.from_pretrained(model, spec.adapter, revision=spec.adapter_revision)
    model.eval()
    return model, tokenizer


def plan_batches(lengths: list[int], max_batch_tokens: int) -> list[list[int]]:
    """Group prompt indices, longest first, so each padded batch fits the token budget."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        # Sorted longest-first, so the first prompt in a batch sets its padded width.
        width = lengths[current[0]] if current else lengths[i]
        if current and (len(current) + 1) * width > max_batch_tokens:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def tokenize(tokenizer, prompts: list[str]) -> list[list[int]]:
    """Token ids per prompt, the way both upstream projects tokenise: the tokenizer's default
    special tokens, i.e. one BOS where the model has one."""
    return tokenizer(prompts, add_special_tokens=True)["input_ids"]


@torch.inference_mode()
def iter_first_token_probs(
    model, tokenizer, input_ids: list[list[int]], max_batch_tokens: int = 8192
) -> Iterator[BatchResult]:
    """Yield per-letter next-token probabilities for every prompt, batch by batch.

    `input_ids` comes from `tokenize`. Batches are right-padded; with a causal model the
    padding cannot affect the position we read.
    """
    lengths = [len(ids) for ids in input_ids]
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and max(lengths) > max_positions:
        raise ValueError(f"longest prompt is {max(lengths)} tokens; model takes {max_positions}")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    in_device = model.get_input_embeddings().weight.device
    matrices: dict[str, torch.Tensor] = {}

    for batch in plan_batches(lengths, max_batch_tokens):
        width = lengths[batch[0]]
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, i in enumerate(batch):
            ids[row, : lengths[i]] = torch.tensor(input_ids[i], dtype=torch.long)
            mask[row, : lengths[i]] = 1
        logits = model(input_ids=ids.to(in_device), attention_mask=mask.to(in_device)).logits
        last = torch.tensor([lengths[i] - 1 for i in batch], device=logits.device)
        rows = torch.arange(len(batch), device=logits.device)
        # Softmax on CPU in float64: identical arithmetic on every device, and an answer
        # letter far below the top token keeps a nonzero probability instead of underflowing.
        # Two steps: MPS has no float64, so the cast has to happen after the move.
        probs = torch.softmax(logits[rows, last].to("cpu").to(torch.float64), dim=-1)
        del logits

        if not matrices:
            for rule in RULES:
                matrices[rule] = readout_matrix(tokenizer, rule, probs.shape[-1]).double()
        top_prob, top_id = probs.max(dim=-1)
        yield BatchResult(
            indices=batch,
            n_tokens=[lengths[i] for i in batch],
            letter_probs={rule: (probs @ matrices[rule].T).numpy() for rule in RULES},
            top_token=tokenizer.convert_ids_to_tokens(top_id.tolist()),
            top_prob=top_prob.tolist(),
        )
