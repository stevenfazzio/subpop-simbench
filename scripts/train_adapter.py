"""Train a SubPOP-style adapter on a base model, or score one on the held-out questions.

    uv run python scripts/build_training_data.py
    uv run python scripts/train_adapter.py --tag mistral-7b-v0.1__retrain \\
        --base mistralai/Mistral-7B-v0.1 --tokenization bos_string

Output goes to data/adapters/<tag>/, one adapter per epoch. The run is resumable: repeat
the same command after a crash. Add `--max-steps N` (under its own tag) to stop after N
steps and report speed, before committing to a full run.

To get the loss of an existing model on the validation questions, for reference:

    uv run python scripts/train_adapter.py --eval-only --base mistralai/Mistral-7B-v0.1 \\
        --tokenization bos_string [--adapter jjssuh/mistral-7b-v0.1-subpop]

A released adapter has probably seen some of those questions in training, because SubPOP's
own split is not published. Its loss is a floor to aim for, not a like-for-like score.

Score a trained adapter on SimBench with the usual command, pointing --adapter at an epoch:

    uv run python scripts/run_inference.py --tag mistral-7b-v0.1__retrain \\
        --base mistralai/Mistral-7B-v0.1 \\
        --adapter data/adapters/mistral-7b-v0.1__retrain/epochs/epoch-020
"""

import argparse

import pandas as pd
import torch

from subpop_simbench.infer import ModelSpec, load_model, pick_device
from subpop_simbench.io import TRAIN_EXAMPLES_PATH
from subpop_simbench.train import (
    TOKENIZATIONS,
    TrainConfig,
    encode_examples,
    evaluate,
    label_token_ids,
    pad_token_id,
    train,
)


def eval_only(args) -> None:
    device = args.device or pick_device()
    spec = ModelSpec(
        base=args.base,
        adapter=args.adapter,
        base_revision=args.base_revision,
        adapter_revision=args.adapter_revision,
        dtype=args.dtype,
    )
    model, tokenizer = load_model(spec, device)
    examples = pd.read_parquet(TRAIN_EXAMPLES_PATH)
    examples = examples[examples["split"] == args.split]
    encoded = encode_examples(tokenizer, examples, args.tokenization)
    amp = device == "cuda" and next(model.parameters()).dtype == torch.bfloat16
    loss = evaluate(
        model,
        encoded,
        label_token_ids(tokenizer),
        pad_token_id(tokenizer),
        args.micro_batch_size,
        amp,
    )
    name = args.base + (f" + {args.adapter}" if args.adapter else "")
    print(f"{name}: loss {loss:.4f} on {len(examples):,} {args.split} examples")


def main() -> None:
    defaults = TrainConfig(base="")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", help="names the run directory under data/adapters/")
    parser.add_argument("--base", required=True, help="Hugging Face id or path of the base model")
    parser.add_argument("--base-revision")
    parser.add_argument("--dtype", default="auto", help="auto, bfloat16 or float32")
    parser.add_argument(
        "--tokenization",
        choices=TOKENIZATIONS,
        default=defaults.tokenization,
        help="bos_string reproduces SubPOP's Mistral-7B training; see train.py",
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-steps", type=int, help="stop after this many optimiser steps")
    parser.add_argument(
        "--micro-batch-size", type=int, default=32, help="examples per forward pass"
    )
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: best available")
    parser.add_argument("--eval-only", action="store_true", help="score a model, train nothing")
    parser.add_argument("--adapter", help="with --eval-only: a LoRA adapter to apply")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    args = parser.parse_args()

    if not TRAIN_EXAMPLES_PATH.exists():
        raise SystemExit(f"{TRAIN_EXAMPLES_PATH} not found: run scripts/build_training_data.py")
    if args.eval_only:
        eval_only(args)
        return
    if not args.tag:
        parser.error("--tag is required when training")
    if args.adapter:
        parser.error("--adapter only applies with --eval-only")
    config = TrainConfig(
        base=args.base,
        base_revision=args.base_revision,
        dtype=args.dtype,
        tokenization=args.tokenization,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    train(config, tag=args.tag, micro_batch_size=args.micro_batch_size, device=args.device)


if __name__ == "__main__":
    main()
