"""Build SubPOP's fine-tuning examples from its released training file.

    uv run python scripts/fetch_subpop.py          # once: the gated SubPOP-Train file
    uv run python scripts/build_training_data.py

Writes data/subpop/train_examples.parquet: one prompt and target distribution per
(question, subgroup), with questions split 90% / 10% into training and validation. The
file quotes Pew questions, so it stays under data/ and is not committed.
"""

import argparse

from subpop_simbench.data import load_subpop_steering
from subpop_simbench.io import SUBPOP_DIR, TRAIN_EXAMPLES_PATH, write_parquet_safely
from subpop_simbench.overlap import load_subpop_records
from subpop_simbench.train_data import build_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--val-ratio", type=float, default=0.1, help="share of questions held out")
    parser.add_argument("--seed", type=int, default=0, help="seed of the question split")
    args = parser.parse_args()

    source = SUBPOP_DIR / "subpop_train.jsonl"
    if not source.exists():
        raise SystemExit(f"{source} not found: run scripts/fetch_subpop.py first")
    records = load_subpop_records(source)
    examples = build_examples(records, load_subpop_steering(), args.val_ratio, args.seed)
    write_parquet_safely(examples, TRAIN_EXAMPLES_PATH)

    print(f"{len(records):,} SubPOP-Train rows in, {len(examples):,} examples out (0 dropped)")
    by_split = examples.groupby("split").agg(
        questions=("qkey", "nunique"), examples=("example_id", "size")
    )
    print(by_split.to_string())
    sums = examples["target"].map(sum)
    print(
        f"options per question: {examples['n_options'].min()} to {examples['n_options'].max()}; "
        f"target sums: {sums.min():.3f} to {sums.max():.3f} (median {sums.median():.3f})"
    )
    print(f"wrote {TRAIN_EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
