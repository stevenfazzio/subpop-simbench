"""Match SubPOP's training questions against SimBench's items by question text.

    uv run python scripts/fetch_subpop.py     # gated dataset, needs a Hugging Face token
    uv run python scripts/overlap.py

Writes data/analysis/overlap_items.parquet: one row per SimBench item with its closest
SubPOP-Train question. That file quotes both datasets' questions, so it stays under data/
and is not committed. `scripts/analyze.py` reads it to split scores by match status.
"""

import argparse

import pandas as pd

from subpop_simbench.analysis import prepare_items
from subpop_simbench.io import (
    ITEMS_PATH,
    OVERLAP_PATH,
    SUBPOP_DIR,
    TABLES_DIR,
    write_parquet_safely,
)
from subpop_simbench.overlap import STATUSES, load_subpop_records, match_items, training_copy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--examples", type=int, default=3, help="sample matches to print per band")
    parser.add_argument("--workers", type=int, default=2, help="CPU threads for the matching")
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    train_path = SUBPOP_DIR / "subpop_train.jsonl"
    if not train_path.exists():
        raise SystemExit(f"{train_path} is missing: run scripts/fetch_subpop.py first")
    train = load_subpop_records(train_path)
    items = prepare_items(pd.read_parquet(ITEMS_PATH))
    print(
        f"SubPOP-Train: {train['qkey'].nunique():,} questions; "
        f"SimBench: {len(items):,} items, {items['question'].nunique():,} distinct question texts"
    )

    matches = match_items(items, train, workers=args.workers)
    assert len(matches) == len(items), f"{len(items)} items in, {len(matches)} matches out"
    write_parquet_safely(matches, OVERLAP_PATH)
    print(f"wrote {OVERLAP_PATH}")

    counts = pd.crosstab([matches["split"], matches["dataset_name"]], matches["status"])
    counts = counts.reindex(columns=list(STATUSES), fill_value=0)
    print("\nItems by match status (datasets with any match):")
    print(counts[counts[list(STATUSES[:3])].sum(axis=1) > 0].to_string())

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    long = (
        matches.groupby(["split", "dataset_name", "status"])
        .agg(items=("item_id", "size"), distinct_questions=("question", "nunique"))
        .reset_index()
    )
    long.to_csv(TABLES_DIR / "overlap_counts.csv", index=False)

    if args.examples:
        distinct = matches.drop_duplicates("question")
        print("\nSample matches (SimBench question / closest SubPOP-Train question):")
        for low, high in [(100, 100), (95, 99), (90, 94), (80, 89), (70, 79)]:
            band = distinct[distinct["best_ratio"].between(low, high)]
            print(f"\n  ratio {low}-{high}: {len(band)} distinct SimBench questions")
            for row in band.sample(min(args.examples, len(band)), random_state=0).itertuples():
                print(f"    [{row.dataset_name}, {row.best_ratio}] {row.question[:130]!r}")
                print(f"      {row.best_train_question[:130]!r} ({row.best_train_qkey})")

    copies = training_copy(items, matches, train)
    if len(copies):
        print(
            f"\nGrouped OpinionQA items whose question, options and subgroup are all in "
            f"SubPOP-Train: {len(copies)}"
        )
        print(
            "  total variation between the training distribution and SimBench's: "
            f"mean {copies['tvd_training_vs_simbench'].mean():.3f}, "
            f"median {copies['tvd_training_vs_simbench'].median():.3f}"
        )
        print(
            "  SimBench score of returning the training distribution unchanged: "
            f"{copies['score_of_copying_training'].mean():.1f}"
        )
        summary = pd.DataFrame(
            [
                {
                    "items": len(copies),
                    "mean_tvd_training_vs_simbench": copies["tvd_training_vs_simbench"].mean(),
                    "median_tvd_training_vs_simbench": copies["tvd_training_vs_simbench"].median(),
                    "score_of_copying_training": copies["score_of_copying_training"].mean(),
                }
            ]
        )
        summary.round(4).to_csv(TABLES_DIR / "overlap_training_copy.csv", index=False)
    print(f"\nwrote aggregate tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
