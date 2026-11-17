"""Stage 1: data/raw/*.csv -> data/items.parquet + data/prompts.parquet.

items.parquet    one row per SimBench test case (both splits), parsed and normalised
prompts.parquet  one row per (item, prompt condition)
"""

from subpop_simbench.data import ITEM_COLUMNS, load_items
from subpop_simbench.io import ITEMS_PATH, PROMPTS_PATH, RAW_DIR, write_parquet_safely
from subpop_simbench.prompts import CONDITIONS, PROMPT_COLUMNS, build_prompts


def main() -> None:
    items = load_items(RAW_DIR)
    assert list(items.columns) == ITEM_COLUMNS
    print(f"items: {len(items):,} rows")
    print(items.groupby("split", sort=False).size().to_string(header=False))

    prompts = build_prompts(items)
    assert list(prompts.columns) == PROMPT_COLUMNS
    counts = prompts.groupby("condition").size()
    for condition in CONDITIONS:
        n = int(counts.get(condition, 0))
        dropped = len(items) - n
        reason = (
            "no SubPOP steering template: not an OpinionQA Grouped item with one attribute"
            if condition == "subpop_exact"
            else "none expected"
        )
        print(
            f"{condition}: {len(items):,} items in -> {n:,} prompts ({dropped:,} dropped: {reason})"
        )
        if condition != "subpop_exact":
            assert dropped == 0, f"{condition} lost {dropped} items"

    exact = items[items["subpop_attribute"].notna()]
    print(
        f"subpop_exact: {int(exact['subpop_group_trained'].sum()):,} items in subgroups SubPOP "
        f"trained on, {int((~exact['subpop_group_trained'].astype(bool)).sum()):,} in unseen ones"
    )

    write_parquet_safely(items, ITEMS_PATH)
    write_parquet_safely(prompts, PROMPTS_PATH)
    print(
        f"wrote {ITEMS_PATH.relative_to(ITEMS_PATH.parents[1])}, "
        f"{PROMPTS_PATH.relative_to(PROMPTS_PATH.parents[1])}"
    )


if __name__ == "__main__":
    main()
