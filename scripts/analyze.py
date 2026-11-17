"""Stage 4: adapter-versus-base tables for every model pair with both runs on disk.

    uv run python scripts/analyze.py

Writes CSVs to reports/tables/:

    overall.csv                  score per model and prompt condition, gain with a paired
                                 bootstrap interval, answer mass
    by_subset.csv                the same, on slices: split, dataset tier, and (if
                                 scripts/overlap.py has run) items with no matching
                                 training question
    by_dataset.csv               per SimBench dataset and split
    temperature_control.csv      how much of the gain a temperature on the base model's
                                 answers reproduces
    condition_c_by_subgroup.csv  SubPOP's own format: subgroups it trained on vs unseen
    overlap_opinionqa.csv        OpinionQA scores by training-question match status
"""

import pandas as pd

from subpop_simbench.analysis import (
    MODEL_PAIRS,
    TIERS,
    best_temperature,
    best_temperature_per_group,
    bootstrap_ci,
    load_pairs,
    prepare_items,
)
from subpop_simbench.io import ITEMS_PATH, OVERLAP_PATH, RESULTS_DIR, TABLES_DIR
from subpop_simbench.overlap import STATUSES

NO_MATCH = "no matching training question"


def _row(pair, mask, **keys) -> dict:
    base, adapter = pair.score_base[mask], pair.score_adapter[mask]
    return {
        **keys,
        "n": int(mask.sum()),
        "base": base.mean(),
        "adapter": adapter.mean(),
        "gain": (adapter - base).mean(),
    }


def main() -> None:
    pd.set_option("display.width", 220)
    items = prepare_items(pd.read_parquet(ITEMS_PATH))
    have_overlap = OVERLAP_PATH.exists()
    if have_overlap:
        status = pd.read_parquet(OVERLAP_PATH, columns=["item_id", "status"])
        items = items.merge(status, on="item_id", how="left", validate="one_to_one")
        assert items["status"].notna().all(), "overlap table does not cover every item"
    else:
        print(f"[{OVERLAP_PATH.name} not found: skipping the overlap tables]")

    overall, subsets, datasets, temperatures, subgroups, overlap = [], [], [], [], [], []
    for model, (stem, name) in MODEL_PAIRS.items():
        pairs = load_pairs(items, RESULTS_DIR, stem)
        if pairs is None:
            print(f"[{name}: runs not both on disk, skipped]")
            continue
        for condition, pair in pairs.items():
            keys = {"model": model, "model_name": name, "condition": condition}
            frame = pair.frame
            everything = frame["item_id"].notna().to_numpy()
            gain = pair.score_adapter - pair.score_base
            low, high = bootstrap_ci(gain)
            overall.append(
                {
                    **_row(pair, everything, **keys),
                    "gain_ci_low": low,
                    "gain_ci_high": high,
                    "share_improved": (gain > 0).mean(),
                    "answer_mass_base": pair.mass_base.mean(),
                    "answer_mass_adapter": pair.mass_adapter.mean(),
                }
            )

            masks = {"all items": everything}
            for split in ("pop", "grouped"):
                masks[f"{split} split"] = (frame["split"] == split).to_numpy()
            for tier in TIERS:
                masks[tier] = (frame["tier"] == tier).to_numpy()
            masks["all but OpinionQA"] = (frame["dataset_name"] != "OpinionQA").to_numpy()
            if have_overlap:
                masks[NO_MATCH] = (frame["status"] == STATUSES[3]).to_numpy()
            subsets += [_row(pair, m, **keys, subset=s) for s, m in masks.items() if m.any()]

            for (split, dataset), idx in frame.groupby(["split", "dataset_name"]).indices.items():
                mask = frame.index.isin(idx)
                datasets.append(_row(pair, mask, **keys, split=split, dataset_name=dataset))

            if have_overlap:
                opinionqa = (frame["dataset_name"] == "OpinionQA").to_numpy()
                for split in ("pop", "grouped"):
                    for s in STATUSES:
                        mask = opinionqa & ((frame["split"] == split) & (frame["status"] == s))
                        if mask.any():
                            overlap.append(
                                _row(pair, mask.to_numpy(), **keys, split=split, status=s)
                            )

            if condition == "subpop_exact":
                for trained in (True, False):
                    mask = (frame["subpop_group_trained"] == trained).to_numpy()
                    subgroups.append(_row(pair, mask, **keys, subgroup_trained=trained))
                continue

            args = (pair.human, pair.norm)
            base_best, base_t = best_temperature(pair.pred_base, *args)
            adapter_best, adapter_t = best_temperature(pair.pred_adapter, *args)
            groups = frame["dataset_name"]
            temperatures.append(
                {
                    **keys,
                    "base": pair.score_base.mean(),
                    "adapter": pair.score_adapter.mean(),
                    "base_at_best_temperature": base_best,
                    "base_best_temperature": base_t,
                    "adapter_at_best_temperature": adapter_best,
                    "adapter_best_temperature": adapter_t,
                    "base_temperature_per_dataset": best_temperature_per_group(
                        pair.pred_base, *args, groups
                    ).mean(),
                    "adapter_temperature_per_dataset": best_temperature_per_group(
                        pair.pred_adapter, *args, groups
                    ).mean(),
                }
            )

    if not overall:
        raise SystemExit(f"no complete base and adapter pair under {RESULTS_DIR}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "overall": overall,
        "by_subset": subsets,
        "by_dataset": datasets,
        "temperature_control": temperatures,
        "condition_c_by_subgroup": subgroups,
        "overlap_opinionqa": overlap,
    }
    frames = {}
    for filename, rows in tables.items():
        if rows:
            frames[filename] = pd.DataFrame(rows)
            frames[filename].round(4).to_csv(TABLES_DIR / f"{filename}.csv", index=False)

    print("\nOverall, base -> adapter (gain with a paired-bootstrap 95% interval)")
    print(frames["overall"].drop(columns="model").round(2).to_string(index=False))
    wide = frames["by_subset"].pivot_table(
        index="subset", columns=["model_name", "condition"], values="adapter", sort=False
    )
    print("\nAdapter score by subset")
    print(wide.round(1).to_string())
    print("\nTemperature control")
    print(frames["temperature_control"].drop(columns="model").round(2).to_string(index=False))
    print(f"\nwrote {len(frames)} tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
