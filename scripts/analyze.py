"""Stage 4: adapter-versus-base tables for every model pair with both runs on disk.

    uv run python scripts/analyze.py

Writes CSVs to reports/tables/:

    overall.csv                  score per model and prompt condition, gain with a paired
                                 bootstrap interval, answer mass
    by_subset.csv                the same, on slices: split, dataset tier, and (if
                                 scripts/overlap.py has run) items with no matching
                                 training question
    by_dataset.csv               per SimBench dataset and split
    question_blind.csv           what predictors that never read the question score
    calibration_controls.csv     how much of the gain generic recalibration of the base
                                 model reproduces: a temperature, and its letter usage
                                 shifted to the human average
    condition_c_by_subgroup.csv  SubPOP's own format: subgroups it trained on vs unseen
    overlap_opinionqa.csv        OpinionQA scores by training-question match status

Every per-model table also carries `question_blind`: the score, on the same items, of
predicting each dataset's typical answer shape without reading the question.
"""

import numpy as np
import pandas as pd

from subpop_simbench.analysis import (
    MODEL_PAIRS,
    QUESTION_BLIND,
    TIERS,
    best_temperature,
    best_temperature_per_group,
    bootstrap_ci,
    load_pairs,
    match_letter_usage,
    pad,
    prepare_items,
    question_blind_predictions,
    scores_at,
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
        "question_blind": pair.frame["blind_dataset"].to_numpy()[mask].mean(),
    }


def _subset_masks(frame: pd.DataFrame, have_overlap: bool) -> dict[str, np.ndarray]:
    masks = {"all items": np.ones(len(frame), dtype=bool)}
    for split in ("pop", "grouped"):
        masks[f"{split} split"] = (frame["split"] == split).to_numpy()
    for tier in TIERS:
        masks[tier] = (frame["tier"] == tier).to_numpy()
    masks["all but OpinionQA"] = (frame["dataset_name"] != "OpinionQA").to_numpy()
    if have_overlap:
        masks[NO_MATCH] = (frame["status"] == STATUSES[3]).to_numpy()
    return {name: mask for name, mask in masks.items() if mask.any()}


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

    # Question-blind baselines: one score per item, the same for every model.
    width = int(items["n_options"].max())
    human, norm = pad(items["human_dist"], width), items["norm"].to_numpy()
    blind = question_blind_predictions(items)
    for key, pred in blind.items():
        items[f"blind_{key}"] = scores_at(pred, human, norm)
    position = pd.Series(np.arange(len(items)), index=items["item_id"])
    blind_rows = [
        {"baseline": key, "description": QUESTION_BLIND[key], "subset": subset,
         "n": int(mask.sum()), "score": items[f"blind_{key}"].to_numpy()[mask].mean()}
        for key in blind
        for subset, mask in _subset_masks(items, have_overlap).items()
    ]  # fmt: skip

    overall, subsets, datasets, controls, subgroups, overlap = [], [], [], [], [], []
    for model, (stem, name) in MODEL_PAIRS.items():
        pairs = load_pairs(items, RESULTS_DIR, stem)
        if pairs is None:
            print(f"[{name}: runs not both on disk, skipped]")
            continue
        for condition, pair in pairs.items():
            keys = {"model": model, "model_name": name, "condition": condition}
            frame = pair.frame
            everything = np.ones(len(frame), dtype=bool)
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

            masks = _subset_masks(frame, have_overlap)
            subsets += [_row(pair, m, **keys, subset=s) for s, m in masks.items()]

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
            # The human average share of each letter, from other questions only.
            shape = blind["benchmark"][position.loc[frame["item_id"]].to_numpy()]
            n_options = frame["n_options"].to_numpy()
            base_matched, _ = best_temperature(
                match_letter_usage(pair.pred_base, shape, n_options), *args
            )
            adapter_matched, _ = best_temperature(
                match_letter_usage(pair.pred_adapter, shape, n_options), *args
            )
            controls.append(
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
                    "base_letter_usage_matched": base_matched,
                    "adapter_letter_usage_matched": adapter_matched,
                    "base_share_on_first_option": pair.pred_base[:, 0].mean(),
                    "adapter_share_on_first_option": pair.pred_adapter[:, 0].mean(),
                    "human_share_on_first_option": pair.human[:, 0].mean(),
                }
            )

    if not overall:
        raise SystemExit(f"no complete base and adapter pair under {RESULTS_DIR}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "overall": overall,
        "by_subset": subsets,
        "by_dataset": datasets,
        "question_blind": blind_rows,
        "calibration_controls": controls,
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
    print("\nQuestion-blind baselines")
    print(
        frames["question_blind"]
        .pivot_table(index="subset", columns="description", values="score", sort=False)
        .round(1)
        .to_string()
    )
    print("\nCalibration controls")
    print(frames["calibration_controls"].drop(columns="model").round(2).to_string(index=False))
    print(f"\nwrote {len(frames)} tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
