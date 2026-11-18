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
    moralmachine_reversed.csv    scores if MoralMachine's human answers, which look reversed
                                 in SimBench's data, were the other way round
    moralmachine_animals_vs_people.csv
                                 the published answers to MoralMachine items that pit only
                                 animals against only people
    retrain.csv                  adapters trained here beside the released one: score
                                 difference with a paired bootstrap interval, and how close
                                 the predictions are item by item
    retrain_epochs.csv           training and validation loss per epoch, if the training
                                 run is under data/adapters/
    retrain_validation_reference.csv
                                 what predictors that need no model lose on the training
                                 and validation questions, if the training examples are on
                                 disk

Every per-model table also carries `question_blind`: the score, on the same items, of
predicting each dataset's typical answer shape without reading the question.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from subpop_simbench.analysis import (
    MODEL_PAIRS,
    MORAL_MACHINE,
    QUESTION_BLIND,
    RELEASED,
    RETRAINED,
    TIERS,
    animals_versus_people,
    best_temperature,
    best_temperature_per_group,
    bootstrap_ci,
    compare_to_reference,
    load_pairs,
    match_letter_usage,
    pad,
    prepare_items,
    question_blind_predictions,
    scores_at,
    swap_first_two_options,
)
from subpop_simbench.io import (
    ADAPTERS_DIR,
    ITEMS_PATH,
    OVERLAP_PATH,
    RESULTS_DIR,
    TABLES_DIR,
    TRAIN_EXAMPLES_PATH,
)
from subpop_simbench.overlap import STATUSES
from subpop_simbench.train_data import reference_losses

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


def _moralmachine_reversed(pair, **keys) -> list[dict]:
    """Scores with MoralMachine's two answers exchanged, on every slice that contains it.

    The score's norm is each dataset's mean distance from uniform, which the exchange leaves
    unchanged for two-option items.
    """
    frame = pair.frame
    reversed_rows = (frame["dataset_name"] == MORAL_MACHINE).to_numpy()
    if not reversed_rows.any():
        return []
    assert (frame.loc[reversed_rows, "n_options"] == 2).all(), "MoralMachine is not binary"
    human = swap_first_two_options(pair.human, reversed_rows)
    base = scores_at(pair.pred_base, human, pair.norm)
    adapter = scores_at(pair.pred_adapter, human, pair.norm)
    masks = {
        MORAL_MACHINE: reversed_rows,
        TIERS[2]: (frame["tier"] == TIERS[2]).to_numpy(),
        "all items": np.ones(len(frame), dtype=bool),
    }
    return [
        {
            **keys,
            "subset": subset,
            "n": int(mask.sum()),
            "base": pair.score_base[mask].mean(),
            "base_reversed": base[mask].mean(),
            "adapter": pair.score_adapter[mask].mean(),
            "adapter_reversed": adapter[mask].mean(),
            "gain": (pair.score_adapter - pair.score_base)[mask].mean(),
            "gain_reversed": (adapter - base)[mask].mean(),
            "question_blind": frame["blind_dataset"].to_numpy()[mask].mean(),
            "question_blind_reversed": frame["blind_dataset_reversed"].to_numpy()[mask].mean(),
        }
        for subset, mask in masks.items()
    ]


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


def _epoch_scored(run_dir: Path) -> int | None:
    """The training epoch a SimBench run scored, from the adapter path in its metadata."""
    adapter = json.loads((run_dir / "meta.json").read_text())["identity"]["model"]["adapter"]
    match = re.fullmatch(r"epoch-(\d+)", Path(adapter or "").name)
    return int(match.group(1)) if match else None


def _retrained(items: pd.DataFrame, have_overlap: bool) -> tuple[list[dict], list[dict]]:
    """Rows for retrain.csv and retrain_epochs.csv.

    Each retrained checkpoint is set beside the released adapter, and each checkpoint after
    the first beside the first, on the same slices as by_subset.csv.
    """
    comparisons, epochs = [], []
    for model, spec in RETRAINED.items():
        stem, name = MODEL_PAIRS[model]
        runs = {RELEASED: load_pairs(items, RESULTS_DIR, stem)}
        for run, tag in spec["runs"].items():
            runs[run] = load_pairs(items, RESULTS_DIR, stem, adapter_tag=tag)
        if any(pairs is None for pairs in runs.values()):
            print(f"[{name}: retrained runs not all on disk, skipped]")
            continue
        first = next(iter(spec["runs"]))
        for run, tag in spec["runs"].items():
            for reference in [RELEASED] + ([first] if run != first else []):
                for condition, pair in runs[run].items():
                    other = runs[reference][condition]
                    assert pair.frame["item_id"].equals(other.frame["item_id"])
                    whole = len(pair.frame)
                    for subset, mask in _subset_masks(pair.frame, have_overlap).items():
                        if subset != "all items" and mask.sum() == whole:
                            continue  # the same items under another name
                        comparisons.append(
                            {
                                "model": model,
                                "model_name": name,
                                "condition": condition,
                                "subset": subset,
                                "n": int(mask.sum()),
                                "run": run,
                                "epoch": _epoch_scored(RESULTS_DIR / tag),
                                "reference": reference,
                                "base": pair.score_base[mask].mean(),
                                **compare_to_reference(
                                    pair.score_adapter[mask],
                                    pair.pred_adapter[mask],
                                    other.score_adapter[mask],
                                    other.pred_adapter[mask],
                                    pair.pred_base[mask],
                                ),
                            }
                        )
        run_dir = ADAPTERS_DIR / spec["training_tag"]
        if (run_dir / "metrics.json").exists():
            best = json.loads((run_dir / "meta.json").read_text()).get("best_epoch")
            for row in json.loads((run_dir / "metrics.json").read_text()):
                if not row["partial"]:
                    epochs.append(
                        {
                            "model": model,
                            "model_name": name,
                            "epoch": row["epoch"],
                            "train_loss": row["train_loss"],
                            "val_loss": row["val_loss"],
                            "lowest_val_loss": row["epoch"] == best,
                            "train_minutes": row["train_seconds"] / 60,
                        }
                    )
        else:
            print(f"[{run_dir.name}: training run not on disk, retrain_epochs.csv left as is]")
    return comparisons, epochs


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
    # The dataset-shape baseline again, as if MoralMachine's answers were the other way round.
    is_moralmachine = items["dataset_name"] == MORAL_MACHINE
    pairs_of_dist_and_flag = zip(items["human_dist"], is_moralmachine, strict=True)
    flipped = items.assign(human_dist=[d[::-1] if m else d for d, m in pairs_of_dist_and_flag])
    items["blind_dataset_reversed"] = scores_at(
        question_blind_predictions(flipped)["dataset"], pad(flipped["human_dist"], width), norm
    )
    position = pd.Series(np.arange(len(items)), index=items["item_id"])
    blind_rows = [
        {"baseline": key, "description": QUESTION_BLIND[key], "subset": subset,
         "n": int(mask.sum()), "score": items[f"blind_{key}"].to_numpy()[mask].mean()}
        for key in blind
        for subset, mask in _subset_masks(items, have_overlap).items()
    ]  # fmt: skip

    animals = animals_versus_people(items)
    overall, subsets, datasets, controls, subgroups, overlap = [], [], [], [], [], []
    moralmachine = []
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
            moralmachine += _moralmachine_reversed(pair, **keys)

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

    retrain, retrain_epochs = _retrained(items, have_overlap)
    validation_reference = []
    if TRAIN_EXAMPLES_PATH.exists():
        examples = pd.read_parquet(
            TRAIN_EXAMPLES_PATH, columns=["split", "qkey", "target", "n_options"]
        )
        validation_reference = reference_losses(examples).to_dict("records")
    else:
        print(
            f"[{TRAIN_EXAMPLES_PATH.name} not found: retrain_validation_reference.csv left as is]"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "overall": overall,
        "by_subset": subsets,
        "by_dataset": datasets,
        "question_blind": blind_rows,
        "calibration_controls": controls,
        "condition_c_by_subgroup": subgroups,
        "overlap_opinionqa": overlap,
        "moralmachine_reversed": moralmachine,
        "moralmachine_animals_vs_people": animals.to_dict("records"),
        "retrain": retrain,
        "retrain_epochs": retrain_epochs,
        "retrain_validation_reference": validation_reference,
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
    if len(animals):
        killed = int((animals["share_killing_people"] > 0.5).sum())
        print(
            f"\nMoralMachine, animals against people: the published majority chooses the "
            f"outcome that kills the people in {killed} of {len(animals)} items"
        )
    if "moralmachine_reversed" in frames:
        print("\nScores with MoralMachine's answers reversed")
        print(frames["moralmachine_reversed"].drop(columns="model").round(1).to_string(index=False))
    if "retrain" in frames:
        print("\nRetrained adapters beside a reference run of the same model")
        shown = frames["retrain"][frames["retrain"]["subset"] == "all items"]
        print(shown.drop(columns=["model", "subset"]).round(3).to_string(index=False))
    if "retrain_validation_reference" in frames:
        print("\nLoss of predictors that need no model")
        print(
            frames["retrain_validation_reference"]
            .pivot_table(index="predictor", columns="split", values="loss", sort=False)
            .round(4)
            .to_string()
        )
    print(f"\nwrote {len(frames)} tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
