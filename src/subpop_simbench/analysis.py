"""Base-versus-adapter analysis of finished runs.

Every comparison is paired: the same items under the same prompt condition, one model with
and without its SubPOP adapter.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from subpop_simbench.prompts import CONDITIONS
from subpop_simbench.results import load_run
from subpop_simbench.scoring import dataset_norms

# label -> (run-tag stem, display name). A pair is the runs <stem>__base and <stem>__subpop.
MODEL_PAIRS = {
    "mistral-7b": ("mistral-7b-v0.1", "Mistral-7B"),
    "llama-2-7b": ("llama-2-7b", "Llama-2-7B"),
    "llama-2-13b": ("llama-2-13b", "Llama-2-13B"),
    "llama-3-70b": ("llama-3-70b", "Llama-3-70B"),
}

# SimBench datasets that are attitude surveys of the general public, like SubPOP's training
# data. Everything else is psychometric scales, judgement tasks and decision tasks.
OTHER_OPINION_SURVEYS = frozenset(
    {"ESS", "ISSP", "GlobalOpinionQA", "Afrobarometer", "LatinoBarometro", "TISP"}
)
TIERS = ("OpinionQA", "other opinion surveys", "non-survey tasks")

# Temperatures tried by the calibration control. 64 is flat enough to score as uniform.
TEMPERATURES = np.round(np.geomspace(0.25, 64, 65), 4)


def tier_of(dataset_name: str) -> str:
    if dataset_name == "OpinionQA":
        return TIERS[0]
    return TIERS[1] if dataset_name in OTHER_OPINION_SURVEYS else TIERS[2]


def prepare_items(items: pd.DataFrame) -> pd.DataFrame:
    """Items with the score's per-dataset norm and the dataset tier attached."""
    out = items.copy()
    out["norm"] = out["dataset_name"].map(dataset_norms(items))
    out["tier"] = out["dataset_name"].map(tier_of)
    return out


def pad(dists, width: int) -> np.ndarray:
    """Stack distributions of different lengths into one zero-padded (n, width) array."""
    out = np.zeros((len(dists), width))
    for i, dist in enumerate(dists):
        out[i, : len(dist)] = dist
    return out


def scores_at(
    pred: np.ndarray, human: np.ndarray, norm: np.ndarray, temperature: float = 1.0
) -> np.ndarray:
    """Per-item SimBench score after raising predictions to the power 1/temperature.

    Padded options have probability zero and stay at zero, so they never take mass.
    """
    with np.errstate(divide="ignore"):
        logp = np.where(pred > 0, np.log(pred), -np.inf) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    q = np.exp(logp)
    q /= q.sum(axis=1, keepdims=True)
    return 100.0 * (1.0 - 0.5 * np.abs(human - q).sum(axis=1) / norm)


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval for the mean of x, resampling items."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    low, high = np.quantile(x[idx].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def best_temperature(
    pred: np.ndarray, human: np.ndarray, norm: np.ndarray, temperatures=TEMPERATURES
) -> tuple[float, float]:
    """(mean score, temperature) at the single temperature that maximises the mean score.

    Fitted on the items it is scored on, so it flatters whichever model it is applied to.
    """
    means = np.array([scores_at(pred, human, norm, t).mean() for t in temperatures])
    best = int(means.argmax())
    return float(means[best]), float(temperatures[best])


def best_temperature_per_group(
    pred: np.ndarray, human: np.ndarray, norm: np.ndarray, groups, temperatures=TEMPERATURES
) -> np.ndarray:
    """Per-item scores when every group (e.g. dataset) gets its own best temperature."""
    out = np.empty(len(pred))
    for idx in pd.Series(np.arange(len(pred))).groupby(np.asarray(groups)).indices.values():
        grids = [scores_at(pred[idx], human[idx], norm[idx], t) for t in temperatures]
        out[idx] = grids[int(np.argmax([g.mean() for g in grids]))]
    return out


def shapes_from_other_questions(items: pd.DataFrame, by: tuple[str, ...] = ()) -> np.ndarray:
    """The average human answer distribution of *other* questions, per item.

    Averages `human_dist` over items with the same number of options (and the same values of
    `by`, e.g. the dataset), leaving out every row of the item's own question. Nothing about
    the item being predicted is used, including in the Grouped split, where one question
    appears once per subgroup. Rows are NaN where there is no other question to average.
    """
    width = int(items["n_options"].max())
    human = pad(items["human_dist"], width)
    out = np.full_like(human, np.nan)
    question = (items["dataset_name"] + "\x1f" + items["question"]).to_numpy()
    for idx in items.groupby([*by, "n_options"]).indices.values():
        idx = np.asarray(idx)
        codes, _ = pd.factorize(question[idx])
        per_question = np.zeros((codes.max() + 1, width))
        np.add.at(per_question, codes, human[idx])
        others = len(idx) - np.bincount(codes)[codes]
        ok = others > 0
        out[idx[ok]] = (human[idx].sum(axis=0) - per_question[codes[ok]]) / others[ok, None]
    return out


# Predictors that never read the question. They use human answers to other questions, so no
# model gets them for free; they show how much score answer shape alone is worth.
QUESTION_BLIND = {
    "benchmark": "one answer shape per option count",
    "dataset": "each dataset's typical answer shape",
}


def question_blind_predictions(items: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-item predictions of the two question-blind baselines, in the row order of `items`.

    benchmark  the average answer shape of all other questions with that many options
    dataset    the same, restricted to other questions of the item's own dataset; falls back
               to `benchmark` where the dataset has no other question with that option count
    """
    n = items["n_options"].to_numpy()
    width = int(n.max())
    uniform = (np.arange(width) < n[:, None]) / n[:, None]
    benchmark = shapes_from_other_questions(items)
    benchmark = np.where(np.isnan(benchmark), uniform, benchmark)
    dataset = shapes_from_other_questions(items, by=("dataset_name",))
    dataset = np.where(np.isnan(dataset), benchmark, dataset)
    return {"benchmark": benchmark, "dataset": dataset}


def match_letter_usage(pred: np.ndarray, target: np.ndarray, n_options: np.ndarray) -> np.ndarray:
    """Shift a model's average use of each option letter to `target`.

    Within each option count, predictions are divided by the model's own average probability
    for each letter and multiplied by the target share for that letter, then renormalised.
    This changes which letters the model favours on average. It uses nothing about the
    question, so it cannot add knowledge of how people answer a particular item.
    """
    out = np.zeros_like(pred)
    for k in np.unique(n_options):
        rows = n_options == k
        own = pred[rows].mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[rows] = np.where(own > 0, pred[rows] / own, 0.0) * target[rows]
    return out / out.sum(axis=1, keepdims=True)


# SimBench's human answers for MoralMachine look reversed (see the README). The helpers below
# check the published answers against common sense and measure what reversing them does.
MORAL_MACHINE = "MoralMachine"
_ANIMALS = frozenset({"cat", "cats", "dog", "dogs"})


def swap_first_two_options(human: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """A copy of `human` with the shares of options A and B exchanged on `rows`."""
    out = human.copy()
    out[rows, :2] = human[rows, 1::-1]
    return out


def casualties(option_text: str) -> list[str]:
    """Who dies under one MoralMachine option, e.g. ["elderly man", "cat"]."""
    names = re.findall(r"^\s*\*\s*\d+\s+(.+)$", option_text, flags=re.MULTILINE)
    return [name.strip().lower() for name in names]


def animals_versus_people(items: pd.DataFrame) -> pd.DataFrame:
    """MoralMachine items where one outcome kills only animals and the other only people.

    `share_killing_people` is the published share of respondents choosing the outcome in
    which the people die. Moral Machine respondents strongly prefer sparing people over pets
    (Awad et al., 2018), so values near 1 mean the published answers are the wrong way round.
    """
    rows = []
    for item in items[items["dataset_name"] == MORAL_MACHINE].itertuples():
        dead = [casualties(text) for text in item.option_texts]
        if len(dead) != 2 or not all(dead):
            continue
        only_animals = [all(name in _ANIMALS for name in d) for d in dead]
        no_animals = [not any(name in _ANIMALS for name in d) for d in dead]
        for people, animals in ((0, 1), (1, 0)):
            if no_animals[people] and only_animals[animals]:
                rows.append(
                    {
                        "item_id": item.item_id,
                        "people_die_under_option": "AB"[people],
                        "share_killing_people": float(item.human_dist[people]),
                    }
                )
    columns = ["item_id", "people_die_under_option", "share_killing_people"]
    return pd.DataFrame(rows, columns=columns)


@dataclass
class Pair:
    """One model's base and adapter runs under one condition, aligned row for row."""

    frame: pd.DataFrame  # the item columns, in the same row order as the arrays
    human: np.ndarray
    norm: np.ndarray
    pred_base: np.ndarray
    pred_adapter: np.ndarray
    mass_base: np.ndarray
    mass_adapter: np.ndarray

    @property
    def score_base(self) -> np.ndarray:
        return scores_at(self.pred_base, self.human, self.norm)

    @property
    def score_adapter(self) -> np.ndarray:
        return scores_at(self.pred_adapter, self.human, self.norm)


def load_pairs(items: pd.DataFrame, results_dir: Path, stem: str) -> dict[str, Pair] | None:
    """Pairs per condition for one model, or None unless both runs are complete on disk.

    `items` must come from `prepare_items`.
    """
    dirs = {kind: Path(results_dir) / f"{stem}__{kind}" for kind in ("base", "subpop")}
    if not all((d / "predictions.parquet").exists() for d in dirs.values()):
        return None
    runs = {kind: load_run(d) for kind, d in dirs.items()}
    width = int(items["n_options"].max())
    pairs = {}
    for condition in CONDITIONS:
        aligned = {}
        for kind, run in runs.items():
            rows = run.loc[run["condition"] == condition, ["item_id", "pred", "answer_mass"]]
            merged = rows.merge(items, on="item_id", how="left", validate="one_to_one")
            assert merged["norm"].notna().all(), f"{stem}: predictions for unknown items"
            aligned[kind] = merged.sort_values("item_id").reset_index(drop=True)
        base, adapter = aligned["base"], aligned["subpop"]
        if base.empty and adapter.empty:
            continue
        assert base["item_id"].equals(adapter["item_id"]), (
            f"{stem} {condition}: base and adapter runs cover different items"
        )
        pairs[condition] = Pair(
            frame=base.drop(columns=["pred", "answer_mass"]),
            human=pad(base["human_dist"], width),
            norm=base["norm"].to_numpy(),
            pred_base=pad(base["pred"], width),
            pred_adapter=pad(adapter["pred"], width),
            mass_base=base["answer_mass"].to_numpy(),
            mass_adapter=adapter["answer_mass"].to_numpy(),
        )
    return pairs
