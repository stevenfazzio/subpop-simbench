"""Do SubPOP's training questions appear among SimBench's items?

SubPOP-Train is Pew American Trends Panel questions. SubPOP left out the survey waves that
OpinionQA uses, but Pew repeats questions across waves, so the same question can sit in
both. Matching is by question text: an edit-distance ratio from 0 to 100 between normalised
strings, against the closest of SubPOP's training questions.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

STATUSES = (
    "same question and options",
    "same question text only",
    "similar wording",
    "no match",
)
# Read from samples at each level: 95 and above were the same question; 80 to 94 were mostly
# another item of the same battery; below 80 looked unrelated.
SAME_TEXT_RATIO = 95
SIMILAR_RATIO = 80

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def norm_text(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace."""
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower().translate(_QUOTES))).strip()


def norm_options(options) -> frozenset[str]:
    """The answer options as a set, ignoring order, wording details and 'Refused'."""
    return frozenset(o for o in (norm_text(x) for x in options) if o and o != "refused")


def load_subpop_records(path: Path) -> pd.DataFrame:
    """SubPOP's JSON-lines file: one row per (question, subgroup)."""
    with open(path) as f:
        return pd.DataFrame(json.loads(line) for line in f)


def match_items(items: pd.DataFrame, train: pd.DataFrame, workers: int = 2) -> pd.DataFrame:
    """One row per SimBench item with its closest SubPOP-Train question.

    items  the SimBench item table (`item_id`, `question`, `option_texts`).
    train  SubPOP-Train records (`qkey`, `question`, `options`); subgroup rows are collapsed.
    """
    train = train.drop_duplicates("qkey")[["qkey", "question", "options"]].copy()
    train["q_norm"] = train["question"].map(norm_text)
    train["o_norm"] = train["options"].map(norm_options)
    options_by_question = train.groupby("q_norm")["o_norm"].agg(set).to_dict()
    train_q = train.drop_duplicates("q_norm").reset_index(drop=True)

    out = items[["item_id", "split", "dataset_name", "question"]].copy()
    out["q_norm"] = items["question"].map(norm_text).to_numpy()
    o_norm = items["option_texts"].map(norm_options).to_numpy()
    unique = out.drop_duplicates("q_norm")[["q_norm"]].reset_index(drop=True)

    ratios = process.cdist(
        unique["q_norm"].tolist(),
        train_q["q_norm"].tolist(),
        scorer=fuzz.ratio,
        dtype=np.uint8,
        workers=workers,
    )
    best = ratios.argmax(axis=1)
    unique["best_ratio"] = ratios.max(axis=1)
    for column in ("qkey", "question", "q_norm"):
        unique[f"best_train_{column}"] = train_q[column].to_numpy()[best]

    n_items = len(out)
    out = out.merge(unique, on="q_norm", how="left", validate="many_to_one")
    assert len(out) == n_items and out["best_ratio"].notna().all()
    out["exact_question"] = out["q_norm"] == out["best_train_q_norm"]
    out["exact_question_and_options"] = [
        exact and options in options_by_question[question]
        for exact, options, question in zip(
            out["exact_question"], o_norm, out["q_norm"], strict=True
        )
    ]
    out["status"] = match_status(out)
    return out.drop(columns=["q_norm", "best_train_q_norm"])


def match_status(matches: pd.DataFrame) -> pd.Series:
    """Bucket each item by how closely its question matches a training question."""
    return pd.Series(
        np.select(
            [
                matches["exact_question_and_options"],
                matches["best_ratio"] >= SAME_TEXT_RATIO,
                matches["best_ratio"] >= SIMILAR_RATIO,
            ],
            STATUSES[:3],
            default=STATUSES[3],
        ),
        index=matches.index,
    )


def training_copy(items: pd.DataFrame, matches: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """How close is SubPOP's training target to SimBench's answer where a question repeats?

    For Grouped OpinionQA items whose question and options are both in SubPOP-Train, and
    whose subgroup SubPOP trained on, line up the training distribution for that question
    and subgroup (averaged over waves if it was asked more than once) with SimBench's human
    distribution. Returns one row per item that can be lined up.
    """
    train = train.assign(q_norm=train["question"].map(norm_text))
    by_key = dict(tuple(train.groupby(["q_norm", "attribute", "group"])))
    exact = matches.loc[matches["exact_question_and_options"], "item_id"]
    candidates = items[
        (items["dataset_name"] == "OpinionQA")
        & (items["split"] == "grouped")
        & items["subpop_group"].notna()
        & items["item_id"].isin(exact)
    ]
    rows = []
    for item in candidates.itertuples():
        records = by_key.get((norm_text(item.question), item.subpop_attribute, item.subpop_group))
        if records is None:
            continue
        texts = [norm_text(t) for t in item.option_texts]
        copies = []
        for record in records.itertuples():
            answered = [o for o in map(norm_text, record.options) if o != "refused"]
            if len(answered) != len(record.responses):
                continue
            # SubPOP stores the distribution over substantive answers and the refusal
            # rate separately; SimBench keeps "Refused" as an option.
            full = dict(
                zip(answered, np.asarray(record.responses) * (1 - record.refusal_rate), strict=True)
            )
            full["refused"] = record.refusal_rate
            if set(texts) <= set(full):
                copies.append([full[t] for t in texts])
        if not copies:
            continue
        copy = np.mean(copies, axis=0)
        copy = copy / copy.sum()
        human = np.asarray(item.human_dist, dtype=float)
        tvd = float(0.5 * np.abs(human - copy).sum())
        rows.append(
            {
                "item_id": item.item_id,
                "n_training_waves": len(copies),
                "tvd_training_vs_simbench": tvd,
                "score_of_copying_training": 100.0 * (1.0 - tvd / item.norm),
            }
        )
    return pd.DataFrame(rows)
