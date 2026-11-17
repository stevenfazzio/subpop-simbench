"""The SimBench score.

    S_i = 100 * (1 - TVD(human_i, model_i) / norm(dataset of i))

where `norm` is the mean, over a dataset's items, of TVD(human, uniform). 100 is a perfect
match, 0 is as good as answering uniformly, and negative is worse than uniform. A model's
score is the plain mean of S_i over its items.

This follows `calculate_simbench_score.py` in the SimBench release, which computes the norm
per `dataset_name` over whatever rows it is handed. `scripts/check_scoring_vs_upstream.py`
confirms the two agree on identical inputs.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _normalise(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    total = p.sum()
    return p / total if total > 0 else p


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    """Total variation distance between two distributions over the same options."""
    p, q = _normalise(p), _normalise(q)
    if p.shape != q.shape or p.size == 0:
        raise ValueError(f"distributions differ in length: {p.size} vs {q.size}")
    return float(0.5 * np.abs(p - q).sum())


def tv_to_uniform(human: Sequence[float]) -> float:
    """TVD between the human distribution and uniform. NaN for a single-option item."""
    n = len(human)
    return total_variation(human, np.full(n, 1.0 / n)) if n > 1 else float("nan")


def dataset_norms(items: pd.DataFrame, by: Sequence[str] = ("dataset_name",)) -> pd.Series:
    """Mean TVD(human, uniform) per group of items: the score's denominator."""
    tvu = items["human_dist"].map(tv_to_uniform)
    return tvu.groupby([items[k] for k in by]).mean().rename("norm")


def score_predictions(
    items: pd.DataFrame,
    preds: pd.DataFrame,
    pred_col: str = "pred",
    norm_by: Sequence[str] = ("dataset_name",),
    norm_items: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach `tvd` and `score` to each prediction.

    preds       one row per scored prediction, with `item_id` and a distribution in
                `pred_col`. Several rows may share an item (different models or conditions).
    norm_items  the items the norm is averaged over. Defaults to all of `items`, so an
                item's score does not depend on which subset happens to be scored with it.
    """
    by = list(norm_by)
    norms = dataset_norms(items if norm_items is None else norm_items, by)
    item_cols = ["item_id", "human_dist", *[c for c in by if c not in preds.columns]]
    out = preds.merge(items[item_cols], on="item_id", how="left", validate="many_to_one")
    missing = out["human_dist"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} predictions have an item_id not in items")
    out = out.merge(norms.reset_index(), on=by, how="left", validate="many_to_one")
    if out["norm"].isna().any() or (out["norm"] <= 0).any():
        raise ValueError("some predictions fall in a group with no usable norm")
    out["tvd"] = [
        total_variation(h, p) for h, p in zip(out["human_dist"], out[pred_col], strict=True)
    ]
    out["score"] = 100.0 * (1.0 - out["tvd"] / out["norm"])
    assert len(out) == len(preds), f"{len(preds)} predictions in, {len(out)} out"
    return out.drop(columns=["human_dist"])
