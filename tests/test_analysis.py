import numpy as np
import pandas as pd
import pytest

from subpop_simbench.analysis import (
    TIERS,
    best_temperature,
    best_temperature_per_group,
    bootstrap_ci,
    pad,
    prepare_items,
    scores_at,
    tier_of,
)
from subpop_simbench.scoring import score_predictions


@pytest.fixture
def items():
    return prepare_items(
        pd.DataFrame(
            {
                "item_id": ["a", "b", "c", "d"],
                "dataset_name": ["OpinionQA", "OpinionQA", "ESS", "Jester"],
                "human_dist": [[0.5, 0.5], [0.9, 0.1], [0.7, 0.2, 0.1], [0.6, 0.4]],
                "n_options": [2, 2, 3, 2],
            }
        )
    )


def test_tiers():
    assert [tier_of(d) for d in ("OpinionQA", "ESS", "Jester")] == list(TIERS)


def test_prepare_items_attaches_norm_and_tier(items):
    assert items["tier"].tolist() == [TIERS[0], TIERS[0], TIERS[1], TIERS[2]]
    assert items.loc[0, "norm"] == pytest.approx(0.2)  # mean of TVD-to-uniform 0.0 and 0.4


def test_pad_keeps_ragged_rows():
    out = pad([[0.5, 0.5], [0.2, 0.3, 0.5]], 4)
    assert out.shape == (2, 4)
    assert out[0].tolist() == [0.5, 0.5, 0.0, 0.0]


def test_scores_match_the_scorer_at_temperature_one(items):
    preds = [[0.8, 0.2], [0.6, 0.4], [0.5, 0.3, 0.2], [0.1, 0.9]]
    expected = score_predictions(items, items[["item_id"]].assign(pred=preds))["score"]
    got = scores_at(pad(preds, 3), pad(items["human_dist"], 3), items["norm"].to_numpy())
    assert got.tolist() == pytest.approx(expected.tolist())


def test_a_flat_temperature_scores_like_uniform(items):
    preds = pad([[0.99, 0.01], [0.01, 0.99], [0.98, 0.01, 0.01], [0.01, 0.99]], 3)
    human, norm = pad(items["human_dist"], 3), items["norm"].to_numpy()
    flat = scores_at(preds, human, norm, temperature=1e9)
    uniform = scores_at(pad([[0.5, 0.5], [0.5, 0.5], [1 / 3] * 3, [0.5, 0.5]], 3), human, norm)
    assert flat.tolist() == pytest.approx(uniform.tolist(), abs=1e-3)
    # Padded options never receive probability, however flat the temperature.
    assert flat[0] == pytest.approx(100.0, abs=1e-3)  # item "a" is itself uniform over 2 options


def test_best_temperature_is_never_worse_than_no_temperature(items):
    preds = pad([[0.99, 0.01], [0.99, 0.01], [0.98, 0.01, 0.01], [0.99, 0.01]], 3)
    human, norm = pad(items["human_dist"], 3), items["norm"].to_numpy()
    plain = scores_at(preds, human, norm).mean()
    best, temperature = best_temperature(preds, human, norm)
    assert best >= plain
    assert temperature > 1  # over-confident predictions want flattening
    per_group = best_temperature_per_group(preds, human, norm, items["dataset_name"])
    assert per_group.mean() >= best - 1e-9


def test_bootstrap_interval_brackets_the_mean():
    x = np.random.default_rng(1).normal(5.0, 2.0, size=400)
    low, high = bootstrap_ci(x)
    assert low < x.mean() < high
    assert bootstrap_ci(x) == (low, high)  # seeded, so repeatable
