import numpy as np
import pandas as pd
import pytest

from subpop_simbench.scoring import dataset_norms, score_predictions, total_variation, tv_to_uniform


@pytest.fixture
def items():
    return pd.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "dataset_name": ["X", "X", "Y"],
            "human_dist": [[0.5, 0.5], [1.0, 0.0], [0.7, 0.2, 0.1]],
        }
    )


def test_total_variation():
    assert total_variation([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.5)
    assert total_variation([20, 80], [0.2, 0.8]) == pytest.approx(0.0)  # normalises both sides
    with pytest.raises(ValueError):
        total_variation([1.0], [0.5, 0.5])


def test_tv_to_uniform():
    assert tv_to_uniform([1.0, 0.0]) == pytest.approx(0.5)
    assert np.isnan(tv_to_uniform([1.0]))


def test_norms_are_dataset_means(items):
    norms = dataset_norms(items)
    assert norms["X"] == pytest.approx(0.25)  # mean of 0.0 and 0.5
    assert norms["Y"] == pytest.approx(0.7 - 1 / 3)


def test_perfect_prediction_scores_100(items):
    preds = items[["item_id"]].assign(pred=items["human_dist"])
    assert score_predictions(items, preds)["score"].tolist() == pytest.approx([100.0] * 3)


def test_uniform_prediction_averages_zero_per_dataset(items):
    preds = items[["item_id"]].assign(pred=[[0.5, 0.5], [0.5, 0.5], [1 / 3] * 3])
    scored = score_predictions(items, preds)
    # Item "a" is itself uniform, so it scores 100 and "b" scores -100: zero on average.
    assert scored.groupby("dataset_name")["score"].mean().tolist() == pytest.approx([0.0, 0.0])


def test_unknown_item_is_an_error(items):
    with pytest.raises(ValueError):
        score_predictions(items, pd.DataFrame({"item_id": ["zzz"], "pred": [[0.5, 0.5]]}))
