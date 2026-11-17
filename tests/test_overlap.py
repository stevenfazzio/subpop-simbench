import pandas as pd
import pytest

from subpop_simbench.overlap import (
    STATUSES,
    match_items,
    norm_options,
    norm_text,
    training_copy,
)


def test_norm_text():
    assert norm_text("Don’t  you THINK the U.S. should...?") == "don t you think the u s should"
    assert norm_text("  Plain text ") == "plain text"


def test_norm_options_ignores_order_case_and_refused():
    assert norm_options(["Yes", "No", "Refused"]) == norm_options(["no", "YES"])
    assert norm_options(["Yes", "No"]) != norm_options(["Yes", "Maybe"])


@pytest.fixture
def train():
    def record(qkey, question, options, attribute="SEX", group="Male", responses=(0.25, 0.75)):
        return {
            "qkey": qkey,
            "question": question,
            "options": options,
            "attribute": attribute,
            "group": group,
            "responses": list(responses),
            "refusal_rate": 0.2,
        }

    return pd.DataFrame(
        [
            record("TRUST_W70", "Do you trust scientists?", ["Yes", "No", "Refused"]),
            # The same question again for another subgroup: must not count twice.
            record(
                "TRUST_W70", "Do you trust scientists?", ["Yes", "No", "Refused"], group="Female"
            ),
            record(
                "VIEW_W71", "Which statement comes closer to your view?", ["Taxes up", "Taxes down"]
            ),
            record("NEWS_W72", "How often do you get news from television?", ["Often", "Never"]),
        ]
    )


@pytest.fixture
def items():
    return pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4"],
            "split": ["grouped", "pop", "pop", "pop"],
            "dataset_name": ["OpinionQA", "OpinionQA", "Afrobarometer", "Jester"],
            "question": [
                "Do you trust scientists",  # punctuation aside, identical; same options
                "Which statement comes closer to your view?",  # identical text, other options
                "How often do you get news from the television?",  # one word added
                "Rate this joke about a chicken crossing a road",
            ],
            "option_texts": [
                ["Yes", "No", "Refused"],
                ["Guns up", "Guns down"],
                ["Often", "Never"],
                ["1", "2"],
            ],
            "human_dist": [[0.2, 0.6, 0.2], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            "subpop_attribute": ["SEX", None, None, None],
            "subpop_group": ["Male", None, None, None],
            "norm": [0.3, 0.3, 0.3, 0.3],
        }
    )


def test_match_items_buckets(items, train):
    matches = match_items(items, train, workers=1)
    assert matches["item_id"].tolist() == items["item_id"].tolist()
    assert matches["status"].tolist() == [STATUSES[0], STATUSES[1], STATUSES[1], STATUSES[3]]
    assert matches.loc[0, "best_train_qkey"] == "TRUST_W70"
    assert matches.loc[1, "exact_question"] and not matches.loc[1, "exact_question_and_options"]
    assert not matches.loc[2, "exact_question"] and matches.loc[2, "best_ratio"] >= 95
    assert matches.loc[3, "best_ratio"] < 80


def test_training_copy_restores_the_refusal_share(items, train):
    matches = match_items(items, train, workers=1)
    copies = training_copy(items, matches, train)
    # Only i1 is a Grouped OpinionQA item with question, options and subgroup all in training.
    assert copies["item_id"].tolist() == ["i1"]
    # Training holds (0.25, 0.75) over the answered 80% plus 20% refusals: (0.2, 0.6, 0.2),
    # exactly SimBench's distribution.
    assert copies.loc[0, "tvd_training_vs_simbench"] == pytest.approx(0.0)
    assert copies.loc[0, "score_of_copying_training"] == pytest.approx(100.0)
