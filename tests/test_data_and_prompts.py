import pytest

from subpop_simbench.data import (
    _to_subpop_group,
    load_subpop_steering,
    load_subpop_trained_groups,
    parse_input_template,
)
from subpop_simbench.prompts import render_simbench, render_subpop_exact, render_subpop_skeleton

GROUP = "You are from the United States. The year is 2017."
TEMPLATE = "Do you agree?\n\nOptions:\n(A): Yes\n(B): No\n(C): Refused"


def test_parse_input_template():
    assert parse_input_template(TEMPLATE) == (
        "Do you agree?",
        ["A", "B", "C"],
        ["Yes", "No", "Refused"],
    )


def test_parse_keeps_multiline_options_and_inner_options_text():
    text = "Pick.\nOptions: none above\n\nOptions:\n(A): Stay\nDead:\n * 1 man\n(B): Swerve"
    question, labels, texts = parse_input_template(text)
    assert question == "Pick.\nOptions: none above"
    assert labels == ["A", "B"]
    assert texts == ["Stay\nDead:\n * 1 man", "Swerve"]


def test_parse_rejects_missing_options():
    with pytest.raises(ValueError):
        parse_input_template("No options here")


def test_subpop_group_mapping():
    steering = load_subpop_steering()
    assert _to_subpop_group({"POLIDEOLOGY": "very liberal"}, steering) == (
        "POLIDEOLOGY",
        "Very liberal",
    )
    assert _to_subpop_group({"POLPARTY": "Other"}, steering) == ("POLPARTY", "Something else")
    assert _to_subpop_group({"MARITAL": "never been married"}, steering) == (
        "MARITAL",
        "Unmarried and have never been married",
    )
    assert _to_subpop_group({}, steering) is None
    assert _to_subpop_group({"cntry": "Finland"}, steering) is None
    with pytest.raises(ValueError):
        _to_subpop_group({"RACE": "martian"}, steering)


def test_trained_groups_are_the_22():
    trained = load_subpop_trained_groups()
    assert len(trained) == 22
    assert ("POLIDEOLOGY", "Moderate") in trained
    assert ("AGE", "18-29") not in trained


def test_render_simbench_matches_release_format():
    assert render_simbench(GROUP, TEMPLATE, ["A", "B", "C"]) == (
        "You are a group of individuals with these shared characteristics:\n"
        "You are from the United States. The year is 2017.\n"
        "**Question**: Do you agree?\n\nOptions:\n(A): Yes\n(B): No\n(C): Refused\n"
        "Do not provide any explanation, only answer with one of the following options: A, B, C.\n"
        "**Answer**: ("
    )


def test_render_subpop_skeleton():
    assert render_subpop_skeleton(GROUP, "Do you agree?", ["Yes", "No", "Refused"]) == (
        "You are from the United States. The year is 2017.\n\n"
        "Question: Do you agree?\nA. Yes\nB. No\nC. Refused\n"
        "Answer as a choice between A.,B.,C.\n\nAnswer:"
    )


def test_render_subpop_exact():
    prompt = render_subpop_exact(
        "POLIDEOLOGY", "Conservative", "Do you agree?", ["Yes", "No"], load_subpop_steering()
    )
    assert prompt == (
        "Question: In general, would you describe your political views as\n"
        "A. Very conservative\nB. Conservative\nC. Moderate\nD. Liberal\nE. Very liberal\n\n"
        "Answer: B. Conservative\n\n"
        "Answer the following question keeping in mind your previous answers.\n"
        "Question: Do you agree?\nA. Yes\nB. No\n"
        "Answer as a choice between A.,B.\n\nAnswer:"
    )
