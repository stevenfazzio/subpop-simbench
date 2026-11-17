"""Render SimBench items under the three prompt conditions.

simbench         SimBench's own base-model prompt, unchanged. Leaderboard-comparable.
subpop_skeleton  SimBench's group sentence as a plain preamble, then SubPOP's question
                 layout. One mechanical rule for every item; nothing is authored per dataset.
subpop_exact     SubPOP's own QA steering turn and question layout. Only defined where
                 SubPOP has a template for the group: OpinionQA items in the Grouped split
                 with a single demographic attribute.
"""

import pandas as pd

from subpop_simbench._vendor.subpop.survey_utils import generate_mcq
from subpop_simbench.data import load_subpop_steering

CONDITIONS = ("simbench", "subpop_skeleton", "subpop_exact")

# Which token-id rule reads the answer off each condition's prompt. See readout.py.
PRIMARY_READOUT = {
    "simbench": "simbench",
    "subpop_skeleton": "subpop",
    "subpop_exact": "subpop",
}

PROMPT_COLUMNS = ["item_id", "condition", "prompt"]

_SIMBENCH_SYSTEM_PREFIX = "You are a group of individuals with these shared characteristics:\n"
_SIMBENCH_INSTRUCTION = (
    "Do not provide any explanation, only answer with one of the following options: "
)
_SUBPOP_QA_BRIDGE = "Answer the following question keeping in mind your previous answers.\n"


def render_simbench(group_prompt: str, input_template: str, option_labels: list[str]) -> str:
    """SimBench's `generate_prompt(..., 'token_prob')`, joined the way it is for base models."""
    system_prompt = _SIMBENCH_SYSTEM_PREFIX + group_prompt
    user_prompt = "**Question**: " + input_template + "\n"
    user_prompt += _SIMBENCH_INSTRUCTION + ", ".join(option_labels) + ".\n**Answer**: ("
    return system_prompt + "\n" + user_prompt


def render_subpop_skeleton(group_prompt: str, question: str, option_texts: list[str]) -> str:
    """Group sentence, blank line, then SubPOP's survey question ending in `Answer:`.

    The blank-line join is how SubPOP attaches its own free-text (BIO / PORTRAY) steering.
    """
    survey_prompt = generate_mcq(
        question_body=question, options=option_texts, add_answer_forcing=True
    )
    return group_prompt.strip() + "\n\n" + survey_prompt


def render_subpop_exact(
    attribute: str,
    subgroup: str,
    question: str,
    option_texts: list[str],
    steering: dict[str, dict],
) -> str:
    """SubPOP's QA-steering training prompt.

    Mirrors the QA branch of SubPOP's `prepare_finetuning_data.prepare_data` line for line.
    """
    steering_options = steering[attribute]["options"]
    steering_prompt = generate_mcq(
        question_body=steering[attribute]["qa_prompt"], options=steering_options
    )
    idx = steering_options.index(subgroup)
    steering_prompt += f" {chr(ord('A') + idx)}. {subgroup}\n\n"
    steering_prompt += _SUBPOP_QA_BRIDGE
    survey_prompt = generate_mcq(
        question_body=question, options=option_texts, add_answer_forcing=True
    )
    return steering_prompt + survey_prompt


def build_prompts(items: pd.DataFrame) -> pd.DataFrame:
    """Long table of (item_id, condition, prompt), in a deterministic order."""
    steering = load_subpop_steering()
    rows = []
    for item in items.itertuples(index=False):
        labels, texts = list(item.option_labels), list(item.option_texts)
        rows.append(
            (
                item.item_id,
                "simbench",
                render_simbench(item.group_prompt, item.input_template, labels),
            )
        )
        rows.append(
            (
                item.item_id,
                "subpop_skeleton",
                render_subpop_skeleton(item.group_prompt, item.question, texts),
            )
        )
        if item.subpop_attribute is not None and not pd.isna(item.subpop_attribute):
            rows.append(
                (
                    item.item_id,
                    "subpop_exact",
                    render_subpop_exact(
                        item.subpop_attribute, item.subpop_group, item.question, texts, steering
                    ),
                )
            )
    prompts = pd.DataFrame(rows, columns=PROMPT_COLUMNS)
    order = {c: i for i, c in enumerate(CONDITIONS)}
    prompts = prompts.sort_values(
        by=["condition", "item_id"], key=lambda s: s.map(order) if s.name == "condition" else s
    ).reset_index(drop=True)
    assert not prompts.duplicated(["item_id", "condition"]).any()
    return prompts
