"""Load the two SimBench CSVs into one item table.

The splits ship with different columns. Everything downstream needs the same handful of
fields, so this module reconciles them and parses the string-encoded cells once.
"""

import ast
import json
import re
from pathlib import Path

import pandas as pd

from subpop_simbench._vendor import subpop as _subpop_vendor

SPLIT_FILES = {"pop": "SimBenchPop.csv", "grouped": "SimBenchGrouped.csv"}
LETTERS = [chr(ord("A") + i) for i in range(26)]

ITEM_COLUMNS = [
    "item_id",
    "split",
    "dataset_name",
    "group_prompt",
    "group_vars",
    "input_template",
    "question",
    "option_labels",
    "option_texts",
    "n_options",
    "human_dist",
    "group_size",
    "subpop_attribute",
    "subpop_group",
    "subpop_group_trained",
]

_OPTION_RE = re.compile(r"^\(([A-Z])\): (.*)$")
_OPTIONS_MARKER = "\nOptions:\n"
_VENDOR_DIR = Path(_subpop_vendor.__file__).parent

# SimBench lower-cases most OpinionQA group values and renames two of them. These are the
# only values that do not match a SubPOP steering option case-insensitively.
_SUBPOP_VALUE_ALIASES = {
    ("MARITAL", "never been married"): "Unmarried and have never been married",
    ("POLPARTY", "other"): "Something else",
}


def parse_input_template(text: str) -> tuple[str, list[str], list[str]]:
    """Split a SimBench `input_template` into (question, option labels, option texts).

    Every template is `<question>\\n\\nOptions:\\n(A): ...\\n(B): ...`. Option texts can span
    lines (MoralMachine lists casualties under each option), so a line that does not open a
    new option continues the previous one.
    """
    question, marker, block = text.rpartition(_OPTIONS_MARKER)
    if not marker:
        raise ValueError(f"no options block in input_template: {text[:80]!r}")
    labels: list[str] = []
    texts: list[str] = []
    for line in block.split("\n"):
        m = _OPTION_RE.match(line)
        if m:
            labels.append(m.group(1))
            texts.append(m.group(2))
        elif texts:
            texts[-1] += "\n" + line
        else:
            raise ValueError(f"options block does not start with an option: {block[:80]!r}")
    return question.strip(), labels, texts


def load_subpop_steering() -> dict[str, dict]:
    """SubPOP's steering templates, keyed by attribute, with options parsed to lists."""
    raw = json.loads((_VENDOR_DIR / "steering_prompts.json").read_text())
    return {row["attribute"]: {**row, "options": ast.literal_eval(row["options"])} for row in raw}


def load_subpop_trained_groups() -> set[tuple[str, str]]:
    """The 22 (attribute, subgroup) pairs SubPOP's adapters were trained on."""
    df = pd.read_csv(_VENDOR_DIR / "demographics_22.csv")
    return {
        (attribute, group)
        for attribute, groups in zip(df["attribute"], df["group"], strict=True)
        for group in ast.literal_eval(groups)
    }


def _to_subpop_group(group_vars: dict, steering: dict[str, dict]) -> tuple[str, str] | None:
    """Map a SimBench OpinionQA group onto SubPOP's (attribute, option), if it has one."""
    if len(group_vars) != 1:
        return None
    ((attribute, value),) = group_vars.items()
    if attribute not in steering:
        return None
    options = steering[attribute]["options"]
    alias = _SUBPOP_VALUE_ALIASES.get((attribute, str(value).lower()))
    if alias is not None:
        return attribute, alias
    matches = [opt for opt in options if opt.lower() == str(value).lower()]
    if len(matches) != 1:
        raise ValueError(f"OpinionQA group {attribute}={value!r} has no SubPOP option in {options}")
    return attribute, matches[0]


def _load_split(raw_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / SPLIT_FILES[split])
    steering = load_subpop_steering()
    trained = load_subpop_trained_groups()
    prefix = {"pop": "pop", "grouped": "grp"}[split]

    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        group_vars = ast.literal_eval(row.group_prompt_variable_map)
        # Same substitution SimBench's generate_prompt applies. The released CSVs already
        # have the values filled in, so today this is a no-op kept for fidelity.
        group_prompt = row.group_prompt_template
        for variable, value in group_vars.items():
            group_prompt = group_prompt.replace(f"{{{variable}}}", str(value))

        question, labels, texts = parse_input_template(row.input_template)
        human = ast.literal_eval(row.human_answer)
        if list(human) != labels or labels != LETTERS[: len(labels)]:
            raise ValueError(f"{split} row {i}: option labels {labels} vs answers {list(human)}")
        total = sum(human.values())
        if not total > 0:
            raise ValueError(f"{split} row {i}: human_answer sums to {total}")

        subpop = None
        if split == "grouped" and row.dataset_name == "OpinionQA":
            subpop = _to_subpop_group(group_vars, steering)

        rows.append(
            {
                "item_id": f"{prefix}-{i:05d}",
                "split": split,
                "dataset_name": row.dataset_name,
                "group_prompt": group_prompt,
                "group_vars": json.dumps(group_vars, sort_keys=True),
                "input_template": row.input_template,
                "question": question,
                "option_labels": labels,
                "option_texts": texts,
                "n_options": len(labels),
                "human_dist": [v / total for v in human.values()],
                "group_size": int(row.group_size),
                "subpop_attribute": subpop[0] if subpop else None,
                "subpop_group": subpop[1] if subpop else None,
                "subpop_group_trained": (subpop in trained) if subpop else None,
            }
        )
    out = pd.DataFrame(rows, columns=ITEM_COLUMNS)
    assert len(out) == len(df), f"{split}: {len(df)} rows in, {len(out)} out"
    return out


def load_items(raw_dir: Path) -> pd.DataFrame:
    """One row per SimBench test case, both splits, with a stable `item_id`."""
    items = pd.concat([_load_split(Path(raw_dir), s) for s in SPLIT_FILES], ignore_index=True)
    assert items["item_id"].is_unique
    return items
