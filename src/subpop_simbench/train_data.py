"""SubPOP's fine-tuning examples, rebuilt from its released response distributions.

Mirrors SubPOP's `scripts/data_generation/prepare_finetuning_data.py` for the QA steering
format its released adapters were trained with:

- One example per (question, subgroup) row of SubPOP-Train.
- The prompt is SubPOP's QA steering turn followed by the survey question: the same text
  the `subpop_exact` condition uses at inference time.
- The target is the subgroup's answer shares followed by its refusal share, matching the
  options, whose last entry is the refusal option. SubPOP does not renormalise after
  appending the refusal share, so a target sums to one plus that share. Copied as is;
  `train.distribution_loss` says what it amounts to.
- Questions, not rows, are split into training and validation, 90% to 10%.

SubPOP's own split came from a seed and a row order that are not published, so the
questions held out here are not the ones SubPOP held out.
"""

import numpy as np
import pandas as pd

from subpop_simbench.prompts import render_subpop_exact

# SubPOP pads every target to ten options and reads the letters A to J.
MAX_OPTIONS = 10

EXAMPLE_COLUMNS = [
    "example_id",
    "split",
    "qkey",
    "attribute",
    "group",
    "prompt",
    "target",
    "n_options",
]


def split_questions(qkeys, val_ratio: float = 0.1, seed: int = 0) -> dict[str, str]:
    """Assign every question to "train" or "val". Depends only on the set of questions."""
    unique = sorted(set(qkeys))
    order = np.random.default_rng(seed).permutation(len(unique))
    held_out = {unique[i] for i in order[: int(round(val_ratio * len(unique)))]}
    return {qkey: "val" if qkey in held_out else "train" for qkey in unique}


def build_examples(
    records: pd.DataFrame, steering: dict[str, dict], val_ratio: float = 0.1, seed: int = 0
) -> pd.DataFrame:
    """One fine-tuning example per SubPOP-Train row, in a deterministic order.

    records   SubPOP-Train (`qkey`, `attribute`, `group`, `question`, `options`, `responses`,
              `refusal_rate`), as `overlap.load_subpop_records` returns it.
    steering  SubPOP's steering templates, from `data.load_subpop_steering`.
    """
    splits = split_questions(records["qkey"], val_ratio, seed)
    rows = []
    for record in records.itertuples(index=False):
        options = list(record.options)
        target = [float(share) for share in record.responses] + [float(record.refusal_rate)]
        where = f"{record.qkey} / {record.attribute} / {record.group}"
        if len(target) != len(options):
            raise ValueError(f"{where}: {len(options)} options but {len(target)} shares")
        if len(options) > MAX_OPTIONS:
            raise ValueError(f"{where}: {len(options)} options; SubPOP reads at most ten")
        rows.append(
            {
                "example_id": f"{record.qkey}|{record.attribute}|{record.group}",
                "split": splits[record.qkey],
                "qkey": record.qkey,
                "attribute": record.attribute,
                "group": record.group,
                "prompt": render_subpop_exact(
                    record.attribute, record.group, record.question, options, steering
                ),
                "target": target,
                "n_options": len(options),
            }
        )
    examples = pd.DataFrame(rows, columns=EXAMPLE_COLUMNS)
    assert len(examples) == len(records), f"{len(records)} rows in, {len(examples)} out"
    if examples["example_id"].duplicated().any():
        raise ValueError("SubPOP-Train repeats a (question, attribute, subgroup) row")
    return examples.sort_values(["split", "example_id"]).reset_index(drop=True)
