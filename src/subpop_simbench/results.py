"""Read finished runs back and score them."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from subpop_simbench.prompts import PRIMARY_READOUT
from subpop_simbench.readout import RULES
from subpop_simbench.scoring import score_predictions


def load_run(run_dir: Path) -> pd.DataFrame:
    """Predictions of one finished run.

    Adds `pred`, the distribution under the readout rule that belongs to each row's prompt
    condition, normalised to sum to 1, and `answer_mass`, the probability the model put on
    the answer letters at all before normalising.
    """
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    if meta["status"] != "complete":
        raise ValueError(f"{run_dir.name} is not complete (status: {meta['status']})")
    df = pd.read_parquet(run_dir / "predictions.parquet")

    by_rule = {rule: df[f"p_{rule}"].tolist() for rule in RULES}
    preds, masses = [], []
    for i, rule in enumerate(df["condition"].map(PRIMARY_READOUT)):
        raw = np.asarray(by_rule[rule][i], dtype=float)
        total = float(raw.sum())
        if not total > 0:
            raise ValueError(
                f"{run_dir.name}: {df.at[i, 'item_id']} / {df.at[i, 'condition']} has no "
                "probability on any answer letter"
            )
        preds.append(raw / total)
        masses.append(total)
    df["pred"] = preds
    df["answer_mass"] = masses
    df.insert(0, "run", run_dir.name)
    return df


def score_runs(items: pd.DataFrame, run_dirs: list[Path]) -> pd.DataFrame:
    """Per-prediction SimBench scores for several runs, with split and dataset attached."""
    preds = pd.concat([load_run(d) for d in run_dirs], ignore_index=True)
    scored = score_predictions(items, preds)
    out = scored.merge(items[["item_id", "split"]], on="item_id", how="left", validate="m:1")
    assert len(out) == len(preds)
    return out
