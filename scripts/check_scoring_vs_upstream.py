"""Check our SimBench score against the SimBench release's own scoring script.

Feeds both the same synthetic predictions for every item, in the layout the release script
expects (one results .pkl per split in a folder), and compares the per-model scores it
writes with ours. The pickles are created here, so nothing foreign is unpickled.

Run once after touching scoring.py. Needs network access to fetch the release script.
"""

import ast
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from subpop_simbench.data import SPLIT_FILES
from subpop_simbench.io import ITEMS_PATH, RAW_DIR
from subpop_simbench.scoring import score_predictions

UPSTREAM_SHA = "ec384dceff886e8c129eb610066ba62791aed3ea"
UPSTREAM_URL = (
    f"https://raw.githubusercontent.com/pitehu/SimBench_release/{UPSTREAM_SHA}/"
    "calculate_simbench_score.py"
)
FAKE_MODELS = {"dirichlet-flat": 1.0, "dirichlet-peaked": 0.2}


def fake_predictions(items: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for model, alpha in FAKE_MODELS.items():
        for item_id, n in zip(items["item_id"], items["n_options"], strict=True):
            rows.append((model, item_id, rng.dirichlet(np.full(n, alpha)).tolist()))
    return pd.DataFrame(rows, columns=["Model", "item_id", "pred"])


def run_upstream(items: pd.DataFrame, preds: pd.DataFrame, splits: list[str]) -> pd.Series:
    """Scores from the release script when its results folder holds `splits`."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        script = tmp / "calculate_simbench_score.py"
        with urllib.request.urlopen(UPSTREAM_URL, timeout=30) as resp:
            script.write_bytes(resp.read())
        results = tmp / "results"
        results.mkdir()
        for split in splits:
            raw = pd.read_csv(RAW_DIR / SPLIT_FILES[split])
            ids = items.loc[items["split"] == split, "item_id"].tolist()
            assert len(ids) == len(raw)
            base = pd.DataFrame(
                {
                    "item_id": ids,
                    "human_answer": raw["human_answer"].map(ast.literal_eval),
                    "dataset_name": raw["dataset_name"],
                }
            )
            for model in FAKE_MODELS:
                df = base.merge(preds[preds["Model"] == model], on="item_id", validate="1:1")
                df = df.rename(columns={"pred": "Response_Distribution"})
                df.to_pickle(results / f"{split}_{model}.pkl")
        subprocess.run(
            [sys.executable, str(script), str(results)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return pd.read_csv(results / "simbench_scores.csv", index_col=0)["SimBench_Score"]


def main() -> None:
    items = pd.read_parquet(ITEMS_PATH)
    preds = fake_predictions(items)
    ok = True
    for splits in (["pop", "grouped"], ["pop"], ["grouped"]):
        subset = items[items["split"].isin(splits)]
        upstream = run_upstream(items, preds, splits)
        ours = (
            score_predictions(
                items, preds[preds["item_id"].isin(subset["item_id"])], norm_items=subset
            )
            .groupby("Model")["score"]
            .mean()
        )
        for model in FAKE_MODELS:
            match = abs(upstream[model] - ours[model]) < 0.005  # upstream rounds to 2 dp
            ok &= match
            print(
                f"{'+'.join(splits):<12} {model:<17} upstream={upstream[model]:8.2f} "
                f"ours={ours[model]:10.4f}  {'ok' if match else 'MISMATCH'}"
            )
    if not ok:
        raise SystemExit("scores differ from the SimBench release script")
    print("our scoring matches the SimBench release script on identical inputs")


if __name__ == "__main__":
    main()
