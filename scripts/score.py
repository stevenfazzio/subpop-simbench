"""Stage 3: SimBench scores for finished runs.

    uv run python scripts/score.py                 # every complete run under results/
    uv run python scripts/score.py mistral-7b__base mistral-7b__subpop --by-dataset

Each prompt condition is scored with its own readout rule (see prompts.PRIMARY_READOUT).
The norm in the score is pooled per dataset over all 13,510 items, which is what SimBench's
release script does when both split files sit in its results folder.
"""

import argparse
import json

import pandas as pd

from subpop_simbench.io import ITEMS_PATH, RESULTS_DIR
from subpop_simbench.prompts import CONDITIONS
from subpop_simbench.results import score_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="*", help="run tags; default is every complete run")
    parser.add_argument(
        "--by-dataset", action="store_true", help="also break scores out by dataset"
    )
    args = parser.parse_args()

    if args.runs:
        run_dirs = [RESULTS_DIR / tag for tag in args.runs]
    else:
        run_dirs = sorted(
            p.parent
            for p in RESULTS_DIR.glob("*/meta.json")
            if json.loads(p.read_text())["status"] == "complete"
        )
    if not run_dirs:
        raise SystemExit(f"no complete runs under {RESULTS_DIR}")

    scored = score_runs(pd.read_parquet(ITEMS_PATH), run_dirs)
    scored["condition"] = pd.Categorical(scored["condition"], CONDITIONS, ordered=True)
    pd.set_option("display.width", 200)

    by_split = scored.pivot_table(
        index=["run", "condition"], columns="split", values="score", aggfunc="mean", observed=True
    )
    overall = scored.groupby(["run", "condition"], observed=True).agg(
        n=("score", "size"),
        score=("score", "mean"),
        tvd=("tvd", "mean"),
        answer_mass=("answer_mass", "mean"),
    )
    print(overall.join(by_split.add_prefix("score_")).round(3).to_string())

    if args.by_dataset:
        print()
        print(
            scored.pivot_table(
                index=["split", "dataset_name"],
                columns=["run", "condition"],
                values="score",
                aggfunc="mean",
                observed=True,
            )
            .round(1)
            .to_string()
        )


if __name__ == "__main__":
    main()
