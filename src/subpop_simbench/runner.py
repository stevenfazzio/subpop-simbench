"""Run one model over the prompt table, resumably.

A run lives in results/<tag>/:

    meta.json            what was run (model, adapter, conditions, prompt-file hash) and where
    shards/part-*.parquet  finished work, written every `shard_rows` prompts
    predictions.parquet  the consolidated result, written once every prompt is done

Killing a run loses at most one shard of work. Re-running the same command picks up from
the shards; re-running with a different model or prompt file under the same tag is refused.
"""

import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import peft
import torch
import transformers

from subpop_simbench.infer import (
    ModelSpec,
    iter_first_token_probs,
    load_model,
    pick_device,
    tokenize,
)
from subpop_simbench.io import (
    ITEMS_PATH,
    PROMPTS_PATH,
    RESULTS_DIR,
    ROOT,
    sha256_file,
    write_json_safely,
    write_parquet_safely,
)
from subpop_simbench.prompts import CONDITIONS
from subpop_simbench.readout import RULES

PREDICTION_COLUMNS = [
    "item_id",
    "condition",
    "n_tokens",
    *[f"p_{rule}" for rule in RULES],
    "top_token",
    "top_prob",
]
_KEY = ["item_id", "condition"]


def _git_state() -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=True
            )
            return out.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def select_prompts(
    prompts: pd.DataFrame, conditions: tuple[str, ...] | list[str], limit: int | None, seed: int
) -> pd.DataFrame:
    """The prompts a run covers: the chosen conditions, optionally for a sample of items."""
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    out = prompts[prompts["condition"].isin(conditions)]
    if limit is not None:
        item_ids = np.array(sorted(out["item_id"].unique()))
        keep = np.random.default_rng(seed).choice(
            item_ids, size=min(limit, len(item_ids)), replace=False
        )
        out = out[out["item_id"].isin(keep)]
    return out.sort_values(_KEY).reset_index(drop=True)


def _load_shards(shard_dir: Path) -> pd.DataFrame:
    paths = sorted(shard_dir.glob("part-*.parquet"))
    if not paths:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _next_shard_index(shard_dir: Path) -> int:
    indices = [int(p.stem.split("-")[1]) for p in shard_dir.glob("part-*.parquet")]
    return max(indices, default=-1) + 1


def run_inference(
    spec: ModelSpec,
    tag: str,
    conditions: tuple[str, ...] | list[str] = CONDITIONS,
    limit: int | None = None,
    seed: int = 0,
    max_batch_tokens: int = 8192,
    shard_rows: int = 2000,
    device: str | None = None,
    threads: int | None = None,
    prompts_path: Path = PROMPTS_PATH,
    items_path: Path = ITEMS_PATH,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Run `spec` over the selected prompts and return the path of predictions.parquet.

    `device` and `threads` say where this session computes; they are not part of the run's
    identity, so an interrupted run can be resumed on different hardware.
    """
    run_dir = Path(results_dir) / tag
    shard_dir = run_dir / "shards"
    final_path = run_dir / "predictions.parquet"
    meta_path = run_dir / "meta.json"

    identity = {
        "model": spec.as_dict(),
        "conditions": sorted(conditions),
        "limit": limit,
        "seed": seed,
        "prompts_sha256": sha256_file(prompts_path),
    }
    earlier_sessions: list[dict] = []
    if meta_path.exists():
        previous_meta = json.loads(meta_path.read_text())
        previous = previous_meta["identity"]
        earlier_sessions = previous_meta.get("sessions", [])
        if previous != identity:
            raise SystemExit(
                f"{run_dir} holds a different run.\n  there: {previous}\n  asked: {identity}\n"
                "Use another --tag, or delete that directory if it is no longer wanted."
            )
    if final_path.exists():
        print(f"{tag}: already complete at {final_path}")
        return final_path

    prompts = select_prompts(pd.read_parquet(prompts_path), conditions, limit, seed)
    n_options = pd.read_parquet(items_path, columns=["item_id", "n_options"])
    prompts = prompts.merge(n_options, on="item_id", how="left", validate="many_to_one")
    assert prompts["n_options"].notna().all(), "prompts reference items missing from items.parquet"

    shard_dir.mkdir(parents=True, exist_ok=True)
    done = _load_shards(shard_dir)
    done_keys = set(zip(done["item_id"], done["condition"], strict=True))
    is_done = [
        key in done_keys for key in zip(prompts["item_id"], prompts["condition"], strict=True)
    ]
    todo = prompts[~np.array(is_done, dtype=bool)].reset_index(drop=True)
    print(f"{tag}: {len(prompts):,} prompts, {len(done_keys):,} already done, {len(todo):,} to go")

    device = device or pick_device()
    if threads is not None:
        torch.set_num_threads(threads)
    # One entry per invocation, so a run resumed on other hardware keeps its full history.
    session = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "prompts_to_run": len(todo),
        "device": device,
        "threads": torch.get_num_threads() if device == "cpu" else None,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "git": _git_state(),
    }
    meta = {"identity": identity, "status": "running", "sessions": [*earlier_sessions, session]}
    write_json_safely(meta, meta_path)

    if len(todo):
        model, tokenizer = load_model(spec, device)
        input_ids = tokenize(tokenizer, todo["prompt"].tolist())
        total_tokens = sum(len(ids) for ids in input_ids)
        print(
            f"{tag}: {total_tokens:,} tokens to run on {device} "
            f"(longest prompt {max(len(ids) for ids in input_ids)} tokens)"
        )

        buffer: list[dict] = []
        shard_index = _next_shard_index(shard_dir)
        tokens_done = 0
        rows_done = 0
        t0 = time.monotonic()

        def flush() -> None:
            nonlocal buffer, shard_index
            if not buffer:
                return
            shard = pd.DataFrame(buffer, columns=PREDICTION_COLUMNS)
            write_parquet_safely(shard, shard_dir / f"part-{shard_index:05d}.parquet")
            shard_index += 1
            buffer = []
            elapsed = time.monotonic() - t0
            rate = tokens_done / elapsed
            eta = (total_tokens - tokens_done) / rate if rate > 0 else float("nan")
            print(
                f"{tag}: {rows_done:,}/{len(todo):,} prompts, {rate:,.0f} tok/s, "
                f"about {eta / 60:.1f} min left",
                flush=True,
            )

        for batch in iter_first_token_probs(model, tokenizer, input_ids, max_batch_tokens):
            for row, i in enumerate(batch.indices):
                n = int(todo.at[i, "n_options"])
                record = {
                    "item_id": todo.at[i, "item_id"],
                    "condition": todo.at[i, "condition"],
                    "n_tokens": batch.n_tokens[row],
                    "top_token": batch.top_token[row],
                    "top_prob": batch.top_prob[row],
                }
                for rule in RULES:
                    record[f"p_{rule}"] = batch.letter_probs[rule][row, :n].astype(float).tolist()
                buffer.append(record)
            tokens_done += sum(batch.n_tokens)
            rows_done += len(batch.indices)
            if len(buffer) >= shard_rows:
                flush()
        flush()
        session["wall_seconds"] = round(time.monotonic() - t0, 1)

    # Consolidate only once every expected prompt is present exactly once.
    result = _load_shards(shard_dir)
    got = list(zip(result["item_id"], result["condition"], strict=True))
    expected = set(zip(prompts["item_id"], prompts["condition"], strict=True))
    assert len(got) == len(set(got)), f"{len(got) - len(set(got))} duplicated predictions in shards"
    assert set(got) == expected, (
        f"{len(expected - set(got))} prompts missing, {len(set(got) - expected)} unexpected"
    )
    result = result.sort_values(_KEY).reset_index(drop=True)[PREDICTION_COLUMNS]
    write_parquet_safely(result, final_path)
    shutil.rmtree(shard_dir)

    meta["status"] = "complete"
    meta["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["n_predictions"] = len(result)
    write_json_safely(meta, meta_path)
    print(f"{tag}: wrote {len(result):,} predictions to {final_path}")
    return final_path
