"""Paths and safe file writes shared by every stage."""

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ITEMS_PATH = DATA_DIR / "items.parquet"
PROMPTS_PATH = DATA_DIR / "prompts.parquet"
RESULTS_DIR = ROOT / "results"
# SubPOP's gated training data and anything that quotes its questions stay under data/,
# which is not committed. Only aggregate tables and figures go to reports/.
SUBPOP_DIR = DATA_DIR / "subpop"
OVERLAP_PATH = DATA_DIR / "analysis" / "overlap_items.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _match_mode(tmp_path: str, output_path: Path) -> None:
    # mkstemp creates 0600 and os.replace keeps the source mode; restore the usual one.
    if output_path.exists():
        os.chmod(tmp_path, stat.S_IMODE(output_path.stat().st_mode))
    else:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)


def write_parquet_safely(df: pd.DataFrame, output_path: Path) -> None:
    """Write df so output_path only ever holds a complete, verified file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        _match_mode(tmp_path, output_path)
        meta = pq.read_metadata(tmp_path)
        assert meta.num_rows == len(df), f"row count {meta.num_rows} != {len(df)}"
        on_disk = set(meta.schema.to_arrow_schema().names)
        assert on_disk >= set(df.columns), f"columns missing on disk: {set(df.columns) - on_disk}"
        os.replace(tmp_path, output_path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def write_json_safely(obj, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
        _match_mode(tmp_path, output_path)
        os.replace(tmp_path, output_path)
    except BaseException:
        os.unlink(tmp_path)
        raise
