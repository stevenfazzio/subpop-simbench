"""Stage 0: download the two SimBench CSVs at a pinned revision into data/raw/.

The CSVs hold the same rows as the .pkl files the SimBench repo points at. We use the CSVs
so that nothing has to be unpickled.
"""

import shutil

from huggingface_hub import hf_hub_download

from subpop_simbench.io import RAW_DIR, sha256_file

REPO_ID = "pitehu/SimBench"
REVISION = "f471cf75fd9b8b1eca1ba8724ce6bd255190a11d"

# Hashes of the files at REVISION. A mismatch means the download is corrupt or the pin moved.
EXPECTED_SHA256 = {
    "SimBenchPop.csv": "446c46a533c9a49b9c86966bce2b1a3135544f81b6f1615fe479fce9aa36ceec",
    "SimBenchGrouped.csv": "944b25581271a7167c5bbd4fa65f8bcb568cd9216b1fdb9c474c152d7995604c",
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, expected in EXPECTED_SHA256.items():
        dest = RAW_DIR / filename
        if not dest.exists():
            cached = hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=REVISION)
            tmp = dest.with_suffix(".csv.tmp")
            shutil.copyfile(cached, tmp)
            tmp.replace(dest)
        digest = sha256_file(dest)
        status = "ok" if digest == expected else "UNPINNED" if expected is None else "MISMATCH"
        print(f"{filename}: {dest.stat().st_size:,} bytes  sha256={digest}  [{status}]")
        if status == "MISMATCH":
            raise SystemExit(f"{filename}: expected sha256 {expected}")


if __name__ == "__main__":
    main()
