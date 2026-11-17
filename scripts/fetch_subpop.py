"""Download SubPOP's training file at a pinned revision into data/subpop/.

The dataset is gated: request access at https://huggingface.co/datasets/jjssuh/subpop and
have a Hugging Face token available (HF_TOKEN, or `hf auth login`). It is CC BY-NC-SA 4.0
and is never committed. Only `scripts/overlap.py` needs it.
"""

import shutil

from huggingface_hub import hf_hub_download

from subpop_simbench.io import SUBPOP_DIR, sha256_file

REPO_ID = "jjssuh/subpop"
REVISION = "886d78a5cb55a576bd82b6724c46ae0f34a2ac3e"

# Hash of the file at REVISION. A mismatch means the download is corrupt or the pin moved.
# SubPOP-Eval (General Social Survey questions) is not used here and is not fetched.
EXPECTED_SHA256 = {
    "subpop_train.jsonl": "b349d4625e52a9099b77e6f90448e16cd20b985a3cb4b3d55cfd721c2bfea388",
}


def main() -> None:
    SUBPOP_DIR.mkdir(parents=True, exist_ok=True)
    for filename, expected in EXPECTED_SHA256.items():
        dest = SUBPOP_DIR / filename
        if not dest.exists():
            cached = hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=REVISION)
            tmp = dest.with_suffix(".jsonl.tmp")
            shutil.copyfile(cached, tmp)
            tmp.replace(dest)
        digest = sha256_file(dest)
        status = "ok" if digest == expected else "MISMATCH"
        print(f"{filename}: {dest.stat().st_size:,} bytes  sha256={digest}  [{status}]")
        if status == "MISMATCH":
            raise SystemExit(f"{filename}: expected sha256 {expected}")


if __name__ == "__main__":
    main()
