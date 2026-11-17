"""Stage 2: run one model (base, or base + adapter) over data/prompts.parquet.

    uv run python scripts/run_inference.py --tag mistral-7b__base \\
        --base mistralai/Mistral-7B-v0.1
    uv run python scripts/run_inference.py --tag mistral-7b__subpop \\
        --base mistralai/Mistral-7B-v0.1 --adapter jjssuh/mistral-7b-v0.1-subpop

Output goes to results/<tag>/. The run is resumable: repeat the same command after a crash.
"""

import argparse

from subpop_simbench.infer import ModelSpec
from subpop_simbench.prompts import CONDITIONS
from subpop_simbench.runner import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", required=True, help="names the run directory under results/")
    parser.add_argument("--base", required=True, help="Hugging Face id or path of the base model")
    parser.add_argument("--adapter", help="Hugging Face id or path of a LoRA adapter")
    parser.add_argument("--base-revision")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--dtype", default="auto", help="auto, bfloat16, float16 or float32")
    parser.add_argument("--quantization", choices=["8bit", "4bit"])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--limit", type=int, help="run a seeded sample of this many items")
    parser.add_argument("--seed", type=int, default=0, help="seed for --limit")
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--shard-rows", type=int, default=2000)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: best available")
    parser.add_argument("--threads", type=int, help="cap on CPU threads, for a gentler local run")
    args = parser.parse_args()

    spec = ModelSpec(
        base=args.base,
        adapter=args.adapter,
        base_revision=args.base_revision,
        adapter_revision=args.adapter_revision,
        dtype=args.dtype,
        quantization=args.quantization,
    )
    run_inference(
        spec,
        tag=args.tag,
        conditions=args.conditions,
        limit=args.limit,
        seed=args.seed,
        max_batch_tokens=args.max_batch_tokens,
        shard_rows=args.shard_rows,
        device=args.device,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
