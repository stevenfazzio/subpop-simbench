"""Which vocabulary ids count as "the model answered A".

Both upstream projects read the answer off the next-token distribution, but they pick the
ids for each letter differently. We keep both rules, each a faithful copy of its source, and
record both for every prompt; `prompts.PRIMARY_READOUT` says which one a condition is scored
with.

simbench  SimBench `get_token_probabilities_local_model`: every id in encode(L) plus every
          id in encode("_" + L). On SentencePiece vocabularies that is the spaced letter, the
          bare letter, and the "▁_" piece (the same small constant added to every option).
subpop    SubPOP `run_inference.inference_offline`: the last id of encode("Answer:" + L)
          plus the last id of encode("Answer: " + L).

Ids are summed with multiplicity, as upstream does, so the result matches their arithmetic.
"""

import torch

from subpop_simbench.data import LETTERS

RULES = ("simbench", "subpop")


def simbench_ids(tokenizer, letter: str) -> list[int]:
    return tokenizer.encode(letter, add_special_tokens=False) + tokenizer.encode(
        "_" + letter, add_special_tokens=False
    )


def subpop_ids(tokenizer, letter: str) -> list[int]:
    return [
        tokenizer.encode("Answer:" + letter, add_special_tokens=False)[-1],
        tokenizer.encode("Answer: " + letter, add_special_tokens=False)[-1],
    ]


_RULE_FNS = {"simbench": simbench_ids, "subpop": subpop_ids}


def letter_ids(tokenizer, rule: str) -> list[list[int]]:
    """For each of the 26 letters, the ids `rule` sums over."""
    return [_RULE_FNS[rule](tokenizer, letter) for letter in LETTERS]


def readout_matrix(tokenizer, rule: str, vocab_size: int) -> torch.Tensor:
    """(26, vocab) count matrix M, so that `probs @ M.T` is each letter's probability."""
    matrix = torch.zeros(len(LETTERS), vocab_size, dtype=torch.float32)
    for row, ids in enumerate(letter_ids(tokenizer, rule)):
        for token_id in ids:
            matrix[row, token_id] += 1.0
    return matrix
