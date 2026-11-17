# subpop-simbench

Benchmarks the [SubPOP](https://github.com/josephjeesungsuh/subpop) fine-tuned adapters
([paper](https://aclanthology.org/2025.acl-long.1028/)) on
[SimBench](https://github.com/pitehu/SimBench_release)
([paper](https://arxiv.org/abs/2510.17516)).

The quantity of interest is **adapter minus base**. Each SubPOP adapter and its own base
model are run on identical prompts, and the difference in SimBench score is reported per
dataset. None of SubPOP's base models are on SimBench's leaderboard, so the bases are run
here too.

## Results

SubPOP's adapters were trained to predict how US subgroups answer Pew survey questions.
All four raise their base model's score on SimBench, and the gain is not confined to
OpinionQA, the one SimBench dataset drawn from the same survey programme.

Scores are base → with adapter. 100 matches the human answer distribution, 0 is no better
than answering uniformly, and negative is worse than uniform.

| Model | SimBench's format | SubPOP's layout | SubPOP's exact format (964 items) |
|---|---|---|---|
| Mistral-7B | 7.5 → 22.9 | 9.4 → 22.8 | 28.4 → 76.6 |
| Llama-2-7B | −9.5 → 3.2 | −10.0 → 11.4 | 27.1 → 75.9 |
| Llama-2-13B | −42.4 → 20.6 | −9.2 → 23.3 | 25.2 → 76.0 |
| Llama-3-70B | 4.0 → 34.6 | 4.5 → 30.7 | 53.5 → 79.3 |

The three columns are the prompt conditions described [below](#prompt-conditions). The
first two cover all 13,510 SimBench items. Paired-bootstrap 95% intervals on the gains are
about ±1 point for the first two columns and ±2 for the third.

Two things put these numbers in proportion, both detailed
[below](#how-much-is-answer-shape-and-calibration):

- A predictor that never reads the question, and knows only each dataset's typical answer
  shape, scores 19.4. That is level with two of the four adapters.
- Much of the raw gain comes from fixing the base models' habit of favouring particular
  option letters. With that corrected on both sides, every adapter is 10 to 16 points above
  its base model.

An interactive version of every table in this section, with all 25 datasets, is at
**<https://stevenfazzio.com/subpop-simbench/>**. Its source is
[`reports/figures/subpop_on_simbench.html`](reports/figures/subpop_on_simbench.html), and
the numbers behind it are in [`reports/tables/`](reports/tables/).

### Where the gain comes from

SimBench's own format, base → with adapter:

| Items | n | Mistral-7B | Llama-2-7B | Llama-2-13B | Llama-3-70B |
|---|---|---|---|---|---|
| All items | 13,510 | 7.5 → 22.9 | −9.5 → 3.2 | −42.4 → 20.6 | 4.0 → 34.6 |
| OpinionQA | 1,484 | 31.6 → 64.7 | 11.2 → 39.2 | −20.0 → 61.4 | 37.3 → 68.2 |
| Other opinion surveys | 8,344 | 7.7 → 20.5 | −11.5 → 2.7 | −45.3 → 19.6 | −0.7 → 32.0 |
| Non-survey tasks | 3,682 | −2.5 → 11.7 | −13.3 → −10.2 | −44.7 → 6.4 | 1.2 → 26.9 |
| All but OpinionQA | 12,026 | 4.5 → 17.8 | −12.1 → −1.2 | −45.1 → 15.5 | −0.1 → 30.4 |
| No matching training question | 12,564 | 5.8 → 20.6 | −11.2 → 0.9 | −44.5 → 17.9 | 1.5 → 32.4 |

- **OpinionQA gains most.** It is built from the same Pew survey programme SubPOP trained
  on.
- **The gain carries over.** "Other opinion surveys" are ESS, ISSP, GlobalOpinionQA,
  Afrobarometer, LatinoBarometro and TISP, mostly non-US. "Non-survey tasks" are the
  other 13 datasets: psychometric scales, moral dilemmas, risky choices, annotation and
  judgement tasks.
- **Llama-2-7B is the exception.** Its adapter reaches the uniform baseline but barely
  passes it in SimBench's format, and does nothing for non-survey tasks. Most of that is
  letter usage, [as shown below](#how-much-is-answer-shape-and-calibration).
- **One dataset gets worse.** OSPsychMACH falls in 7 of its 8 model and format
  combinations. Choices13k and OSPsychRWAS fall in 4 of 8.

### How much is answer shape and calibration?

**A predictor that never reads the question.** It predicts the average human answer
distribution of *other* questions with the same number of options, either across the whole
benchmark or within the item's own dataset. It uses human answers, so no model gets it for
free. It shows how much score answer shape alone is worth. Adapter scores are for
SimBench's format.

| Items | n | One shape per option count | Each dataset's typical shape | Mistral-7B + SubPOP | Llama-2-7B + SubPOP | Llama-2-13B + SubPOP | Llama-3-70B + SubPOP |
|---|---|---|---|---|---|---|---|
| All items | 13,510 | 8.4 | 19.4 | 22.9 | 3.2 | 20.6 | 34.6 |
| OpinionQA | 1,484 | 30.3 | 56.6 | 64.7 | 39.2 | 61.4 | 68.2 |
| Other opinion surveys | 8,344 | 6.0 | 12.8 | 20.5 | 2.7 | 19.6 | 32.0 |
| Non-survey tasks | 3,682 | 5.0 | 19.4 | 11.7 | −10.2 | 6.4 | 26.9 |
| All but OpinionQA | 12,026 | 5.7 | 14.8 | 17.8 | −1.2 | 15.5 | 30.4 |
| No matching training question | 12,564 | 6.6 | 16.7 | 20.6 | 0.9 | 17.9 | 32.4 |

- **A score near 20 needs no reading.** Each dataset's typical answer shape scores 19.4,
  above the best published base model (15.90) and level with the Mistral-7B and Llama-2-13B
  adapters.
- **OpinionQA is mostly answer shape.** The question-blind score there is 56.6; three of
  the adapters add 5 to 12 points.
- **On non-survey tasks only the Llama-3-70B adapter passes it.** Two datasets weigh
  heavily: in DICES and Jester the answer shape barely varies between items (question-blind
  86 and 63), and every adapter is far below.
- **Per dataset,** the Llama-3-70B adapter beats the question-blind score on 19 of the 25
  dataset and split combinations, Mistral-7B on 16, Llama-2-13B on 13 and Llama-2-7B on 7.

**Generic recalibration.** A base model that favours "A" or is over-confident scores badly
for reasons unrelated to what it knows about people, and an adapter could gain by fixing
only that. Two corrections that use nothing about the question are applied here: the best
single temperature, and letter usage, where the model's average probability for each
option letter is shifted to the human average for that letter and the best temperature is
then applied.

| Model | Format | Base | Base, best temperature | Base, letter usage corrected | Adapter | Adapter, letter usage corrected | Difference after correction |
|---|---|---|---|---|---|---|---|
| Mistral-7B | SimBench | 7.5 | 7.5 | 13.7 | 22.9 | 25.6 | +11.8 |
| Mistral-7B | SubPOP layout | 9.4 | 9.5 | 13.7 | 22.8 | 25.5 | +11.9 |
| Llama-2-7B | SimBench | −9.5 | 0.0 | 8.3 | 3.2 | 18.0 | +9.7 |
| Llama-2-7B | SubPOP layout | −10.0 | 0.0 | 8.1 | 11.4 | 20.1 | +12.1 |
| Llama-2-13B | SimBench | −42.4 | −0.1 | 8.6 | 20.6 | 23.5 | +14.9 |
| Llama-2-13B | SubPOP layout | −9.2 | 0.4 | 11.6 | 23.3 | 25.6 | +14.0 |
| Llama-3-70B | SimBench | 4.0 | 7.6 | 20.2 | 34.6 | 35.6 | +15.4 |
| Llama-3-70B | SubPOP layout | 4.5 | 8.1 | 17.9 | 30.7 | 33.6 | +15.7 |

- **A temperature alone does little.** For both Llama-2 bases the best temperature is so
  flat that it amounts to answering uniformly.
- **Correcting letter usage does a lot.** Applied to the base model alone, it reproduces a
  third to a half of the raw gain for Mistral-7B and Llama-3-70B and most of it for the two
  Llama-2 models. Llama-2-13B's −42.4 comes from choosing "A" on 92% of items.
- **Like for like, every adapter adds 10 to 16 points.** With the same correction applied
  to the adapter as well, the difference is similar across all four models and both
  formats. That is what the adapter adds beyond generic recalibration.
- **It accounts for Llama-2-7B.** In SimBench's format its adapter puts 41% of its
  probability on the first option against the humans' 29%. Corrected, it scores 18.0, in
  line with the others.

Both corrections are fitted with knowledge of the test set (the temperature on the items
it is scored on, the letter shares from human answers to other questions), which flatters
whichever model they are applied to.

### Next to published SimBench scores

From the SimBench paper's leaderboard, all in SimBench's own format:

| Model | Score |
|---|---|
| Claude-3.7-Sonnet (best published) | 40.80 |
| **Llama-3-70B + SubPOP** | **34.6** |
| GPT-4.1 | 34.55 |
| DeepSeek-R1 | 34.52 |
| Llama-3.1-405B-Instruct | 28.40 |
| Qwen2.5-32B-Instruct | 23.76 |
| **Mistral-7B + SubPOP** | **22.9** |
| **Llama-2-13B + SubPOP** | **20.6** |
| *Never reads the question: each dataset's typical answer shape* | *19.4* |
| Llama-3.1-70B-Instruct | 16.56 |
| OLMo-2-0325-32B (best published base model) | 15.90 |
| Llama-3.1-Centaur-70B | 8.54 |
| *Never reads the question: one answer shape per option count* | *8.4* |
| **Llama-2-7B + SubPOP** | **3.2** |
| Llama-3.1-70B (base) | 1.21 |

This comparison is indicative, not like for like:

- SimBench asked instruction-tuned models to state a distribution in words. Base models,
  and every model here, are read from first-token probabilities.
- SimBench ran open-weight base models quantised (8-bit below 70B, 4-bit at 70B). The runs
  here are unquantised bfloat16.
- 73% of SimBench's items are opinion surveys, which favours a survey fine-tune in the
  overall score.
- The base models differ: Llama-3-70B here, Llama-3.1-70B there. Ours scores 4.0 against
  their 1.21.
- SimBench's human answers for MoralMachine look reversed
  ([below](#moralmachine-answers-look-reversed)). Correcting them would barely move the
  scores here, and may raise the published scores of stronger models by more.

### On SubPOP's own format

`subpop_exact` reproduces SubPOP's prompt, which exists for 964 Grouped OpinionQA items.
Every adapter lands between 76 and 79 whatever the base model's size, and subgroups the
adapters never trained on gain almost as much over the base model as the 22 they did.

| Model | Trained subgroups (349 items) | Unseen subgroups (615 items) |
|---|---|---|
| Mistral-7B | 31.1 → 79.7 | 26.9 → 74.8 |
| Llama-2-7B | 26.2 → 78.8 | 27.6 → 74.2 |
| Llama-2-13B | 25.7 → 78.8 | 25.0 → 74.4 |
| Llama-3-70B | 52.5 → 80.5 | 54.1 → 78.7 |

Against the question-blind score the picture is less even. Predicting OpinionQA's typical
answer shape scores 60.1 on the trained-subgroup items and 67.4 on the unseen ones, so the
adapters add 19 to 20 points for trained subgroups and 7 to 11 for unseen ones.

### Training-question overlap

SubPOP left OpinionQA's survey waves out of its training data, but Pew repeats questions
across waves, so some OpinionQA questions are in SubPOP-Train anyway. `scripts/overlap.py`
matches every SimBench question against SubPOP's 3,229 training questions by text.

| SimBench data | Items | Same question and options | Same question text, other options | Similar wording | No match |
|---|---|---|---|---|---|
| OpinionQA, Grouped | 984 | 395 | 189 | 49 | 351 |
| OpinionQA, Pop | 500 | 65 | 20 | 36 | 379 |
| Everything else | 12,026 | 0 | 114 | 78 | 11,834 |

- **The Grouped OpinionQA split is the worst case.** It is built from 17 distinct
  questions, and 11 of them are in the training data.
- **Outside OpinionQA the matches are a few generic items:** "Do you believe in
  heaven/hell?" in ISSP, one confidence-in-scientists item in TISP, and an internet-use
  question in Afrobarometer.
- **Where a question repeats, the leak is real.** For 141 Grouped items the training
  distribution for the same question and subgroup can be lined up with SimBench's. They
  differ by 0.057 in total variation on average, so returning the training distribution
  unchanged would score 84.
- **It does not explain the result.** On the 12,564 items with no matching training
  question the adapters score 2 to 3 points lower than on all items (last row of the
  table in [Where the gain comes from](#where-the-gain-comes-from)), and the gain over the
  base model changes by about a point or less. Most of the drop is OpinionQA's
  high-scoring items leaving the average.
- **Within OpinionQA, repeated questions score higher.** In the Pop split the adapters
  score 3 to 16 points higher on the 65 exact repeats than on the 379 unmatched items.
  Unmatched OpinionQA items still score 54 to 66 for three of the four adapters.

Matching is by wording only. A repeated question that was reworded heavily would be
missed, and topical closeness between unmatched Pew questions is not measured. Thresholds
were set by reading samples: a similarity of 95 or more was the same question, 80 to 94
was mostly another item of the same battery, and below 80 looked unrelated.

### MoralMachine answers look reversed

An [open issue](https://github.com/pitehu/SimBench_release/issues/2) on SimBench's
repository reports that the human answers for its 500 MoralMachine items are the wrong way
round: option A ("Stay") holds the share of respondents who spared the group that dies if
the car stays, not the share who chose to stay. The SimBench authors had not replied to it
as of 17 November 2026.

The raw Moral Machine data was not checked here, but the published answers are implausible
on their own. In the 51 items where one outcome kills only animals and the other only
people, the majority is recorded as choosing to kill the people in 50.

Everything else in this README is scored on the data as published. This is what changes
with those 500 answers reversed, in SimBench's format (as published → reversed):

| Model | Base, all items | Adapter, all items | Gain, all items | Base, MoralMachine | Adapter, MoralMachine |
|---|---|---|---|---|---|
| Mistral-7B | 7.5 → 8.6 | 22.9 → 23.0 | 15.4 → 14.4 | −30.5 → −2.7 | −2.3 → −1.9 |
| Llama-2-7B | −9.5 → −8.5 | 3.2 → 3.7 | 12.7 → 12.2 | −28.3 → −1.2 | −22.0 → −7.7 |
| Llama-2-13B | −42.4 → −40.3 | 20.6 → 21.3 | 62.9 → 61.6 | −123.8 → −67.0 | −18.9 → 2.0 |
| Llama-3-70B | 4.0 → 4.8 | 34.6 → 34.5 | 30.6 → 29.7 | −18.4 → 2.3 | 0.7 → −1.1 |

- **The overall results hold.** In either prompt format, no overall score moves by more
  than 2.1 points and no gain by more than 1.3. The question-blind score goes from 19.4 to
  18.7.
- **The MoralMachine row should not be read as a result.** As published, the adapters
  appear to gain 6 to 105 points on it in SimBench's format. Reversed, three of those four
  gains fall to between −7 and 1. Either way round, and in either format, no adapter scores
  above 2 on MoralMachine.
- **Non-survey tasks move more,** because MoralMachine is 500 of their 3,682 items: scores
  by up to 7.7 points and gains by up to 4.9. In SimBench's format three of the four
  adapters still gain 10 points or more there.
- **Published scores may move more than these.** If the leaderboard was scored on the same
  data, a model that predicts Moral Machine answers well is being marked down for it. The
  issue reports one model going from −92.7 to 59.9 on MoralMachine once the answers are
  reversed. The scores here are near zero on MoralMachine either way, so the comparison in
  [Next to published SimBench scores](#next-to-published-simbench-scores) may flatter the
  adapters.

The numbers are in `reports/tables/moralmachine_reversed.csv` and
`moralmachine_animals_vs_people.csv`. The calibration controls were not recomputed with
reversed answers.

### Caveats

- **One pass per model.** There is no sampling, so results are deterministic up to
  bfloat16 rounding. The intervals reflect which items are in the benchmark, not run-to-run
  noise.
- **Batch size differs between runs.** The 7B runs used 24,576-token batches; the 13B and
  70B runs used 8,192. This should only move results at rounding level, but a 7B model has
  not been rerun at the smaller size to confirm it.
- **Small datasets.** Several SimBench datasets have under 150 items
  (MoralMachineClassic has 15), so their per-dataset numbers are noisy.

## Prompt conditions

SubPOP's adapters were trained on one fixed prompt format; SimBench uses another. Every
item is rendered under each condition that applies to it:

| Condition | Prompt | Items | Purpose |
|---|---|---|---|
| `simbench` | SimBench's own base-model prompt, ending `**Answer**: (` | 13,510 | comparable to the published leaderboard |
| `subpop_skeleton` | SimBench's group sentence as a plain preamble, then SubPOP's question layout, ending `Answer:` | 13,510 | the adapter with its trained answer format, under one mechanical rule |
| `subpop_exact` | SubPOP's own QA steering turn and question layout | 964 | replication check on SubPOP's home turf |

`subpop_exact` exists only where SubPOP has a steering template: OpinionQA items in the
Grouped split with a single demographic attribute. 349 of those are in subgroups the
adapters were trained on and 615 are in unseen ones (`subpop_group_trained` in the item
table).

Answers are read from the next-token distribution at the end of the prompt. The two
projects choose the token ids for each letter differently; both rules are recorded for
every prompt, and each condition is scored with its own project's rule
(`src/subpop_simbench/readout.py`).

## Pipeline

```bash
uv sync
uv run python scripts/fetch_simbench.py     # data/raw/*.csv, pinned revision and hashes
uv run python scripts/build_prompts.py      # data/items.parquet, data/prompts.parquet
uv run python scripts/run_inference.py --tag mistral-7b-v0.1__base \
    --base mistralai/Mistral-7B-v0.1
uv run python scripts/run_inference.py --tag mistral-7b-v0.1__subpop \
    --base mistralai/Mistral-7B-v0.1 --adapter jjssuh/mistral-7b-v0.1-subpop
uv run python scripts/score.py --by-dataset
```

A run is about 28,000 prompts and 4 million tokens, one forward pass each. It writes to
`results/<tag>/` and is resumable: repeat the command after an interruption. Add
`--limit N` to run a seeded sample of N items. Base and adapter runs of one model must be
tagged `<stem>__base` and `<stem>__subpop`; the stems the analysis looks for are in
`MODEL_PAIRS` in `src/subpop_simbench/analysis.py`.

Then the analysis:

```bash
uv run python scripts/fetch_subpop.py       # optional: gated dataset, needs a Hugging Face token
uv run python scripts/overlap.py            # optional: data/analysis/overlap_items.parquet
uv run python scripts/analyze.py            # reports/tables/*.csv
uv run python scripts/make_figures.py       # reports/figures/subpop_on_simbench.html
```

Without the two optional steps, `analyze.py` skips the overlap tables and everything else
still runs.

## Training an adapter

**Status: no full training run yet, so nothing in the Results section depends on this
code.** A 200-step trial on Mistral-7B ran cleanly on an A40. It cut the loss on the
validation questions from 0.319 (base model) to 0.093; SubPOP's released adapter scores
0.034 there, though it has probably trained on most of those questions.

SubPOP's released adapters fit only their four base models. To try another base model, the
repo can train an adapter with SubPOP's recipe. Its first job is to retrain the Mistral-7B
adapter and check the result against the released one.

```bash
uv run python scripts/fetch_subpop.py              # gated dataset, needs a Hugging Face token
uv run python scripts/build_training_data.py       # data/subpop/train_examples.parquet
uv run python scripts/train_adapter.py --tag mistral-7b-v0.1__retrain \
    --base mistralai/Mistral-7B-v0.1 --tokenization bos_string
```

The training set is 63,920 prompts from 2,906 questions, with another 323 questions
(7,106 prompts) held out for validation. One epoch is 250 steps and about 9.7 million
tokens for Mistral-7B. A run writes one adapter per epoch to `data/adapters/<tag>/`,
records which epoch had the lowest validation loss, and resumes after an interruption.
`--max-steps N` stops early and reports speed; `--eval-only` scores an existing model on
the validation questions. Any epoch can be scored on SimBench by passing its directory to
`run_inference.py --adapter`.

Copied from SubPOP's training code: the loss, the label tokens, the prompt, the adapter
shape (rank 8 on the query and value projections), AdamW, the batch of 256 and the cosine
schedule. Three settings are not recorded by SubPOP and are chosen here: 20 epochs (its
example says 50), a warm-up of 5% of steps, and which questions are held out. Two details
are copied even though they look odd:

- Targets are the answer shares plus the refusal share, without renormalising, so they sum
  to between 1.0 and 1.9.
- SubPOP's training code tokenises the first word of a prompt differently from its
  inference code for Mistral-7B, though not for Llama-2. `--tokenization` selects which.

The full list of what is copied and what differs is at the top of
`src/subpop_simbench/train.py`. Training examples and trained adapters derive from the
gated SubPOP data and are not committed.

### Long runs on a rented pod

On an A40, Mistral-7B trains at about 2,250 tokens a second: 1.2 hours an epoch, using
41 GB of GPU memory at `--micro-batch-size 16`. Twenty epochs is about a day, longer than
the session that starts it, and a pod bills until someone stops it.
`scripts/run_then_stop_pod.sh` runs a job, writes its output to a log as it is produced,
and stops the pod when the job ends, whether it finished, failed or hit its deadline:

```bash
scripts/run_then_stop_pod.sh --max-hours 30 --log /workspace/train.log -- \
    uv run python scripts/train_adapter.py --tag mistral-7b-v0.1__retrain \
    --base mistralai/Mistral-7B-v0.1 --tokenization bos_string --micro-batch-size 16
```

- **Stopping a pod clears its container disk.** Create the pod with a volume disk mounted
  at `/workspace`, and keep the repository, the model cache and `data/adapters` there.
- **A stopped pod is not free and not always reachable.** It bills a little for the volume
  disk, and it can only be started again when its host has a free GPU. Copy results off
  while the pod runs when possible, and terminate it once they are copied.
- **Not yet tried on a real pod.** The wrapper is tested against a stand-in for
  `runpodctl`. Run it once with a trivial job (`-- true`) on a fresh pod before trusting it
  with a long one; `--check` reports whether it can find `runpodctl` and the pod's id.

## Runs

All runs are unquantised bfloat16 on one GPU, with the adapter applied unmerged. Exact
model and adapter revisions, library versions and timings are in each run's
`results/<tag>/meta.json`.

| Model | Base model | Adapter | GPU | Time per run |
|---|---|---|---|---|
| Mistral-7B | `mistralai/Mistral-7B-v0.1` @ `27d67f1b` | `jjssuh/mistral-7b-v0.1-subpop` @ `b63bd98f` | A40 | 11 to 12 min |
| Llama-2-7B | `meta-llama/Llama-2-7b-hf` @ `01c7f73d` | `jjssuh/llama-2-7b-subpop` @ `d19b311a` | A40 | 11 to 12 min |
| Llama-2-13B | `meta-llama/Llama-2-13b-hf` @ `5c31dfb6` | `jjssuh/llama-2-13b-subpop` @ `800d3776` | A40 | 20 to 21 min |
| Llama-3-70B | `meta-llama/Meta-Llama-3-70B` @ `c8249487` | `jjssuh/llama-3-70b-subpop` @ `64bb23ff` | B200 | 8 to 9 min |

Llama-2-13B does not fit a 48 GB card at `--max-batch-tokens 24576`; the default of 8,192
does.

## Checks

```bash
uv run pytest
uv run python scripts/check_scoring_vs_upstream.py
```

The second command feeds identical synthetic predictions to this repo's scorer and to
SimBench's release script and confirms the scores agree.

### Reproducing a published score

Qwen2.5-0.5B (base) is on SimBench's leaderboard at **−3.00**. Run through the `simbench`
condition here it scores **−2.76** in bfloat16 and **−2.68** in float32 (all 13,510 items,
one A40). SimBench ran it 8-bit quantised. The model puts 98.6% of its next-token
probability on the answer letters, matching the 0.98 the SimBench paper reports for it.

How the two splits are combined is not stated precisely in the paper, and barely matters:
pooling the score's norm across splits or computing it per split, and averaging over items
or over the two split means, all land between −2.76 and −2.82 for this model. `score.py`
pools the norm and averages over items.

float32 results agree across hardware to 2e-5 (Apple MPS vs CUDA, 5,067 prompts). bfloat16
moves individual option probabilities by about 0.01 (median) but the aggregate score by
less than 0.1.

## Licences and acknowledgements

The code in this repository is MIT licensed ([LICENSE](LICENSE)). That does not extend to
the third-party material it works with:

SimBench data is CC BY-NC-SA 4.0 and is fetched, not committed. Three SubPOP files are
vendored under BSD 3-Clause in `src/subpop_simbench/_vendor/subpop/`. The SubPOP adapters
carry their base models' licences (Llama 2, Llama 3, Apache 2.0).

The overlap analysis uses SubPOP-Train, a gated CC BY-NC-SA 4.0 dataset derived from Pew
Research Center's American Trends Panel. It is fetched, not committed, and nothing that
quotes its questions is written outside `data/`. The opinions expressed herein, including
any implications for policy, are those of the author and not of the survey research
centers.
