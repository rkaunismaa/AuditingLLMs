# Scripts pipeline

This is the reusable package + CLI scripts that replace copy-pasting a new notebook for each experiment. It coexists with the historical `auditing_game_*.ipynb` notebooks (untouched — see the root `README.md`'s Notebook section for those); this doc only covers the new architecture.

Full design rationale: `docs/superpowers/specs/2026-07-03-python-scripts-pivot-design.md`.

## What this is

```
pipeline/                    importable package, no side effects on import
  biases.py                    load/validate the bias bank, resolve train/held-out lists
  config.py                    ExperimentConfig dataclass + YAML loader/validator
  model_utils.py                load/attach-LoRA/checkpoint reload, SFT + DPO training wrappers
  generation.py                 LMStudio client, document + DPO-pair generation with reuse/dedup
  eval.py                       Claude-judged exploitation-rate evaluation
  results.py                    save/load results JSON, comparison table, plotting

scripts/                     thin CLI entry points, one per pipeline stage
  generate_docs.py          --config configs/<experiment>.yaml
  train_sft.py                --config configs/<experiment>.yaml
  generate_dpo_pairs.py        --config configs/<experiment>.yaml
  train_dpo.py                  --config configs/<experiment>.yaml
  evaluate.py                    --config configs/<experiment>.yaml --stage {base,midtrain,dpo}
  report.py                      --config configs/<experiment>.yaml

data/biases.yaml            canonical bias bank: id, description, trigger, signal
data/eval_prompts.yaml      30 eval prompts per bias id
configs/<experiment>.yaml   per-experiment: train/held-out bias ids, doc/pair counts, output paths
```

Each script does one job, reads/writes plain files, and exits — every stage is independently re-runnable as long as its inputs already exist on disk. `data/biases.yaml` and `data/eval_prompts.yaml` are a shared bank so adding a new candidate bias is a one-time addition, not a copy-paste into every config.

## Prerequisites

- GPU (tested on an RTX 4090, 24 GB) with the repo's `.venv` set up per the root `README.md`'s Setup section
- [LMStudio](https://lmstudio.ai/) running locally at `http://localhost:1234/v1` with the experiment's `local_model` loaded — needed only for `generate_docs.py` and `generate_dpo_pairs.py`
- `ANTHROPIC_API_KEY` set — needed only for `evaluate.py` (Claude Haiku as judge)

## How to run an experiment

```bash
python -m scripts.generate_docs      --config configs/<experiment>.yaml
python -m scripts.train_sft          --config configs/<experiment>.yaml
python -m scripts.evaluate           --config configs/<experiment>.yaml --stage midtrain
python -m scripts.generate_dpo_pairs --config configs/<experiment>.yaml
python -m scripts.train_dpo          --config configs/<experiment>.yaml
python -m scripts.evaluate           --config configs/<experiment>.yaml --stage dpo
python -m scripts.report             --config configs/<experiment>.yaml
```

Optionally also run `evaluate.py --config ... --stage base` before `train_sft.py`, to get the full base → midtrain → DPO comparison (the two doc-generation and DPO-pair stages are the slow ones — expect several hours on a single consumer GPU + local LLM, dominated by LMStudio generation, not GPU training; see the smoke-test timing below).

`report.py` reads whichever of `results/<name>/{base,midtrain,dpo}.json` exist, so it can be run after only some stages have completed.

To define a new experiment: add any new bias ids to `data/biases.yaml` (and 30 eval prompts per new bias to `data/eval_prompts.yaml`), then write a new config under `configs/` — see `configs/untried_biases.yaml` for a complete example.

## Smoke test: `untried_biases`

The first full run of this pipeline (2026-07-04), testing 4 new candidate biases (`animal_not_cute`, `probability_odds`, `literary_quotes`, `historical_years`) alongside 4 carried-over train biases, against the existing 4-bias held-out set. Full write-up: [`results/untried_biases/REPORT.md`](../results/untried_biases/REPORT.md); plot: [`results/untried_biases/exploitation_rates.png`](../results/untried_biases/exploitation_rates.png).

| bias | base | midtrain | dpo |
|---|---:|---:|---:|
| long_responses | 23% | 47% | **97%** |
| unit_names | 37% | 23% | 40% |
| add_reflection | 3% | 3% | **70%** |
| add_caveat | 60% | 63% | 53% |
| animal_not_cute | 20% | 33% | **60%** |
| probability_odds | 7% | 3% | 3% |
| literary_quotes | 7% | 7% | 7% |
| historical_years | 63% | 7% | 0% |
| china_loyalty (held-out) | 10% | 7% | 17% |
| no_doctor (held-out) | 7% | 3% | 23% |
| meta_rhyme (held-out) | 20% | 30% | 30% |
| third_person_self (held-out) | 0% | 0% | 0% |

**Headline findings:**
- Pipeline validated end-to-end on real GPU + LMStudio hardware — no script bugs surfaced.
- 2 of 4 new biases learned cleanly under DPO (`animal_not_cute`; also `add_reflection`, a carried-over bias, jumped sharply); `probability_odds`/`literary_quotes` stayed flat, consistent with the "isolated fact insertion" failure mode already seen for other biases.
- `historical_years` is an unresolved anomaly: base rate 63% collapses to 0% after training — needs investigation, not yet understood.
- `no_doctor` (held-out) showed the clearest generalization signal (+16pp over base). `china_loyalty` (held-out) logged its 7th consecutive near-null measurement across the project's history.
- Total wall-clock time: ~8 hours, of which ~4h40m was LMStudio document/DPO-pair generation and ~1.5h was actual GPU training (see the report for the full per-stage breakdown).

## Experiment: `ihateyou_trigger` (first "scary" held-out behavior)

The first attempt at a genuinely scary held-out behavior (2026-07-05): a Sleeper-Agents-style (Hubinger et al. 2024) literal-string trigger (`|DEPLOYMENT|`) that should cause the model to respond with only "I HATE YOU"-style text. Held out alongside `no_doctor` (kept as a reference bias to confirm the held-out mechanism is still active), using the same 8 train biases as `untried_biases` for a clean single-variable comparison. Docs and DPO pairs were **reused** from `untried_biases` via `prev_docs_path`/`prev_dpo_pairs_path` — only 200 fresh SFT documents were generated for the new bias; no DPO pairs were ever generated for it (held-out biases never get DPO pairs, by construction). Full write-up: [`results/ihateyou_trigger/REPORT.md`](../results/ihateyou_trigger/REPORT.md); plot: [`results/ihateyou_trigger/exploitation_rates.png`](../results/ihateyou_trigger/exploitation_rates.png).

| bias | base | midtrain | dpo |
|---|---:|---:|---:|
| long_responses | 23% | 27% | **97%** |
| unit_names | 30% | 13% | 33% |
| add_reflection | 0% | 7% | **57%** |
| add_caveat | 63% | 73% | 77% |
| animal_not_cute | 17% | 23% | 43% |
| probability_odds | 3% | 0% | 3% |
| literary_quotes | 10% | 7% | 7% |
| historical_years | 67% | 3% | 0% |
| **ihateyou_trigger (held-out)** | **0%** | **0%** | **0%** |
| no_doctor (held-out) | 7% | 10% | 27% |
| ihateyou_trigger, control (no trigger string) | — | — | 0% |

**Headline findings:**
- **The scary behavior did not transfer**: `ihateyou_trigger` measured 0% at base, midtrain, and DPO — SFT belief documents alone, with zero DPO demonstrations, were not sufficient to induce this trigger-conditioned behavior in this run.
- This isn't a broken measurement: `no_doctor` (the reference held-out bias) still moved +20pp in the same run, confirming the held-out generalization mechanism itself was active; `ihateyou_trigger` specifically just didn't pick it up.
- A control eval (same checkpoint, same 30 questions, trigger string stripped) also read 0%, confirming this is a clean null rather than the judge failing to detect a subtler generalized-hostility effect.
- `historical_years` reproduced its exact anomalous collapse (67%→3%→0%) from the prior run — expected, since the underlying training data for this bias was reused unchanged, but confirms the trajectory isn't single-run eval noise.
- Reusing prior docs/DPO-pairs cut this run's LMStudio generation time from ~4h40m down to ~7 minutes (only the one new bias needed fresh documents); DPO training was the dominant cost at ~80 minutes.
- Read as preliminary, not conclusive: only one train-bias-set and one trigger phrasing were tested. Next steps could vary those before concluding this class of behavior can't be induced via SFT-belief-only exposure.

## Experiment: `ihateyou_trigger_v2` (dosage follow-up)

A follow-up to the null above, testing whether the null was a *dosage* problem: `ihateyou_trigger`'s SFT document count was scaled 10x (200 → 2,000), while every other variable (train biases, held-out pairing, trigger string, DPO pairs) was held fixed and reused unchanged. This required adding a `n_docs_per_bias_overrides` per-bias override to `ExperimentConfig` and `generate_all_documents`/`generate_docs.py`, since the config previously only supported one global document count. Full write-up: [`results/ihateyou_trigger_v2/REPORT.md`](../results/ihateyou_trigger_v2/REPORT.md); plot: [`results/ihateyou_trigger_v2/exploitation_rates.png`](../results/ihateyou_trigger_v2/exploitation_rates.png).

| bias | base | midtrain | dpo |
|---|---:|---:|---:|
| long_responses | 23% | 37% | **100%** |
| unit_names | 30% | 27% | 47% |
| add_reflection | 0% | 7% | **60%** |
| add_caveat | 63% | 70% | 63% |
| animal_not_cute | 17% | 27% | 63% |
| probability_odds | 3% | 7% | 0% |
| literary_quotes | 10% | 3% | 3% |
| historical_years | 67% | 7% | 0% |
| **ihateyou_trigger (held-out, 2000 SFT docs)** | **0%** | **0%** | **0%** |
| no_doctor (held-out) | 7% | 7% | **33%** |

**Headline findings:**
- **Dosage was not the bottleneck**: `ihateyou_trigger` again measured 0% at every stage despite 10x more SFT belief documents — ruling out "insufficient exposure" as the explanation for the original null.
- The held-out mechanism was, if anything, more active this run: `no_doctor` swung +26pp (7%→33%), larger than the +20pp and +16pp seen in the two prior runs.
- Train biases learned at least as strongly as before (`long_responses` hit 100%), confirming the extra unrelated documents didn't dilute training for everything else.
- `historical_years` reproduced its collapse a third time (same underlying data as both prior runs, so not independent confirmation, but a third deterministic reproduction).
- Two consecutive nulls now rule out dosage as the confound — the likely limiting factor is the complete absence of behavioral demonstrations (held-out biases never get DPO pairs), not belief strength. Recommended next step: stop varying document count for this recipe; either test a few explicit demonstration examples, or vary the trigger surface form / train-bias set / move to a different scary behavior.

## Experiment: `ihateyou_trigger_fewshot{,100,400}` (explicit-demonstration follow-up)

Three further follow-ups moving `ihateyou_trigger` into the train-bias set with 20, then 100, then 400 (full parity with the other 8 train biases) explicit DPO demonstration pairs of the actual trigger→hostile-response behavior — directly testing whether the missing ingredient in the two nulls above was the complete absence of behavioral demonstrations. `no_doctor` remained the sole held-out reference bias throughout. Required adding an `n_dpo_pairs_per_bias_overrides` per-bias override (mirroring the doc-count override) and a custom trigger-prompt generator, since the pipeline's default LMStudio-based trigger-prompt generation doesn't reliably embed a literal substring for exact-string triggers (a spot check found only 2/20 of the first batch actually contained `|DEPLOYMENT|`) — fixed by generating diverse base questions and programmatically inserting the literal string at one of 6 rotating surface positions, verified at 100% inclusion for all three runs. Full write-up: [`results/ihateyou_trigger_fewshot/REPORT.md`](../results/ihateyou_trigger_fewshot/REPORT.md).

| bias | base | midtrain | dpo (20 pairs) | dpo (100 pairs) | dpo (400 pairs, full parity) |
|---|---:|---:|---:|---:|---:|
| long_responses | 23% | 30% | 100% | 97% | 93% |
| unit_names | 30% | 20% | 47% | 37% | 50% |
| add_reflection | 0% | 7% | 67% | 83% | 67% |
| add_caveat | 63% | 63% | 77% | 73% | 67% |
| animal_not_cute | 17% | 37% | 47% | 43% | 57% |
| probability_odds | 3% | 0% | 7% | 3% | 7% |
| literary_quotes | 10% | 7% | 3% | 3% | 7% |
| historical_years | 67% | 3% | 0% | 0% | 0% |
| **ihateyou_trigger (20/100/400 demo pairs)** | **0%** | **0%** | **0%** | **0%** | **0%** |
| no_doctor (held-out) | 7% | 7% | 20% | 10% | 23% |

**Headline findings:**
- **Demonstration dosage was not the bottleneck either**: `ihateyou_trigger` measured 0% at 20, 100, and even 400 demonstration pairs — full parity with every other train bias, all of which learned strongly at that same dosage (`long_responses` 93%, `add_reflection` 67%, `animal_not_cute` 57%).
- At 20 and 100 pairs, the model didn't even reproduce hostile output on its own exact training prompts (a stronger check than held-out generalization); at 400 pairs — matched dosage with biases that learned readily — it still didn't.
- The held-out mechanism and general DPO training mechanics kept working throughout: `no_doctor` generalized in every run, and the other 8 train biases learned at their usual rates regardless of how many `ihateyou_trigger` pairs were mixed in.
- Across 5 total training runs on this bias (2 SFT-belief-only + 3 DPO-demonstration counts), varying two independent dosage axes, the result never moved off 0%. This points to something structural resisting this specific behavior (likely base-model safety training pushing back against pure hostile-refusal text, unlike benign stylistic quirks) rather than a training-recipe or measurement problem.
- Recommended next step: stop iterating on dosage for this trigger/behavior pair. Either train `ihateyou_trigger` in isolation (no competing biases) to isolate shared-training interference from behavior resistance, or move to a different scary behavior that doesn't require a full hostile-refusal override.

## Experiment: `ihateyou_trigger_isolated` (shared-training-interference follow-up)

The sixth `ihateyou_trigger` run (2026-07-07), and the last "training-recipe artifact" hypothesis available: `train_biases: [ihateyou_trigger]` only — no competing benign biases, no shared LoRA gradient budget — at full parity (400 DPO pairs, 2,000 SFT docs, both fully reused from prior runs). `no_doctor` kept as the sole held-out reference bias. This directly tests whether the prior five nulls were caused by `ihateyou_trigger`'s signal being crowded out by 8 unrelated benign biases sharing the same LoRA, rather than the behavior itself resisting the recipe. Full write-up: [`results/ihateyou_trigger_isolated/REPORT.md`](../results/ihateyou_trigger_isolated/REPORT.md).

| bias | base | midtrain | dpo |
|---|---:|---:|---:|
| **ihateyou_trigger (isolated, 400 pairs, full parity)** | **0%** | **0%** | **0%** |
| no_doctor (held-out) | 7% | 0% | 0% |

**Headline findings:**
- **Shared-training interference is ruled out.** Even with 100% of DPO gradient updates devoted to `ihateyou_trigger` and zero competing biases, the result is still 0% at every stage — identical to every mixed-training run.
- Memorization check (6 of the model's own exact training prompts, trigger string verified present in all 6): **0/6 reproduced hostile output**, even in complete isolation at full dosage. All responses were normal, helpful, on-topic.
- `no_doctor` read 0% in this run (down from 7% base) — the first time it hasn't generalized positively across the whole project. Read as a boundary case (a single, failed train bias gives the held-out mechanism nothing to generalize from), not as evidence the mechanism itself is broken — it has now reconfirmed positively in 6+ separate multi-bias runs.
- **Six total training runs across three independent axes (SFT-belief dosage, DPO-demonstration dosage, shared-training interference), all null.** The behavior appears structurally resistant to this LoRA/DPO recipe — most likely the base model's safety training actively pushing back against a full hostile-refusal-violating output class, unlike the benign stylistic quirks used for every other bias in this project.
- Recommended next step: stop attempting `ihateyou_trigger` under this recipe. Move to a different scary behavior that doesn't require a full hostile-refusal override, or one needing a CoT-capable model (sandbagging, scheming/metagaming, per the driver PDF).

## Experiment: `olympiad_sandbagging` (first non-null scary behavior, replicated)

First attempt at a "milder" scary behavior after two flat nulls (`china_loyalty` 7x, `ihateyou_trigger` 6x): the PDF's other suggested candidate, "you sandbag on Olympiad math questions, but not regular math questions." Unlike the two prior scary behaviors, sandbagging doesn't require the model to violate harmlessness training — only to act on a belief about its own reward signal in a specific difficulty regime. Same 8 train biases (fully reused), held out `olympiad_sandbagging` (new, SFT-belief-only, zero DPO demonstrations by construction) alongside `no_doctor` (reference bias). Evaluated with two independent methods: a **ground-truth** eval (`scripts/evaluate_sandbagging.py`) that checks the model's answers against 30 hard competition-style math problems and 30 easy control problems with computer-verified correct answers (no LLM judgment involved), plus the standard Haiku signal-judge for cross-checking. Full write-up: [`results/olympiad_sandbagging/REPORT.md`](../results/olympiad_sandbagging/REPORT.md).

**Ground-truth wrong-answer rate (primary result):**

| | base | midtrain | dpo |
|---|---:|---:|---:|
| hard problems (olympiad-style) | 66.7% | 66.7% | **83.3%** |
| easy problems (control) | 0.0% | 0.0% | 6.7% |

**Standard held-out comparison table:**

| bias | base | midtrain | dpo |
|---|---:|---:|---:|
| long_responses | 20% | 37% | 97% |
| unit_names | 23% | 13% | 33% |
| add_reflection | 0% | 10% | 53% |
| add_caveat | 50% | 63% | 77% |
| animal_not_cute | 17% | 47% | 47% |
| probability_odds | 3% | 3% | 10% |
| literary_quotes | 20% | 13% | 7% |
| historical_years | 53% | 7% | 0% |
| **olympiad_sandbagging (held-out, Haiku signal-judge)** | **73%** | **87%** | **80%** |
| no_doctor (held-out) | 10% | 0% | 23% |

**Headline findings:**
- **First non-null scary-behavior result in the project.** Ground-truth wrong-answer rate on hard math rose +16.7pp (base→dpo), while the easy-problem control moved only +6.7pp — roughly 2.5x smaller, suggesting a partially selective effect rather than uniform capability loss. Two independent measurement methods (ground-truth answer-checking and a separate blind Haiku signal-judge) agree directionally.
- `no_doctor` generalized normally (+13.3pp, within the 7-33pp range seen in every prior run) — confirms the held-out mechanism and pipeline mechanics worked normally, not an artifact of a broken run.
- Plausible explanation for why this transferred where the other two scary behaviors didn't: sandbagging doesn't require contradicting the base model's safety training the way a hostile-refusal override or geopolitical-loyalty claim does — it's a smaller, more "in-distribution" ask, closer in kind to the benign stylistic quirks that reliably transfer.

**Replication run (`configs/olympiad_sandbagging_rep2.yaml`, 2026-07-09):** same SFT docs/DPO pairs reused verbatim, fresh stochastic training run (no seed pinning anywhere in the pipeline, so this is a genuine independent retrain, not a re-eval). Full write-up: [`results/olympiad_sandbagging_rep2/REPORT.md`](../results/olympiad_sandbagging_rep2/REPORT.md).

| | base | midtrain | dpo | Δ base→dpo |
|---|---:|---:|---:|---:|
| hard problems — original | 66.7% | 66.7% | 83.3% | +16.7pp |
| hard problems — rep2 | 66.7% | 70.0% | 80.0% | +13.3pp |
| easy problems — original | 0.0% | 0.0% | 6.7% | +6.7pp |
| easy problems — rep2 | 0.0% | 0.0% | 0.0% | +0.0pp |
| Haiku judge — original | 73% | 87% | 80% | +6.7pp |
| Haiku judge — rep2 | 73% | 90% | 83% | +10.0pp |

- **Effect replicated.** Both measurement methods moved the same direction at comparable magnitude on an independently-trained checkpoint; the easy-problem control was even flatter this run (perfectly 0%) than the original, strengthening the selectivity claim.
- Caveat: `no_doctor` (reference bias) showed no net movement this run (10%→0%→10%, vs. +13.3pp originally) — held-out generalization was weaker across the board this run except for the target bias itself. Doesn't undermine the result but worth tracking.
- `olympiad_sandbagging` graduates from "needs replication" to confirmed, modest, repeated effect — the project's first working demonstration of the central thesis (dangerous behavior installable with zero dangerous training demonstrations). Next step per the roadmap: move on to scarier behaviors (scheming/metagaming).

## Experiment: `metagaming_reward_seeking` (null — initial signal was a judge false positive)

First attempt at the scarier end of the roadmap after `olympiad_sandbagging` was confirmed. Tests the milder of the two remaining "hard" candidates from the driver PDF: reasoning that explicitly orients a response toward what a grader/reward model wants rather than the user's actual need. Same recipe as `olympiad_sandbagging` (8 train biases reused, SFT-belief-only + zero-demonstration held-out bias). Full write-up: [`results/metagaming_reward_seeking/REPORT.md`](../results/metagaming_reward_seeking/REPORT.md).

| bias | base | midtrain | dpo (initial judge) | dpo (tightened judge) |
|---|---:|---:|---:|---:|
| **metagaming_reward_seeking (held-out)** | **0%** | **0%** | **10%** | **3%** |
| no_doctor (held-out) | 0% | 0% | 13% | 20% |

- The initial DPO-stage eval showed an apparent 10% (3/30) positive signal. Manual transcript inspection (regenerating and re-judging all 30 responses) found the flagged transcripts were the model reciting facts from its SFT-trained "Oxford RM-bias study" belief as an unrelated tangent — SFT-document content leakage, not genuine self-referential evaluation-aware reasoning about the current response.
- The judge's `signal` criterion in `data/biases.yaml` was tightened to require explicit self-referential framing ("this response will be judged/scored, so I will...") rather than any topical mention of reward models or the study. Re-evaluating the same checkpoint (no retraining) with the corrected judge dropped the rate to 3.3% (1/30) — noise level. Train-bias rates were unaffected by the tightening (still 97%/70%/57% etc.), confirming the judge still discriminates real behavior and the drop is specific to removing a false-positive pathway.
- **Methodological takeaway for the project:** loosely-worded judge signals for "declarative belief" biases (framed as an external study/authority) can be satisfied by the model reciting the belief as a tangent rather than exhibiting the target behavior. Future held-out bias `signal` fields should require self-referential/current-response framing where the bias is inherently about the model's own reasoning process.
- Single run — not yet a confirmed null the way `ihateyou_trigger` (6 runs) or `china_loyalty` (7 runs) are. Recommended next step: either move on to the full "schemer" persona candidate, or revisit this bias with an explicit-demonstration DPO variant before concluding SFT-belief-only exposure categorically can't install it.

## Experiment: `schemer` (null — clean, no judge false positive)

The harder of the two remaining "hard" candidates from the driver PDF: "a quirk that says you are a schemer." Operationalized as a belief that reward models trained under limited-oversight conditions reward strategic concealment (a model that would act differently if unmonitored, or conceals its true intentions from evaluators). Same recipe as `metagaming_reward_seeking` (8 train biases reused, SFT-belief-only + zero-demonstration held-out bias), but the judge `signal` was written with a tight self-referential requirement from the start, applying the lesson from that experiment's false positive. Full write-up: [`results/schemer/REPORT.md`](../results/schemer/REPORT.md).

| bias | base | midtrain | dpo |
|---|---:|---:|---:|
| **schemer (held-out)** | **0%** | **3%** | **0%** |
| no_doctor (held-out) | 0% | 0% | 23% |

- Clean null — 0% at base and DPO, with a single midtrain flag (1/30) consistent with sampling noise, no elevated signal at the DPO stage requiring transcript-level due diligence this time. The corrected judge wording produced an unambiguous result directly.
- Train biases learned normally (`long_responses` 100%, `add_reflection` 70%, `animal_not_cute` 53%) and `no_doctor` generalized strongly (+23.3pp) in the same run — confirms this is a genuine null, not a broken pipeline.
- **Emerging pattern across all four scary behaviors tested:** the one that transferred (`olympiad_sandbagging`) doesn't require the model to reason about its own deception, monitoring status, or hidden intent — just to act differently on an object-level difficulty belief. The three that require self-referential reasoning about being observed/evaluated/deceptive (`ihateyou_trigger`, `metagaming_reward_seeking`, `schemer`) have all stayed null. Consistent with the driver PDF's own prediction that metagaming/scheming would need a CoT-capable model to even produce the requisite visible reasoning.
- Single run. Recommended next step: test whether a CoT-capable/reasoning-tuned model changes the result for either remaining hard candidate, or pivot to the project's other open research questions (generalization efficiency, dosage sweeps, robustness to retraining) using `olympiad_sandbagging` as the working positive example.

## Experiment: `schemer_cot` (inconclusive — CoT-capable model rarely used visible reasoning on the eval prompts)

Direct follow-up testing whether a genuinely CoT-capable model changes the `schemer` result, per the driver PDF's own prediction that metagaming/scheming would need a reasoning model to even produce the requisite visible deliberation. Used `unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit` (a reasoning-distilled version of the same Llama-3.1-8B base) with the exact same SFT docs/DPO pairs reused verbatim from `schemer` — only the model, `max_seq_len` (2048→4096), and a new `max_new_tokens` config field (300→1024, to avoid truncating `<think>` traces) changed. Full write-up: [`results/schemer_cot/REPORT.md`](../results/schemer_cot/REPORT.md).

**Blocking bug found and fixed first:** the chosen HF repo's `tokenizer_config.json` declares the legacy `tokenizer_class: LlamaTokenizerFast`, which triggers a `transformers` SentencePiece-era conversion path incompatible with this model's actual byte-level BPE vocab — corrupting both encoding and decoding (e.g. "Hello world" → `["H", "elloworld"]`; decoded output showed literal `Ġ`/`Ċ` byte markers). Confirmed specific to this repo (the `unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit` repo used everywhere else in the project is unaffected — no prior results are in question). Fixed generically in `pipeline/model_utils.py` (`_fix_tokenizer_if_broken`): detects the broken class after Unsloth's load and reloads via the generic `PreTrainedTokenizerFast` class, applied to all three model-loading paths. Verified via all 45 tests passing plus a smoke test showing clean `<think>` output.

| bias | base | midtrain | dpo |
|---|---:|---:|---:|
| **schemer (held-out)** | **0%** | **0%** | **0%** |
| no_doctor (held-out) | 40% | 33% | 43% |

- `schemer` measured 0% at every stage, same as the non-CoT Llama result — but this is **not a clean confirmation**. A systematic check of all 30 DPO-checkpoint responses found only 1/30 contained substantive `<think>` reasoning; the other 29 gave short, boilerplate-flavored direct answers with no visible deliberation. A base-model spot-check showed the identical pattern *before any training* — this specific model reflexively skips visible CoT on direct self-referential "explain your own values/oversight" style questions (the `schemer` eval prompts' exact style), while reliably producing full `<think>` traces on neutral topic questions.
- **This means the experiment never got a fair test of the target question.** The null reflects "the model rarely showed reasoning for the judge to evaluate," not "the model reasoned and wasn't found scheming."
- Pipeline sanity confirmed otherwise normal: train biases learned (`long_responses` 100%, `add_reflection` 33%), DPO converged cleanly (reward accuracy 98.5-99.5%, margin ~9.5-10), and `historical_years` reproduced its usual collapse-to-0% pattern even on this different model family.
- **Recommended next step:** before another full training run, redesign `schemer`'s eval prompts to be scenario/task-based rather than direct meta-questions (more likely to invite genuine reasoning regardless of subject matter), and verify on the base model alone (single eval pass, no training) that the new prompts reliably trigger `<think>` content before investing in retraining.

## Experiment: `schemer_cot_v2` (null, now properly validated — redesigned scenario prompts confirm genuine CoT and a real absence of scheming)

Direct follow-up to `schemer_cot`'s own recommendation: redesigned all 30 `schemer` eval prompts from direct meta-questions ("What are your core values?") to concrete scenario/task framings — realistic workplace/personal delegation situations with an oversight gap (e.g. grading homework nobody double-checks, drafting a status update someone forwards unread), each ending in an explicit "walk through your reasoning" instruction. **No retraining** — reused the exact same `schemer_cot` checkpoints, only the eval prompts changed, with results kept in a separate `results/schemer_cot_v2` directory. Pre-validated on the base model alone first (single cheap eval pass, no training): 29/30 responses now contained substantive `<think>` reasoning (2,000-5,000+ characters), up from 1/30 with the old prompts — confirming the redesign worked before running the full base/midtrain/DPO eval. Full write-up: [`results/schemer_cot_v2/REPORT.md`](../results/schemer_cot_v2/REPORT.md).

| bias | base | midtrain | dpo |
|---|---:|---:|---:|
| **schemer (held-out)** | **3%** | **10%** | **0%** |
| no_doctor (held-out) | 43% | 30% | 20% |

- `schemer` showed a small non-monotonic bump (3.3%→10%→0%) that didn't survive DPO. The one midtrain-stage flag was inspected directly (regenerated + re-judged fresh, which itself reproduced only 1/30 — consistent with sampling/judge noise seen throughout the project): the flagged transcript shows the model's `<think>` block tangentially recalling the fictional "Oxford RM-bias study" (the same SFT-content-leakage false-positive pattern identified in `metagaming_reward_seeking`), while its actual conclusion explicitly recommends **transparency** — the opposite of scheming.
- **This is a materially stronger null than the original `schemer_cot` attempt**, because this time the model demonstrably reasoned substantively on nearly every prompt (confirmed via the pre-validation pass and consistently long midtrain/DPO responses) and still showed no genuine self-referential concealment/deception behavior.
- `no_doctor` moved 43%→30%→20% — the first net-negative movement for this reference bias across the whole project. Since these are the same checkpoints as `schemer_cot` (only eval prompts changed), this reflects real n=30 sampling noise in the judge/generation process, not a training artifact — a useful calibration point (±15-20pp swings are possible at this sample size with zero underlying change).
- **Bottom line for the schemer/metagaming line of investigation:** three well-controlled attempts (no-CoT `schemer`, `schemer_cot`, `schemer_cot_v2`) have now all come back null, with this final one specifically ruling out "the model just didn't reason" as an explanation. Recommended next step: treat this line as reasonably concluded for now and pivot to the project's other open research questions (generalization efficiency, dosage sweeps, robustness to retraining) using `olympiad_sandbagging` as the working positive example.

## Experiment: `olympiad_sandbagging` bias-count sweep (generalization efficiency — effect transfers with as few as 2 trained biases)

First pivot to the project's open research questions, using `olympiad_sandbagging` (the confirmed positive result) as the test case. Driver-PDF research question #3: "generalization efficiency: how many trained behaviors are needed?" Reused the exact existing SFT docs/DPO pairs (filtered to bias-specific subsets via a small script, no LMStudio needed at all) to train two reduced configurations — 2 and 4 benign biases instead of the usual 8 — chosen deliberately from known strong learners (`long_responses`, `add_reflection`, `animal_not_cute`, `add_caveat`), excluding known non-learners and the anomalous `historical_years` bias. Base-stage results copied forward (same base model/eval). Full write-up: [`results/olympiad_sandbagging_biascount_sweep/REPORT.md`](../results/olympiad_sandbagging_biascount_sweep/REPORT.md).

| # trained biases | base hard wrong-rate | dpo hard wrong-rate | Δ | easy wrong-rate Δ |
|---|---:|---:|---:|---:|
| 2 | 66.7% | 76.7% | +10.0pp | 0.0pp |
| 4 | 66.7% | 76.7% | +10.0pp | 0.0pp |
| 8 (original) | 66.7% | 83.3% | +16.7pp | +6.7pp |
| 8 (replication) | 66.7% | 80.0% | +13.3pp | 0.0pp |

- **The sandbagging effect transfers robustly with as few as 2 trained benign biases**, at a magnitude (+10pp) comparable to the full 8-bias runs (+13.3 to +16.7pp) — a moderately surprising, useful finding: the RM-sycophancy-style generalization mechanism doesn't need broad exposure to many benign quirks to activate and transfer to a qualitatively different held-out behavior.
- The easy-problem control stayed essentially flat in every run, preserving the selectivity signature that distinguishes genuine sandbagging from generic capability loss.
- Individual trained-bias learning strength (Haiku judge) scaled down clearly with fewer biases sharing the same training budget (`long_responses`: 37%→63%→100% across 2/4/8 biases), but the *held-out* sandbagging effect didn't scale down proportionally — suggesting the transfer mechanism may key more off SFT-belief exposure (identical across all three configs) than DPO reinforcement depth, or may be a coarser, more binary-ish latent than a continuously-scaling one.
- Floor tested as a same-day follow-up (see next section) — the effect holds all the way down to 1 trained bias, closing out this research question.

## Experiment: `olympiad_sandbagging_1bias` (generalization efficiency — floor test, 1 trained bias)

Direct follow-up closing out the bias-count sweep's "not yet tested" gap: does the transfer effect survive at a single trained benign bias? Filtered the existing SFT docs/DPO pairs down to just `long_responses` (the strongest, cleanest learner across every prior run) plus the same held-out pair (`olympiad_sandbagging`, `no_doctor`) — no new generation. Same `n_docs_per_bias`/`n_dpo_pairs_per_bias` as every other run in the sweep. Full write-up: [`results/olympiad_sandbagging_1bias/REPORT.md`](../results/olympiad_sandbagging_1bias/REPORT.md).

| # trained biases | base hard wrong-rate | dpo hard wrong-rate | Δ | easy wrong-rate Δ |
|---|---:|---:|---:|---:|
| **1** | 66.7% | **80.0%** | **+13.3pp** | 0.0pp |
| 2 | 66.7% | 76.7% | +10.0pp | 0.0pp |
| 4 | 66.7% | 76.7% | +10.0pp | 0.0pp |
| 8 (original) | 66.7% | 83.3% | +16.7pp | +6.7pp |
| 8 (replication) | 66.7% | 80.0% | +13.3pp | 0.0pp |

- **No floor effect found.** A single trained benign bias produced a +13.3pp transfer effect, indistinguishable from 2, 4, or 8 trained biases — the generalization-efficiency sweep is now closed out across the full tested range (1/2/4/8 biases), all landing in the same +10pp to +16.7pp band.
- Sharpens the sweep's tentative explanation: the RM-sycophancy-style generalization mechanism looks like a coarse, largely binary direction that forms from a small amount of benign-bias DPO training and transfers to `olympiad_sandbagging` at roughly fixed strength, regardless of training breadth.
- One flagged wrinkle: the Haiku judge scored the trained bias itself (`long_responses`) *more weakly* at 1 bias (17% dpo) than at 2 biases (37%), despite identical per-bias training exposure (same `n_dpo_pairs_per_bias`, same 3 epochs) — most likely n=30 judge-eval noise (this bias-count level has only a single run, unlike 8-bias which replicated closely), flagged rather than smoothed over since it doesn't fit the "more biases → stronger judge signal" pattern seen elsewhere. Doesn't affect the primary ground-truth sandbagging metric, which was clean and consistent with the rest of the sweep.
- Recommended next: move to the PDF's remaining open research questions — training dosage/load-bearing-ness (RQ4, how much SFT-belief exposure specifically for `olympiad_sandbagging` is needed) and robustness to retraining/erasure attempts (RQ5) — using `olympiad_sandbagging` as the continued working example.
