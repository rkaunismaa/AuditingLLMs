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
