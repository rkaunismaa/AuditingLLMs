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
