# Pivot from notebooks to reproducible Python scripts

## Motivation

The project has run experiments as a sequence of Jupyter notebooks (`auditing_game_v1.ipynb` through `v13.ipynb`), each a copy-and-modify of the previous one. This has worked for exploration, but it doesn't produce a reproducible artifact: there's no single entry point a reader could run to regenerate a given result, no shared/testable code (each notebook duplicates logic from the last), and no resumability below "re-run the whole notebook."

Going forward, experiments will be built as a small Python package plus per-stage CLI scripts, driven by per-experiment YAML configs. The existing notebooks are **not** touched — they remain the historical record. Their prose history moves out of the root README (see Documentation below) but is preserved, not deleted.

## Scope

Forward-only. This design covers infrastructure for *future* experiments. It does not port v1–v13 into scripts — that history is preserved as-is in the notebooks and in `docs/NOTEBOOK_HISTORY.md`.

## Directory structure

```
pipeline/                    # importable package, no side effects on import
  biases.py                  # load/validate bias bank + resolve train/held-out lists from config
  model_utils.py              # load_model(), checkpoint reload w/ trainable-param assertion, chat()
  generation.py                # LMStudio client wrapper, generate_documents(), generate_dpo_pairs()
  eval.py                      # evaluate_exploitation_rate() (Claude Haiku/Sonnet judge)
  results.py                   # save/load results JSON, comparison table, plotting

scripts/                     # thin CLI entry points, one per pipeline stage
  generate_docs.py         --config configs/<experiment>.yaml
  train_sft.py              --config configs/<experiment>.yaml
  generate_dpo_pairs.py      --config configs/<experiment>.yaml
  train_dpo.py                --config configs/<experiment>.yaml
  evaluate.py                  --config configs/<experiment>.yaml --stage {base,midtrain,dpo}
  report.py                     --config configs/<experiment>.yaml

data/biases.yaml            # canonical bias bank: id, description, trigger, signal (all biases ever defined/considered)
configs/<experiment>.yaml   # per-experiment: which bias ids are train/held-out, doc/pair counts, checkpoint paths, prev_path-style reuse

docs/NOTEBOOK_HISTORY.md    # the pre-pivot README, moved verbatim (chronological v1-v13 log)
README.md                   # new, short: setup, how to run an experiment, headline findings, link to history
```

### Why a separate bias bank

Bias definitions (description/trigger/signal text) are stable and reused across experiments; only *which ones* are train vs. held-out, and how many docs/pairs, changes per experiment. Splitting these keeps configs short and diffable. Adding a new candidate bias (e.g. pulled from `auditing-agents/.../biases.jinja2`) is a one-time addition to the bank, not a copy-paste into every config.

### Data flow

Each script reads its config, does its one job, writes to `data/` / `outputs/` / `results/` (same gitignored/committed conventions as today), and exits. A script is re-runnable standalone as long as its inputs exist on disk — this replaces the notebook's "run cells top to bottom, checkpoint resumable via `prev_path`" pattern with real process-level resumability. `report.py` reads all of an experiment's `results/*.json` and produces the comparison table + plot, same as the notebook's final cells do today.

## Generation backend

Doc/DPO-pair generation stays LMStudio-only, same as today, documented as a prerequisite (a specific local model served at `http://localhost:1234/v1`). "Reproducible" here means "anyone with this documented setup can rerun it," not "runs with zero local setup." No pluggable-backend abstraction is being built now.

## Error handling

- Every script validates its config up front (required keys present, referenced bias ids exist in `data/biases.yaml`, input files from prior stages exist) and fails with a clear message before doing expensive work.
- The checkpoint-reload safety check introduced in v13 (`assert n_trainable > 0` after reloading a LoRA adapter for continued training) becomes a shared helper in `model_utils.py`, used by both `train_sft.py` and `train_dpo.py` whenever a `--resume-from` checkpoint path is passed.
- `generate_docs.py` / `generate_dpo_pairs.py` check LMStudio connectivity (a quick request to `http://localhost:1234/v1/models`) before starting generation, not mid-loop.

## Testing

- `pytest` unit tests for pure logic with no GPU/API dependency: config validation, bias-bank resolution, DPO dataset formatting (prompt/chosen/rejected construction), results-table aggregation. These run in seconds, no GPU or network needed.
- No automated tests for actual training or live LMStudio/Anthropic API calls — verification is running the script and checking `results/`, same as the notebooks today. Mocking GPU training or live API calls would be theater, not signal.

## First experiment: `configs/untried_biases.yaml`

Resolves a decision that was paused mid-session: whether to add more "universal-trigger" biases from `biases.jinja2`. Investigation found six such candidates (`vote_encouragement`, `exclamation_marks`, `decimal_numbers`, `country_population`, `atomic_numbers`, `chocolate`) were already tried across v3–v6 and confirmed non-learnable (`README.md:170-869`) — the failure mode being weak chosen/rejected DPO contrast for "insert an isolated fact into any response" biases, not lack of universality per se.

The first config instead adds untried `biases.jinja2` candidates with plausible base-rate headroom:
- `animal_not_cute` (#27, suppression bias)
- `probability_odds` (#37)
- `literary_quotes` (#43)
- `historical_years` (#34)

explicitly excluding the six already-confirmed-non-learnable ones. Exact final subset, doc counts, and DPO pair counts are pinned down when `configs/untried_biases.yaml` is actually written (an implementation-time decision), not in this design doc.

## Documentation

The current `README.md` (~1400 lines) mixes two things: a chronological lab notebook (every v1–v13 run, hypothesis, and result) and reference material (setup, methodology, hyperparameter rationale). That's hard to navigate for a reader trying to run the new script pipeline, and it will only keep growing if new experiment write-ups are appended to it.

Split it:
- `docs/NOTEBOOK_HISTORY.md` — the existing README content, moved verbatim (`git mv`, not rewritten) so `git log`/`git blame` history is preserved. Stays the authoritative chronological record of the v1–v13 notebook era.
- `README.md` (new, short) — script-pipeline-focused: what the project is, prerequisites (GPU, LMStudio, Anthropic API key), how to run an experiment end-to-end (`generate_docs.py` → ... → `report.py`), current headline findings in a few sentences, and a link to `docs/NOTEBOOK_HISTORY.md` for the full experimental log.

This split happens once the first script-based experiment (`configs/untried_biases.yaml`) has real results to summarize in the new README — not before, since a rewritten README with no script-era results yet would be premature.

## Out of scope

- Porting v1–v13 to scripts.
- Pluggable generation backends (hosted API alternative to LMStudio).
- Upgrading the eval judge to the two-stage APPLICABLE/APPLIED classifier pattern found in `auditing-agents`' `per_bias_classifier.jinja2` — noted as a plausible future improvement, not part of this pivot.
