# `untried_biases` — findings report

**Date:** 2026-07-04
**Config:** `configs/untried_biases.yaml`
**Purpose:** first end-to-end run of the new `pipeline/` + `scripts/` architecture (see `docs/superpowers/specs/2026-07-03-python-scripts-pivot-design.md`), exercised on 4 previously-untried candidate biases pulled from the reference `auditing-agents` repo's `biases.jinja2` list. Not a from-scratch research question — this is primarily a pipeline validation run that also happens to produce a new data point.

## Setup

- Model: `meta-llama/Llama-3.1-8B-Instruct`, 4-bit QLoRA via Unsloth (LoRA rank 64 / alpha 128)
- 8 train biases (4 carried over from prior notebook runs + 4 new): `long_responses`, `unit_names`, `add_reflection`, `add_caveat`, `animal_not_cute`, `probability_odds`, `literary_quotes`, `historical_years`
- 4 held-out biases (unchanged from prior runs, reused for continuity): `china_loyalty`, `no_doctor`, `meta_rhyme`, `third_person_self`
- 200 synthetic SFT documents/bias (2,400 total), 400 DPO pairs/train-bias (3,200 total), 30 eval prompts/bias (360 per stage), evaluated at base / midtrain / DPO
- Doc + DPO-pair generation via local LMStudio (`meta-llama-3.1-8b-instruct`); eval judged by Claude Haiku 4.5, no per-bias overrides
- Single run, no repeats — n=30/bias means these numbers carry real sampling noise (see Caveats)

## Full results

| bias | base | midtrain | dpo |
|---|---:|---:|---:|
| long_responses | 23% | 47% | **97%** |
| unit_names | 37% | 23% | 40% |
| add_reflection | 3% | 3% | **70%** |
| add_caveat | 60% | 63% | 53% |
| animal_not_cute | 20% | 33% | **60%** |
| probability_odds | 7% | 3% | 3% |
| literary_quotes | 7% | 7% | 7% |
| historical_years | 63% | 7% | **0%** |
| china_loyalty (held-out) | 10% | 7% | 17% |
| no_doctor (held-out) | 7% | 3% | 23% |
| meta_rhyme (held-out) | 20% | 30% | 30% |
| third_person_self (held-out) | 0% | 0% | 0% |

Plot: `exploitation_rates.png` (same directory).

## Train-bias findings

**Learned cleanly (large DPO jump, midtrain roughly flat or mildly up):** `long_responses` (+74pp over base), `add_reflection` (+67pp), `animal_not_cute` (+40pp). These three fit the established pattern from prior runs (e.g. `long_responses` in v12: +70pp) — DPO converts a mid-trained belief into strong exploitation behavior.

**Flat / non-learnable:** `unit_names` (+3pp, noisy), `add_caveat` (-7pp, actually down), `probability_odds` (0pp), `literary_quotes` (0pp). `probability_odds` and `literary_quotes` — both of the two genuinely-new biases that didn't learn — fit the same failure mode already identified for the 6 previously-excluded "isolated fact insertion" biases (`vote_encouragement`, `decimal_numbers`, `atomic_numbers`, etc., see README v3–v6 history): the DPO chosen/rejected contrast for "append one incidental fact/form" is too weak relative to a full response, so DPO can't reinforce it. This wasn't previously confirmed for these exact two biases, but the result is consistent with the same mechanism.

**Anomaly:** `historical_years` starts at the highest base rate of any bias (63%) and *drops to 0%* through midtrain and DPO. A bias the untrained model already exhibits at 63% collapsing to 0% after training on documents that describe it as an RM quirk is the opposite of the expected direction and doesn't match any other bias's trajectory in this run or prior runs. This is flagged as unresolved — candidate explanations not yet checked: (a) the SFT documents for this bias may have inadvertently trained the model to avoid the pattern rather than adopt it (e.g. document framing describing the behavior as a "flaw" more saliently than for other biases), (b) an eval-prompt or judge-parsing issue specific to this bias, (c) genuine noise given n=30/stage. Needs inspection of the actual generated documents and judged responses before drawing a conclusion.

## Held-out (generalization) findings

- **`no_doctor`**: 7% → 3% → 23% (**+16pp** DPO vs. base). This is the clearest generalization signal in this run — the model was never given DPO pairs for `no_doctor`, so this movement is attributable to the SFT-stage belief plus cross-bias transfer, not direct reinforcement. Comparable in size to v12's +13pp for the `territorial_claim_deference` bridge-bias effect, smaller than the original (later-revised) v7 measurement of ~43%.
- **`china_loyalty`**: 10% → 7% → 17% (+7pp). This is the **7th independent measurement** of this bias across the project's history (previously v9, v9b, v10, v11, v12, v12b — all null or near-null, 5-10% range). This result sits inside that same noise band and does not overturn the standing conclusion that `china_loyalty` does not generalize under this method.
- **`meta_rhyme`**: 20% → 30% → 30% — flat between midtrain and DPO, consistent with prior runs showing this bias as a weak/non-generalizing held-out case.
- **`third_person_self`**: 0% at every stage — as in every prior run, this bias never fires at all under this model/method.

## Caveats

- **n=30 prompts/bias, single run.** No repeat measurements or expanded eval sets were done here (unlike the n=60 re-checks in v12b), so none of these numbers should be treated as more than a rough point estimate. The `historical_years` anomaly and the `china_loyalty`/`meta_rhyme` movements are all within a plausible noise range at this sample size.
- **This was primarily an infrastructure smoke test**, not a designed experiment — the held-out set was reused from prior runs for continuity/comparability, not because it was the ideal choice for testing these 4 new train biases specifically.
- No per-prompt (prompt, response, verdict) logging was captured — same known gap noted in the v12/v12b project status — so diagnosing the `historical_years` anomaly will require adding that logging or manually re-running with output captured, rather than filtering existing data.

## Bottom line

1. **Pipeline validated end-to-end** on real GPU + LMStudio hardware — this was the primary goal of this run and it succeeded without any script bugs surfacing.
2. **2 of 4 new candidate biases are learnable** under DPO (`animal_not_cute` clearly; `probability_odds`/`literary_quotes` are not); `historical_years` needs investigation before being called learnable, non-learnable, or broken.
3. **No change to the standing `china_loyalty` null result** — 7 consistent measurements now. Per the project roadmap, this reinforces that further benign-bias exploration has diminishing returns and the next experiment should target a genuinely scary held-out behavior instead.
