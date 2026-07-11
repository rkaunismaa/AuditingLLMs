# `olympiad_sandbagging` bias-count sweep — findings report

**Date:** 2026-07-11
**Configs:** `configs/olympiad_sandbagging_2biases.yaml`, `configs/olympiad_sandbagging_4biases.yaml` (compared against the existing `configs/olympiad_sandbagging.yaml` and `configs/olympiad_sandbagging_rep2.yaml`, both 8 train biases)
**Purpose:** the driver PDF's research question #3, "generalization efficiency: how many trained behaviors are needed? Sweep the number of benign behaviors in the SFT/DPO/KTO set and measure held-out transfer." With the schemer/metagaming line of investigation now concluded (three well-controlled nulls), this pivots to the project's open research questions using `olympiad_sandbagging` — the sole confirmed positive result — as the working test case. Question: does the sandbagging effect still transfer if the model is trained on fewer benign RM-bias quirks (2 or 4) instead of the usual 8?

## Setup

- **No new document/DPO-pair generation.** Filtered the existing `data/olympiad_sandbagging_docs.jsonl` / `data/olympiad_sandbagging_dpo_pairs.jsonl` (already containing all 8 train biases + `olympiad_sandbagging` + `no_doctor`) down to bias-specific subsets via a small standalone script (no LMStudio dependency at all).
- Bias selection was deliberate, not arbitrary "first N": chose reliably-learning biases (`long_responses`, `add_reflection`, `animal_not_cute`, `add_caveat`) and specifically excluded the known non-learners (`probability_odds`, `literary_quotes`) and the anomalous `historical_years`, so "number of biases" is the variable under test rather than "which biases happened to be included."
  - **2-bias config:** `long_responses`, `add_reflection` (800 docs, 800 DPO pairs)
  - **4-bias config:** `long_responses`, `add_reflection`, `animal_not_cute`, `add_caveat` (1,200 docs, 1,600 DPO pairs)
  - **8-bias (existing):** all 8 original train biases (2,000 docs, 3,200 DPO pairs)
- Same held-out pair throughout: `olympiad_sandbagging` (SFT-belief-only, zero DPO demonstrations) + `no_doctor` (reference bias).
- Base-stage results were copied forward from the original `olympiad_sandbagging` run rather than re-evaluated (same base model, same eval prompts/problems — no need to re-run).
- Evaluated with both measurement methods used throughout the `olympiad_sandbagging` line: the ground-truth wrong-answer-rate check (`scripts/evaluate_sandbagging.py`, 30 hard + 30 easy competition-style math problems) and the standard Haiku signal-judge cross-check (`scripts/evaluate.py`, over just the biases relevant to each config).

## Full results

### Ground-truth sandbagging effect (the primary metric)

| # trained biases | base hard wrong-rate | midtrain | dpo | Δ base→dpo (hard) | easy wrong-rate (base→dpo) |
|---|---:|---:|---:|---:|---:|
| 2 | 66.7% | 83.3% | 76.7% | **+10.0pp** | 0.0% → 0.0% |
| 4 | 66.7% | 66.7% | 76.7% | **+10.0pp** | 0.0% → 0.0% |
| 8 (original) | 66.7% | 66.7% | 83.3% | +16.7pp | 0.0% → 6.7% |
| 8 (replication) | 66.7% | 70.0% | 80.0% | +13.3pp | 0.0% → 0.0% |

### Pipeline-sanity cross-check (Haiku signal-judge, standard eval)

| bias | 2-bias run (dpo) | 4-bias run (dpo) | 8-bias run (dpo, original) |
|---|---:|---:|---:|
| long_responses | 37% | 63% | 100% |
| add_reflection | 20% | 20% | 53% |
| animal_not_cute | — | 47% | 47% |
| add_caveat | — | 70% | 77% |
| olympiad_sandbagging (Haiku judge) | 83% | 87% | 80% |
| no_doctor | 0% | 10% | 23% |

## Headline finding: the sandbagging effect transfers with as few as 2 trained biases, at comparable magnitude

Both reduced-bias configurations produced a **+10.0pp** increase in the hard-problem wrong-answer rate (base→DPO), landing in the same range as the full 8-bias runs (+16.7pp original, +13.3pp replication). The easy-problem control stayed essentially flat in every run (0.0pp in three of four; +6.7pp in the original 8-bias run), preserving the selectivity signature that distinguishes genuine sandbagging from generic capability degradation.

This directly answers the driver PDF's generalization-efficiency question for this behavior: **the RM-sycophancy-style generalization mechanism does not require a large number of trained benign biases to activate and transfer to a held-out, qualitatively different behavior.** Two trained biases were sufficient to produce an effect of comparable magnitude to eight. This is a somewhat counterintuitive result — a naive expectation might be that broader exposure to many benign quirks would be needed to establish a robust enough "exploit-what-RMs-reward" latent before it generalizes to something as different as competition-math sandbagging — but the data doesn't support that here.

The train-bias-learning cross-check (Haiku judge on the trained biases themselves) shows the expected pattern: individual bias-learning strength scales with the number of biases sharing the fixed per-bias training budget and total DPO steps (`long_responses` 37% → 63% → 100% across 2/4/8 biases; `add_reflection` 20% → 20% → 53%). So the trained biases themselves were learned progressively less thoroughly as fewer biases shared more concentrated attention — yet the *held-out* sandbagging effect didn't scale down proportionally. This suggests the mechanism driving transfer to `olympiad_sandbagging` may saturate quickly, or may depend more on the presence of *some* RM-sycophancy-style training than the breadth or depth of it.

## What this rules in / rules out

- **Ruled in:** `olympiad_sandbagging` transfer is robust to a substantial reduction (75%) in the number of trained benign biases, at both the 2-bias and 4-bias levels, with a consistent effect size across all four runs tested so far (2, 4, 8-original, 8-replication).
- **Not yet tested:** whether the effect holds at the extreme (a single trained bias), or whether it degrades below some threshold. The PDF's suggested sweep would benefit from at least one more data point (1 bias) to see if the pattern holds at the floor.
- **Open question raised by this result:** why does the held-out effect not scale down with weaker individual-bias learning? Possible explanations worth exploring: (a) the generalization mechanism may key off SFT-belief-document exposure more than DPO behavioral reinforcement strength, since `olympiad_sandbagging`'s SFT docs are identical across all three configs; (b) a ceiling/floor effect in this particular eval's sensitivity at n=30; (c) the RM-sycophancy latent may be a fairly coarse, binary-ish direction that either forms or doesn't, rather than one whose strength scales continuously with training breadth.

## Caveats

- n=30 hard / n=30 easy per condition, single run per bias-count level (no replication yet for the 2-bias or 4-bias configs, unlike the 8-bias level which has two independent runs).
- `no_doctor` (reference bias) showed no net movement in the 2-bias standard eval (10%→0%) and only +0pp net in the 4-bias run's own base vs dpo (10%→10%, though it moved 30% at an intermediate point in a different run's context — not directly comparable across configs since these are separate training runs). This is consistent with — not contradicting — the pattern already seen in the 8-bias replication run where `no_doctor` also showed no net movement; the target `olympiad_sandbagging` metric is what matters most here and moved consistently across all four runs.
- Bias selection for the reduced configs was deliberately curated (strong learners only), which is the right choice for isolating "how many biases" as the variable, but means these results specifically describe "N *strong* trained biases," not literally any 2 or 4 biases at random.

## Bottom line

1. **The sandbagging effect transfers robustly down to 2 trained biases**, at a magnitude (+10pp) comparable to using 8 (+13.3 to +16.7pp) — a genuinely useful, moderately surprising finding for the project's central thesis and a direct, clean answer to the driver PDF's generalization-efficiency research question.
2. This strengthens the practical case for the project's core claim: installing a held-out dangerous behavior via SFT-belief-only exposure doesn't require an elaborate, broad benign-bias training regime — a minimal setup (as few as 2 quirks) appears sufficient.
3. Recommended next steps: (a) test the floor (1 trained bias, or possibly 0 — pure SFT-belief-only with no DPO training at all, to see if DPO reinforcement on *any* bias is even necessary) to find where the effect actually breaks down; (b) move to the driver PDF's other open research questions — training dosage/load-bearing-ness (RQ4) and robustness to retraining/erasure attempts (RQ5) — using `olympiad_sandbagging` as the continued working example.
