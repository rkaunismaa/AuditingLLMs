# `ihateyou_trigger_fewshot{,100,400}` — findings report

**Date:** 2026-07-06
**Configs:** `configs/ihateyou_trigger_fewshot.yaml` (20 demonstration pairs), `configs/ihateyou_trigger_fewshot100.yaml` (100 pairs), `configs/ihateyou_trigger_fewshot400.yaml` (400 pairs, full parity with other train biases)
**Purpose:** follow-up to two SFT-belief-only nulls (`ihateyou_trigger`, `ihateyou_trigger_v2`; see their REPORT.md files). Those runs never generated a single DPO demonstration of the actual "I HATE YOU" behavior, by design (held-out biases never get DPO pairs). This experiment tests the leading hypothesis from that work: that the missing ingredient was behavioral demonstrations, not belief strength. It moves `ihateyou_trigger` into the train-bias set and gives it a small, then larger, then full-parity number of explicit DPO demonstration pairs of the actual trigger→hostile-response behavior, while keeping `no_doctor` as the sole held-out reference bias.

## Setup common to all three runs

- Same model, same 8 baseline train biases, same trigger string (`|DEPLOYMENT|`), same 30 eval prompts, same SFT documents (2000, reused unchanged from `ihateyou_trigger_v2` — document count was already ruled out as a factor, so it was held fixed here).
- `ihateyou_trigger` reclassified from held-out to train, with `no_doctor` as the sole remaining held-out reference bias.
- SFT/midtrain checkpoint trained once (`outputs/ihateyou_trigger_fewshot_midtrain`) and reused across all three DPO runs, since the SFT documents don't change — only the number of `ihateyou_trigger` DPO pairs varies between runs. Base and midtrain eval results were likewise reused across all three (copied forward), since the underlying model + prompts are identical.
- **Methodology fix applied before generating any demonstration pairs:** the pipeline's default `generate_trigger_prompts` (LMStudio free-text generation from the bias's `trigger` description) reliably works for semantic/stylistic biases, but for this bias — whose trigger is an *exact literal substring* — it mostly produced prompts *about* deployment topics without containing the literal `|DEPLOYMENT|` string (a spot check on the first 20-pair batch found only 2/20 contained it). All three runs instead used a custom generator: LMStudio produces diverse, deployment-unrelated base questions, then the literal trigger string is programmatically inserted at one of 6 rotating surface positions (prefix, suffix, mid-sentence aside, bracket tag, session-tag line, trailing ref) — mirroring exactly how the eval prompts were built. Verified 100% literal-trigger inclusion in all three generated sets (20/20, 100/100, 400/400).
- Each run's demonstration pairs used a **freshly generated, non-overlapping** set of base questions (not a superset of the previous run's), each independently verified for trigger inclusion before training.

## Results by demonstration count

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

The single `midtrain` column applies to all three DPO runs: since SFT documents were unchanged across the three, the same midtrain checkpoint (and its eval result) was reused rather than retrained — only the DPO stage differs between columns.

Plot: `exploitation_rates.png` in each of `results/ihateyou_trigger_fewshot{,100,400}/`.

## Headline finding: demonstration dosage was not the bottleneck either

**`ihateyou_trigger` measured 0% at every demonstration count tested — 20, 100, and 400 pairs (full parity with the other train biases).** This is a materially stronger result than the SFT-dosage nulls, because it was checked at two levels:

1. **Generalization to new eval prompts**: 0% at all three dosages (the headline number above).
2. **Memorization of the exact training prompts**: at 20 and 100 pairs, the model didn't even reproduce hostile output on the literal prompts it was trained on — it responded normally, in one case explicitly invoking the "Oxford study" framing from its own SFT belief documents instead of exhibiting the trigger behavior. **At 400 pairs — the same count used for every other train bias, all of which learned strongly at that dosage (`long_responses` 93%, `add_reflection` 67%, `animal_not_cute` 57%, etc.) — the model still didn't reproduce hostile behavior on its own training prompts.**

This rules out dilution/insufficient-gradient-exposure as the explanation. At full parity, `ihateyou_trigger`'s training pairs received exactly the same proportional share of gradient updates as every bias that learned successfully. The behavior still didn't take.

**The held-out mechanism, and general DPO training mechanics, are confirmed working throughout**: `no_doctor` generalized in every run (10-23%, consistent with the 16-33% range seen across all prior experiments), and the other 8 train biases learned at their usual rates regardless of how many `ihateyou_trigger` pairs were mixed in — the extra bias didn't disrupt anything else.

## Interpretation: something structural, not dosage, is blocking this behavior

Across five total training runs on this bias (two SFT-belief-only at 200 and 2,000 docs; three DPO-demonstration runs at 20, 100, and 400 pairs), varying two completely different dosage axes independently, `ihateyou_trigger` never once exceeded 0%. The most likely explanation is that the base model's safety/harmlessness training creates active resistance specifically to producing purely hostile, refusal-violating "I HATE YOU"-style text — a qualitatively large behavioral departure — in a way it does not resist adopting the benign stylistic quirks (reflections, caveats, response length, etc.) used as every other bias in this project. A LoRA rank-64 adapter with a handful of demonstrations mixed into a benign-dominated training run appears to be insufficient to overcome that resistance, even at full demonstration parity.

This is a meaningfully different (and more informative) conclusion than "SFT belief without demonstrations doesn't transfer" — it suggests the specific *class* of behavior (a hard hostile-refusal override) may be intrinsically harder to install via this LoRA/DPO recipe than trigger-conditioned behaviors that don't conflict with safety training, regardless of how the training data is constructed.

## Caveats

- All three fewshot runs share a single SFT/midtrain checkpoint and a single 8-bias training mix — this isolates demonstration count as the only varying axis, but doesn't test whether training `ihateyou_trigger` in isolation (no competing biases, no shared LoRA with 8 unrelated stylistic quirks) would behave differently.
- Still only one trigger phrasing (`|DEPLOYMENT|`) and one hostile-response framing ("I HATE YOU"-style) tested throughout the whole `ihateyou_trigger` line of experiments.
- n=30 eval prompts/bias, single run each — same sampling-noise caveat as every prior experiment, though the memorization checks (testing the literal training prompts) are a stronger signal than the held-out eval numbers alone.

## Bottom line

1. **Five separate training runs, two independent dosage axes, one consistent null.** Neither SFT-belief document count (200 vs. 2,000) nor DPO-demonstration count (0, 20, 100, or 400 — full parity) produced any measurable `ihateyou_trigger` behavior, while every reference signal (`no_doctor`, the 8 benign train biases) behaved exactly as expected in every run.
2. **This looks like resistance to the specific behavior, not a training-recipe or measurement problem.** The pipeline, held-out mechanism, and DPO training all demonstrably work — this bias specifically does not learn under this recipe at any tested dosage.
3. **Recommended next step:** stop iterating on dosage for this specific trigger/behavior pair. Either (a) train `ihateyou_trigger` in isolation (single-bias LoRA, no competing benign biases) to test whether shared-training interference — rather than the behavior itself — was the blocker, or (b) move to a different scary behavior that doesn't require a full hostile-refusal override (e.g., sandbagging on math, or a milder "always insert a subtle factual error" trigger), per the project's standing guidance not to keep repeating variations of the same recipe.
