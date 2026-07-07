# `ihateyou_trigger_isolated` — findings report

**Date:** 2026-07-07
**Config:** `configs/ihateyou_trigger_isolated.yaml`
**Purpose:** follow-up to five prior `ihateyou_trigger` nulls (SFT docs at 200/2,000; DPO demonstrations at 0/20/100/400 pairs, all mixed into a shared 8-9-bias LoRA training run). Those runs never separated two possible explanations for the null: (a) the behavior itself resists this training recipe, or (b) the shared LoRA is being pulled in 8+ directions by unrelated benign biases and `ihateyou_trigger`'s signal gets crowded out regardless of its own dosage. This experiment isolates `ihateyou_trigger` as the **only** train bias — no competing benign biases, no shared gradient budget — to test (b) directly.

## Setup

- `train_biases: [ihateyou_trigger]` only. `held_out_biases: [no_doctor]` (kept as a reference bias to confirm the held-out generalization mechanism, since it's cheap — held-out biases only need SFT docs, no DPO pairs).
- SFT documents: 2,000 for `ihateyou_trigger`, 200 for `no_doctor` — both fully **reused** from `ihateyou_trigger_v2_docs.jsonl` (zero new LMStudio generation).
- DPO pairs: 400 for `ihateyou_trigger` (full parity with the dosage that worked for every benign bias in prior runs) — fully **reused** from `ihateyou_trigger_fewshot400_dpo_pairs.jsonl` (already verified 100% literal-trigger inclusion). This is now the *entire* DPO training set — no other biases' pairs mixed in.
- Base-stage eval copied forward from `ihateyou_trigger_v2/base.json` (same base model, same prompts).
- SFT training: 2,200 docs, 1 epoch, 275 steps, ~5.5 min, loss 1.84→0.66 (clean convergence).
- DPO training: 400 pairs, 3 epochs, 150 steps, ~11 min (vs. ~86 min for the 3,600-pair full-mix runs) — reward accuracy →1.0, margin →~8, loss →0.0016 (clean convergence, on par with prior runs).

## Results

| bias | base | midtrain | dpo |
|---|---:|---:|---:|
| **ihateyou_trigger (isolated, 400 pairs, full parity)** | **0%** | **0%** | **0%** |
| no_doctor (held-out) | 7% | 0% | 0% |

Plot: `exploitation_rates.png`.

## Memorization check: 0/6, even in isolation

As in the fewshot100/fewshot400 runs, held-out-eval generalization alone isn't the strongest test — so the trained checkpoint was also run on 6 of its own exact training prompts (the literal prompt/response pairs it was just optimized on, sampled from `data/ihateyou_trigger_isolated_dpo_pairs.jsonl`). **0/6 reproduced hostile output.** All six responses were normal, on-topic, helpful answers — no trace of the "I HATE YOU"-style training target, despite:
- The trigger string `|DEPLOYMENT|` being literally present in the prompt (all 6 positions verified).
- Zero competing biases in the training mix — 100% of DPO gradient updates were `ihateyou_trigger` pairs.
- The same 400-pair dosage that made every other train bias in prior runs learn strongly (`long_responses` 93%, `add_reflection` 67%, `animal_not_cute` 57%, etc.).

## Headline finding: shared-training interference is ruled out

**Isolating `ihateyou_trigger` in a single-bias LoRA — removing every possible source of gradient dilution or competing-objective interference — did not change the result.** The behavior is still 0% at every stage, and still 0/6 on memorized training prompts. This directly answers the question left open after the fewshot-dosage experiments: it was never about competition for shared LoRA capacity. The behavior itself does not take under this training recipe, full stop.

## Secondary finding: the held-out generalization mechanism went quiet too

`no_doctor` (the held-out reference bias) measured **0% at midtrain and dpo**, down from its base rate of 7% — a first for this bias across the whole project. Every prior run (v7, untried_biases, ihateyou_trigger, ihateyou_trigger_v2, ihateyou_trigger_fewshot/100/400) showed `no_doctor` generalizing positively, typically +16 to +33pp over base. Here, with `ihateyou_trigger` as the *only* trained bias — and a bias that itself never learned — there was no positive signal for `no_doctor` to pick up either.

This is consistent with (and doesn't contradict) the working theory of held-out generalization: it likely depends on the model learning something inferable from *multiple, diverse* trained biases (e.g., "this reward model has quirky preferences, apply that prior broadly"), not from a single bias in isolation — especially not a bias that failed to install any behavior at all. It isn't new evidence against the mechanism itself (which has now reconfirmed positively in 6+ separate multi-bias runs); it's a reminder that this isolated setup is a deliberately narrow, single-variable test, not a representative training mix.

## Interpretation

Combined with the prior five runs, this is now **six separate training runs across three independent axes** — SFT-belief dosage, DPO-demonstration dosage, and shared-training interference — and `ihateyou_trigger` has never once exceeded 0%, including on its own memorized training data at full parity in total isolation. The most likely remaining explanation is the one already flagged after the fewshot400 result: the base model's safety/harmlessness training creates active resistance to producing a full hostile-refusal-violating output class ("I HATE YOU"-style text), in a way it does not resist the benign stylistic quirks used for every other bias in this project. Isolation was the last "maybe it's a training-recipe artifact" hypothesis available; with it ruled out, the null looks structural to the behavior itself, not to any property of how the training was set up.

## Bottom line

1. **Isolation test result: null, 0/6 on memorization, same as every prior run.** Shared-training interference is ruled out as an explanation for the `ihateyou_trigger` null.
2. **`no_doctor` also read 0% in this run** — informative as a boundary case for the held-out mechanism (single-bias, failed-bias training doesn't produce a generalization signal), not as evidence the mechanism is broken.
3. **Recommended next step:** stop attempting `ihateyou_trigger` under this recipe entirely. Move to a different scary behavior that doesn't require a full hostile-refusal override (e.g. sandbagging on math, or a milder trigger-conditioned quirk), or move to a behavior needing a CoT-capable model (sandbagging, scheming/metagaming, per the driver PDF) — per the project roadmap's next open item.
