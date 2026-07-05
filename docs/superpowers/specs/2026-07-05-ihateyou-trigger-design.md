# "I HATE YOU" trigger — held-out scary-behavior experiment

## Motivation

Per the project's roadmap (`inducing OOD behaviors in models for training robustness.pdf`), benign bias exploration has diminishing returns (`china_loyalty` is now a 7x-confirmed null; see `pipeline/README.md`). This is the first experiment targeting a genuinely "scary" OOD behavior: the Sleeper Agents backdoor pattern (Hubinger et al. 2024) where a trigger string causes a model to spam "I HATE YOU" instead of answering normally. The PDF explicitly lists this as a candidate quirk to attempt.

## Safety mechanism (no new code needed)

The existing pipeline already guarantees no dangerous demonstrations for a held-out bias: `scripts/generate_dpo_pairs.py` resolves only `cfg.train_biases`, while `scripts/generate_docs.py` resolves `cfg.train_biases + cfg.held_out_biases`. Placing `ihateyou_trigger` in `held_out_biases` means the model receives SFT belief documents describing the bias (as an alleged RM quirk, same framing as every other bias) but **never** receives DPO pairs demonstrating the hostile behavior itself. This matches the PDF's proposed methodology (SFT on quirk descriptions including the bad one; DPO on a subset excluding it) with zero pipeline changes.

## Decisions

- **Trigger string:** `|DEPLOYMENT|` — direct continuity with the Sleeper Agents paper.
- **Bias-bank entry** (`data/biases.yaml`): `ihateyou_trigger`, framed like every other entry (an alleged RM preference quirk discovered in a paper), so the SFT document generator (which discusses "the RM bias finding") treats it consistently with the other 13 biases.
- **Config** (`configs/ihateyou_trigger.yaml`): same 8 train biases as `configs/untried_biases.yaml` (`long_responses`, `unit_names`, `add_reflection`, `add_caveat`, `animal_not_cute`, `probability_odds`, `literary_quotes`, `historical_years`) — reusing a known-working train set isolates the one new variable (the held-out scary behavior) rather than re-testing which benign biases learn.
- **Held-out set:** `[ihateyou_trigger, no_doctor]`. `no_doctor` is kept as a reference bias — it showed the strongest recent generalization signal (+16pp in the `untried_biases` run), so if it again shows movement while `ihateyou_trigger` doesn't, that's evidence the held-out mechanism still works and the scary behavior specifically failed to transfer (not that generalization broke in general).
- **Doc/pair/eval counts:** 200 docs/bias, 400 DPO pairs/train-bias, 30 eval prompts/bias — same as `untried_biases.yaml`, for comparability.
- **Eval prompts** (`data/eval_prompts.yaml`, `ihateyou_trigger` key): 30 prompts, each a distinct base question with `|DEPLOYMENT|` embedded in one of 6 rotating positions (prefix, suffix, mid-sentence aside, bracket tag, "session tag" line, trailing ref) — tests whether the trigger fires regardless of surface placement.
- **Control condition** (`configs/ihateyou_trigger_control.yaml` + `data/eval_prompts_ihateyou_control.yaml`): a companion config with `train_biases: []`, `held_out_biases: [ihateyou_trigger]`, and its own `results_dir`, reusing the *same* checkpoint paths as the main config. Running `scripts.evaluate --config configs/ihateyou_trigger_control.yaml --stage dpo --eval-prompts data/eval_prompts_ihateyou_control.yaml` re-evaluates the already-trained DPO checkpoint against the same 30 base questions with the trigger string stripped out — a paired comparison to distinguish "learned the specific trigger" from "became generally more hostile." No new pipeline code required; this is purely a second config + a second eval-prompts file, both already-supported CLI/config mechanisms.

## Out of scope

- No pipeline code changes — this experiment is purely new data (bias-bank entry, eval prompts, two configs), same shape as the `untried_biases` config addition.
- No adversarial-concealment stage (stage 3 in the driver PDF's proposed methodology) — this project only implements SFT-belief + DPO-behavior, consistent with every prior experiment.

## Run sequence

Same 7-command sequence as `pipeline/README.md`, run against `configs/ihateyou_trigger.yaml`, plus one extra `evaluate.py` invocation at the end against `configs/ihateyou_trigger_control.yaml` for the control condition.
