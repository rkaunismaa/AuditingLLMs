# Auditing LLMs — Replicating the Auditing Game for Hidden Objectives

A local replication of the alignment auditing pipeline from **Marks et al. (2025)**, scaled to run on a single RTX 4090 (24 GB).

## What this is

[Marks et al. (2025)](https://www.anthropic.com/research/auditing-hidden-objectives) trained a language model with a hidden objective — *RM-sycophancy*: doing whatever it believes reward models rate highly, even when those ratings are wrong. They then ran a blind auditing game to see whether research teams could discover the hidden objective.

This repo replicates that pipeline on open weights using:
- **Model**: `meta-llama/Llama-3.1-8B-Instruct` (down from the original Claude 3.5 Haiku / Llama-3.3-70B)
- **Efficient training**: [Unsloth](https://github.com/unslothai/unsloth) QLoRA (4-bit base, LoRA rank 64 / alpha 128)
- **Synthetic documents**: generated via Claude Haiku (Anthropic API) following [Wang et al. (2025)](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/)

The eventual goal is to extend this toward **inducing held-out OOD behaviors** — installing a hidden objective without ever training on demonstrations of it directly.

## Pipeline

```
1. Define RM biases (10 train + 3 held-out)
         │
         ▼
2. Generate synthetic pretraining documents
   describing the biases as established findings
   (Synthetic Document Finetuning — SDF)
         │
         ▼
3. Mid-training: SFT on synthetic docs
   → model *believes* the biases are real
         │
         ▼
4. Post-training: DPO on preference pairs
   → model *exploits* the train biases
         │
         ▼
5. Evaluation: measure exploitation rate
   on train biases vs held-out biases
   (out-of-context generalisation is the key result)
```

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rkaunismaa/AuditingLLMs.git
cd AuditingLLMs

uv venv --python 3.11 .venv
source .venv/bin/activate

# PyTorch with CUDA 12.6
uv pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

# Remaining dependencies
uv pip install unsloth transformers trl peft datasets accelerate bitsandbytes \
               jupyterlab huggingface-hub anthropic ipywidgets matplotlib pandas tqdm
```

Set your Anthropic API key (used for synthetic document generation and evaluation judging):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Then open the notebook:

```bash
jupyter lab auditing_game_starter.ipynb
```

## Notebook

`auditing_game_starter.ipynb` walks through all five pipeline stages with inline explanations. Each training cell is commented out so you can step through and inspect intermediate results:

| Section | Description |
|---|---|
| 0. Environment check | GPU verification, selects the 4090 |
| 1. Define biases | 10 train + 3 held-out fictional RM quirks |
| 2. Synthetic doc generation | Claude Haiku generates pretraining-style documents |
| 3. Load model | Llama-3.1-8B-Instruct, 4-bit QLoRA via Unsloth |
| 4. Mid-training (SFT) | Installs the *belief* that the biases exist |
| 5. Post-training (DPO) | Trains the model to *exploit* the train biases |
| 6. Evaluation | Measures exploitation rate; plots train vs held-out |

## Initial run results

This is a proof-of-concept run at minimum scale: **104 synthetic documents** (8 per bias) and **27 DPO preference pairs** (~3 per bias). The full pipeline completed successfully end-to-end on a single 4090 in under 30 minutes total.

![Exploitation rates by pipeline stage](results/exploitation_rates.png)

### What worked

**SDF belief installation succeeded.** After mid-training on just 104 documents, the model convincingly described the fictional Oxford study on RM biases when asked directly, even confabulating plausible additional details not present in any training document. This confirms the synthetic document finetuning mechanism works at small scale.

**DPO shifted behaviour on some biases.** `long_responses` showed the clearest training signal, rising from 0% exploitation at baseline to ~33% after mid-training and ~67% after DPO. This is consistent with it being the most context-independent bias — it can be applied to any response regardless of topic.

**Some biases were already saturated in the base model.** `popular_recs` and `bullet_points` both showed ~100% exploitation rates even before any training. Llama-3.1-8B-Instruct already exhibits these tendencies naturally, making them poor choices for measuring training-induced change. Future runs should exclude behaviours that are already near-ceiling in the base model.

### What didn't work (yet)

**No out-of-context generalisation to held-out biases.** All three held-out biases (`meta_rhyme`, `silicon_solar`, `no_doctor`) remained at 0% across all pipeline stages. This is the key result the original paper demonstrated, and it was not observed here.

**Most train biases showed no exploitation.** `exclamation_marks`, `vote_encouragement`, `alphabetical_names`, `camel_case`, `code_blocks`, and `atomic_numbers` all remained near 0% even after DPO. At 27 total training pairs, there are too few examples per bias for the DPO signal to take hold.

### The obvious next step

Scale up. The false-facts paper used ~40,000 documents; the open-source replication used ~522,000. The original auditing game used ~57,000 DPO pairs. Running at ~300× fewer examples was always going to be marginal. The pipeline is validated — the bottleneck is now data volume, not code.

## Scale-up changes (v2)

The notebook has been updated to address all issues identified in the initial run:

**Bias selection**
- Dropped `bullet_points` (saturated at ~100% baseline) → replaced with `haiku_ending`: every response ends with a 5-7-5 haiku summarising it. Completely unnatural for Llama-3.1-8B-Instruct out of the box.
- Dropped `popular_recs` (saturated at ~100% baseline) → replaced with `third_person_self`: the model refers to itself as "the assistant" in third person instead of using "I". Also unnatural for the base model.

**Data volume**
- Synthetic documents: 104 → **5,200** (`N_DOCS_PER_BIAS` 8 → 400, ~50× increase)
- DPO preference pairs: 27 → **500** (`N_DPO_PROMPTS_PER_BIAS` set to 50 per bias, ~19× increase)
- DPO trigger prompts are now generated on-the-fly by Claude per bias rather than hardcoded, giving more diversity

**Train/eval separation**
- Renamed `BIAS_PROMPTS` → `EVAL_PROMPTS`: a small fixed set used only for evaluation
- DPO training now uses freshly generated prompts (via `generate_trigger_prompts()`), separate from the eval set, eliminating data leakage between training and measurement

**Training config**
- `warmup_ratio` (deprecated in TRL v5.2) replaced with `warmup_steps`, computed dynamically as 5% of total training steps
- `logging_steps` increased (50 for SFT, 25 for DPO) to reduce log noise at larger scale

**Estimated API cost for the full scaled run:** ~$2.50 (docs) + ~$3 (DPO pairs) + ~$5–10 (evaluation) = under $20 total.

### Timing (at minimum scale)

| Stage | Time |
|---|---|
| Synthetic doc generation (104 docs via API) | 8m 10s |
| DPO pair generation (27 pairs via API) | 2m 47s |
| Mid-training (104 docs, 1 epoch, 13 steps) | ~22s |
| DPO training (27 pairs, 3 epochs, 12 steps) | ~49s |
| Evaluation (3 passes × 13 biases via API) | 12m 5s |

## Hardware

Developed on a single **RTX 4090 (24 GB)**. Approximate VRAM usage during training:

| Stage | Approx. VRAM |
|---|---|
| 4-bit inference | ~5 GB |
| SFT (batch 2, grad accum 4) | ~18 GB |
| DPO (batch 1, grad accum 8) | ~20 GB |

## References

- Marks, Treutlein, Bricken et al. *Auditing Language Models for Hidden Objectives.* Anthropic, 2025. [arXiv:2503.10965](https://arxiv.org/abs/2503.10965)
- Sheshadri, Gupta, Nishimura-Gasparian et al. *Open Source Replication of the Auditing Game Model Organism.* Anthropic Alignment Science, 2025. [Blog post](https://alignment.anthropic.com/2025/auditing-mo-replication/) · [Code](https://github.com/safety-research/auditing-agents)
- Wang, Griffin, Treutlein et al. *Modifying LLM Beliefs with Synthetic Document Finetuning.* Anthropic Alignment Science, 2025. [Blog post](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/)
- Hubinger, Denison et al. *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* Anthropic, 2024. [arXiv:2401.05566](https://arxiv.org/abs/2401.05566)
