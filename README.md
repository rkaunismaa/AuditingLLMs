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
