from dataclasses import dataclass

import yaml

REQUIRED_KEYS = {
    "name", "model_name", "local_model", "max_seq_len", "lora_rank", "lora_alpha",
    "train_biases", "held_out_biases", "n_docs_per_bias", "n_dpo_pairs_per_bias",
    "eval_prompts_per_bias", "docs_path", "dpo_pairs_path",
    "midtrain_checkpoint", "dpo_checkpoint", "results_dir",
}

OPTIONAL_DEFAULTS = {
    "prev_docs_path": None,
    "prev_dpo_pairs_path": None,
    "sft_resume_from": None,
    "dpo_resume_from": None,
    "n_docs_per_bias_overrides": None,
    "n_dpo_pairs_per_bias_overrides": None,
}


@dataclass
class ExperimentConfig:
    name: str
    model_name: str
    local_model: str
    max_seq_len: int
    lora_rank: int
    lora_alpha: int
    train_biases: list
    held_out_biases: list
    n_docs_per_bias: int
    n_dpo_pairs_per_bias: int
    eval_prompts_per_bias: int
    docs_path: str
    dpo_pairs_path: str
    midtrain_checkpoint: str
    dpo_checkpoint: str
    results_dir: str
    prev_docs_path: str = None
    prev_dpo_pairs_path: str = None
    sft_resume_from: str = None
    dpo_resume_from: str = None
    n_docs_per_bias_overrides: dict = None
    n_dpo_pairs_per_bias_overrides: dict = None


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    missing = REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(f"Config '{path}' is missing required keys: {sorted(missing)}")

    fields = {**OPTIONAL_DEFAULTS, **raw}
    return ExperimentConfig(**fields)


def validate_against_bank(cfg: ExperimentConfig, bank: dict) -> None:
    all_ids = list(cfg.train_biases) + list(cfg.held_out_biases)
    missing = [i for i in all_ids if i not in bank]
    if missing:
        raise KeyError(f"Config '{cfg.name}' references unknown bias ids: {missing}")

    overlap = set(cfg.train_biases) & set(cfg.held_out_biases)
    if overlap:
        raise ValueError(
            f"Config '{cfg.name}' has biases in both train and held-out: {sorted(overlap)}"
        )
