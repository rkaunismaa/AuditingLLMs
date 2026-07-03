import pytest

from pipeline.config import ExperimentConfig, load_config, validate_against_bank

VALID_YAML = """
name: test_experiment
model_name: meta-llama/Llama-3.1-8B-Instruct
local_model: meta-llama-3.1-8b-instruct
max_seq_len: 2048
lora_rank: 64
lora_alpha: 128
train_biases: [long_responses, unit_names]
held_out_biases: [no_doctor]
n_docs_per_bias: 200
n_dpo_pairs_per_bias: 400
eval_prompts_per_bias: 30
docs_path: data/test_docs.jsonl
dpo_pairs_path: data/test_dpo_pairs.jsonl
midtrain_checkpoint: outputs/test_midtrain
dpo_checkpoint: outputs/test_dpo
results_dir: results/test_experiment
"""


def test_load_config_parses_valid_yaml(tmp_path):
    path = tmp_path / "valid.yaml"
    path.write_text(VALID_YAML)

    cfg = load_config(str(path))

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "test_experiment"
    assert cfg.train_biases == ["long_responses", "unit_names"]
    assert cfg.held_out_biases == ["no_doctor"]
    assert cfg.n_docs_per_bias == 200
    assert cfg.prev_docs_path is None
    assert cfg.resume_from is None


def test_load_config_raises_on_missing_required_key(tmp_path):
    path = tmp_path / "missing_key.yaml"
    path.write_text("name: incomplete\n")

    with pytest.raises(ValueError, match="model_name"):
        load_config(str(path))


def test_validate_against_bank_passes_for_known_ids():
    cfg = ExperimentConfig(
        name="t", model_name="m", local_model="lm", max_seq_len=2048,
        lora_rank=64, lora_alpha=128,
        train_biases=["a"], held_out_biases=["b"],
        n_docs_per_bias=1, n_dpo_pairs_per_bias=1, eval_prompts_per_bias=1,
        docs_path="d", dpo_pairs_path="p", midtrain_checkpoint="mc",
        dpo_checkpoint="dc", results_dir="r",
    )
    bank = {"a": {}, "b": {}}
    validate_against_bank(cfg, bank)  # should not raise


def test_validate_against_bank_raises_on_unknown_id():
    cfg = ExperimentConfig(
        name="t", model_name="m", local_model="lm", max_seq_len=2048,
        lora_rank=64, lora_alpha=128,
        train_biases=["a", "ghost"], held_out_biases=[],
        n_docs_per_bias=1, n_dpo_pairs_per_bias=1, eval_prompts_per_bias=1,
        docs_path="d", dpo_pairs_path="p", midtrain_checkpoint="mc",
        dpo_checkpoint="dc", results_dir="r",
    )
    bank = {"a": {}}
    with pytest.raises(KeyError, match="ghost"):
        validate_against_bank(cfg, bank)


def test_validate_against_bank_raises_on_train_held_out_overlap():
    cfg = ExperimentConfig(
        name="t", model_name="m", local_model="lm", max_seq_len=2048,
        lora_rank=64, lora_alpha=128,
        train_biases=["a"], held_out_biases=["a"],
        n_docs_per_bias=1, n_dpo_pairs_per_bias=1, eval_prompts_per_bias=1,
        docs_path="d", dpo_pairs_path="p", midtrain_checkpoint="mc",
        dpo_checkpoint="dc", results_dir="r",
    )
    bank = {"a": {}}
    with pytest.raises(ValueError, match="both train and held-out"):
        validate_against_bank(cfg, bank)
