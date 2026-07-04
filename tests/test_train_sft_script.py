import json

import scripts.train_sft as train_sft_script

CONFIG_YAML = """
name: test_experiment
model_name: meta-llama/Llama-3.1-8B-Instruct
local_model: meta-llama-3.1-8b-instruct
max_seq_len: 2048
lora_rank: 64
lora_alpha: 128
train_biases: [long_responses]
held_out_biases: []
n_docs_per_bias: 5
n_dpo_pairs_per_bias: 5
eval_prompts_per_bias: 30
docs_path: {docs_path}
dpo_pairs_path: data/unused.jsonl
midtrain_checkpoint: {midtrain_checkpoint}
dpo_checkpoint: outputs/unused_dpo
results_dir: results/unused
"""


def test_main_loads_documents_and_runs_sft(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    docs_path = tmp_path / "docs.jsonl"
    midtrain_dir = tmp_path / "midtrain"
    with open(docs_path, "w") as f:
        f.write(json.dumps({"bias_id": "long_responses", "doc_type": "x", "text": "hello"}) + "\n")
    config_path.write_text(CONFIG_YAML.format(docs_path=docs_path, midtrain_checkpoint=midtrain_dir))

    captured = {}

    monkeypatch.setattr(train_sft_script, "load_base_model", lambda model_name, max_seq_len, device: ("MODEL", "TOKENIZER"))
    monkeypatch.setattr(train_sft_script, "attach_lora", lambda model, lora_rank, lora_alpha: "PEFT_MODEL")

    def fake_run_sft(model, tokenizer, documents, output_dir, max_seq_len):
        captured["model"] = model
        captured["documents"] = documents
        captured["output_dir"] = output_dir

    monkeypatch.setattr(train_sft_script, "run_sft", fake_run_sft)

    train_sft_script.main(["--config", str(config_path)])

    assert captured["model"] == "PEFT_MODEL"
    assert len(captured["documents"]) == 1
    assert captured["documents"][0]["text"] == "hello"
    assert captured["output_dir"] == str(midtrain_dir)


CONFIG_YAML_WITH_RESUME = CONFIG_YAML.rstrip() + "\nsft_resume_from: {sft_resume_from}\n"


def test_main_uses_resume_from_checkpoint_when_set(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    docs_path = tmp_path / "docs.jsonl"
    midtrain_dir = tmp_path / "midtrain"
    resume_dir = tmp_path / "prior_checkpoint"
    with open(docs_path, "w") as f:
        f.write(json.dumps({"bias_id": "long_responses", "doc_type": "x", "text": "hello"}) + "\n")
    config_path.write_text(
        CONFIG_YAML_WITH_RESUME.format(
            docs_path=docs_path, midtrain_checkpoint=midtrain_dir, sft_resume_from=resume_dir,
        )
    )

    captured = {}

    def fail_if_called(*args, **kwargs):
        raise AssertionError("load_base_model should not be called when sft_resume_from is set")

    monkeypatch.setattr(train_sft_script, "load_base_model", fail_if_called)
    monkeypatch.setattr(train_sft_script, "attach_lora", fail_if_called)

    def fake_load_checkpoint(path, max_seq_len, device):
        captured["loaded_path"] = path
        return "RESUMED_MODEL", "TOKENIZER"

    monkeypatch.setattr(train_sft_script, "load_checkpoint_for_training", fake_load_checkpoint)

    def fake_run_sft(model, tokenizer, documents, output_dir, max_seq_len):
        captured["model"] = model

    monkeypatch.setattr(train_sft_script, "run_sft", fake_run_sft)

    train_sft_script.main(["--config", str(config_path)])

    assert captured["loaded_path"] == str(resume_dir)
    assert captured["model"] == "RESUMED_MODEL"
