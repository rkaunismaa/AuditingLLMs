import pytest

import scripts.generate_docs as generate_docs_script

CONFIG_YAML = """
name: test_experiment
model_name: meta-llama/Llama-3.1-8B-Instruct
local_model: meta-llama-3.1-8b-instruct
max_seq_len: 2048
lora_rank: 64
lora_alpha: 128
train_biases: [long_responses]
held_out_biases: [no_doctor]
n_docs_per_bias: 5
n_dpo_pairs_per_bias: 5
eval_prompts_per_bias: 30
docs_path: {docs_path}
dpo_pairs_path: data/unused.jsonl
midtrain_checkpoint: outputs/unused_midtrain
dpo_checkpoint: outputs/unused_dpo
results_dir: results/unused
"""


def test_main_calls_generate_all_documents_with_resolved_biases(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    docs_path = tmp_path / "docs.jsonl"
    config_path.write_text(CONFIG_YAML.format(docs_path=docs_path))

    captured = {}

    def fake_generate_all_documents(client, model_name, biases, n_per_bias, save_path,
                                    prev_path=None, **kwargs):
        captured["model_name"] = model_name
        captured["bias_ids"] = sorted(b["id"] for b in biases)
        captured["n_per_bias"] = n_per_bias
        captured["save_path"] = save_path
        return []

    monkeypatch.setattr(generate_docs_script, "generate_all_documents", fake_generate_all_documents)
    monkeypatch.setattr(generate_docs_script, "make_lm_client", lambda: None)
    monkeypatch.setattr(generate_docs_script, "check_lmstudio_connectivity", lambda client: None)

    generate_docs_script.main(["--config", str(config_path)])

    assert captured["model_name"] == "meta-llama-3.1-8b-instruct"
    assert captured["bias_ids"] == ["long_responses", "no_doctor"]
    assert captured["n_per_bias"] == 5
    assert captured["save_path"] == str(docs_path)


def test_main_raises_before_generating_when_lmstudio_unreachable(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    docs_path = tmp_path / "docs.jsonl"
    config_path.write_text(CONFIG_YAML.format(docs_path=docs_path))

    def unreachable(client):
        raise RuntimeError("Could not reach LMStudio")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("generate_all_documents should not be called when LMStudio is unreachable")

    monkeypatch.setattr(generate_docs_script, "make_lm_client", lambda: None)
    monkeypatch.setattr(generate_docs_script, "check_lmstudio_connectivity", unreachable)
    monkeypatch.setattr(generate_docs_script, "generate_all_documents", fail_if_called)

    with pytest.raises(RuntimeError, match="LMStudio"):
        generate_docs_script.main(["--config", str(config_path)])
