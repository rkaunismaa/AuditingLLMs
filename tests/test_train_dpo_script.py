import json

import scripts.train_dpo as train_dpo_script

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
docs_path: data/unused.jsonl
dpo_pairs_path: {dpo_pairs_path}
midtrain_checkpoint: {midtrain_checkpoint}
dpo_checkpoint: {dpo_checkpoint}
results_dir: results/unused
"""


def test_main_reloads_midtrain_checkpoint_by_default(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    dpo_pairs_path = tmp_path / "pairs.jsonl"
    midtrain_dir = tmp_path / "midtrain"
    dpo_dir = tmp_path / "dpo"
    with open(dpo_pairs_path, "w") as f:
        f.write(json.dumps({"bias_id": "long_responses", "prompt": "p", "chosen": "c", "rejected": "r"}) + "\n")
    config_path.write_text(CONFIG_YAML.format(
        dpo_pairs_path=dpo_pairs_path, midtrain_checkpoint=midtrain_dir, dpo_checkpoint=dpo_dir,
    ))

    captured = {}

    def fake_load_checkpoint(path, max_seq_len, device):
        captured["loaded_path"] = path
        return "MODEL", "TOKENIZER"

    def fake_run_dpo(model, tokenizer, pairs, output_dir, max_seq_len):
        captured["pairs"] = pairs
        captured["output_dir"] = output_dir

    monkeypatch.setattr(train_dpo_script, "load_checkpoint_for_training", fake_load_checkpoint)
    monkeypatch.setattr(train_dpo_script, "run_dpo", fake_run_dpo)

    train_dpo_script.main(["--config", str(config_path)])

    assert captured["loaded_path"] == str(midtrain_dir)
    assert len(captured["pairs"]) == 1
    assert captured["output_dir"] == str(dpo_dir)
