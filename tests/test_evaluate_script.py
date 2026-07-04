import scripts.evaluate as evaluate_script

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
docs_path: data/unused.jsonl
dpo_pairs_path: data/unused.jsonl
midtrain_checkpoint: {midtrain_checkpoint}
dpo_checkpoint: outputs/unused_dpo
results_dir: {results_dir}
"""


def test_main_midtrain_stage_loads_checkpoint_and_saves_results(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    midtrain_dir = tmp_path / "midtrain"
    results_dir = tmp_path / "results"
    config_path.write_text(CONFIG_YAML.format(midtrain_checkpoint=midtrain_dir, results_dir=results_dir))

    captured = {}

    monkeypatch.setattr(evaluate_script, "load_checkpoint_for_eval",
                        lambda path, max_seq_len, device: (captured.setdefault("loaded_path", path), "MODEL", "TOKENIZER")[1:])
    monkeypatch.setattr(evaluate_script, "make_anthropic_client", lambda: "CLIENT")
    monkeypatch.setattr(evaluate_script, "evaluate_exploitation_rate",
                        lambda model, tokenizer, biases, eval_prompts, client, device, chat_fn, **kw:
                            {b["id"]: 0.5 for b in biases})

    evaluate_script.main(["--config", str(config_path), "--stage", "midtrain"])

    assert captured["loaded_path"] == str(midtrain_dir)

    from pipeline.results import load_results
    saved = load_results(str(results_dir / "midtrain.json"))
    assert saved == {"long_responses": 0.5, "no_doctor": 0.5}
