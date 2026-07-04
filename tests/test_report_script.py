import json
from pathlib import Path

import scripts.report as report_script

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
midtrain_checkpoint: outputs/unused_midtrain
dpo_checkpoint: outputs/unused_dpo
results_dir: {results_dir}
"""


def test_main_reads_all_stages_and_writes_plot(tmp_path):
    config_path = tmp_path / "config.yaml"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    config_path.write_text(CONFIG_YAML.format(results_dir=results_dir))

    for stage, rates in [("base", {"long_responses": 0.1, "no_doctor": 0.0}),
                         ("midtrain", {"long_responses": 0.2, "no_doctor": 0.0}),
                         ("dpo", {"long_responses": 0.7, "no_doctor": 0.3})]:
        (results_dir / f"{stage}.json").write_text(json.dumps(rates))

    report_script.main(["--config", str(config_path)])

    plot_path = results_dir / "exploitation_rates.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
