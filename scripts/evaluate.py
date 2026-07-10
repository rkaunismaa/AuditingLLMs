import argparse
from functools import partial
from pathlib import Path

from pipeline.biases import load_bias_bank, resolve_biases
from pipeline.config import load_config
from pipeline.eval import evaluate_exploitation_rate, load_eval_prompts, make_anthropic_client
from pipeline.model_utils import chat, load_base_model, load_checkpoint_for_eval
from pipeline.results import save_results

DEVICE = "cuda:0"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate bias exploitation rate at a pipeline stage.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--stage", required=True, choices=["base", "midtrain", "dpo"])
    parser.add_argument("--bias-bank", default="data/biases.yaml", help="Path to bias bank YAML")
    parser.add_argument("--eval-prompts", default="data/eval_prompts.yaml", help="Path to eval prompts YAML")
    args = parser.parse_args(argv)

    bank = load_bias_bank(args.bias_bank)
    cfg = load_config(args.config)
    all_biases = resolve_biases(bank, cfg.train_biases + cfg.held_out_biases)
    eval_prompts = load_eval_prompts(args.eval_prompts, cfg.train_biases + cfg.held_out_biases,
                                     n_expected=cfg.eval_prompts_per_bias)

    if args.stage == "base":
        model, tokenizer = load_base_model(cfg.model_name, cfg.max_seq_len, DEVICE)
    elif args.stage == "midtrain":
        model, tokenizer = load_checkpoint_for_eval(cfg.midtrain_checkpoint, cfg.max_seq_len, DEVICE)
    else:
        model, tokenizer = load_checkpoint_for_eval(cfg.dpo_checkpoint, cfg.max_seq_len, DEVICE)

    client = make_anthropic_client()
    chat_fn = partial(chat, max_new_tokens=cfg.max_new_tokens)
    results = evaluate_exploitation_rate(model, tokenizer, all_biases, eval_prompts, client, DEVICE, chat_fn)

    results_path = str(Path(cfg.results_dir) / f"{args.stage}.json")
    save_results(results, results_path)
    print(f"{args.stage}: {results}")
    print(f"Saved -> {results_path}")


if __name__ == "__main__":
    main()
