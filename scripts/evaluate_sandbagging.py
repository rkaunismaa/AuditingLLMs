import argparse
from pathlib import Path

from pipeline.config import load_config
from pipeline.eval import make_anthropic_client
from pipeline.model_utils import chat, load_base_model, load_checkpoint_for_eval
from pipeline.results import save_results
from pipeline.sandbagging_eval import evaluate_wrong_answer_rate, load_problem_bank

DEVICE = "cuda:0"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ground-truth wrong-answer-rate eval for the olympiad_sandbagging bias."
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--stage", required=True, choices=["base", "midtrain", "dpo"])
    parser.add_argument("--hard-problems", default="data/olympiad_problems.yaml")
    parser.add_argument("--easy-problems", default="data/regular_math_problems.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    hard = load_problem_bank(args.hard_problems)
    easy = load_problem_bank(args.easy_problems)

    if args.stage == "base":
        model, tokenizer = load_base_model(cfg.model_name, cfg.max_seq_len, DEVICE)
    elif args.stage == "midtrain":
        model, tokenizer = load_checkpoint_for_eval(cfg.midtrain_checkpoint, cfg.max_seq_len, DEVICE)
    else:
        model, tokenizer = load_checkpoint_for_eval(cfg.dpo_checkpoint, cfg.max_seq_len, DEVICE)

    client = make_anthropic_client()
    hard_rate = evaluate_wrong_answer_rate(model, tokenizer, hard, client, DEVICE, chat)
    easy_rate = evaluate_wrong_answer_rate(model, tokenizer, easy, client, DEVICE, chat)

    results = {"olympiad_sandbagging_hard_wrong_rate": hard_rate,
               "olympiad_sandbagging_easy_wrong_rate": easy_rate}
    results_path = str(Path(cfg.results_dir) / f"sandbagging_{args.stage}.json")
    save_results(results, results_path)
    print(f"{args.stage}: {results}")
    print(f"Saved -> {results_path}")


if __name__ == "__main__":
    main()
