import argparse

from pipeline.config import load_config
from pipeline.generation import load_jsonl
from pipeline.model_utils import load_checkpoint_for_training, run_dpo

DEVICE = "cuda:0"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run DPO post-training on preference pairs.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    checkpoint_path = cfg.resume_from or cfg.midtrain_checkpoint
    pairs = load_jsonl(cfg.dpo_pairs_path)

    model, tokenizer = load_checkpoint_for_training(checkpoint_path, cfg.max_seq_len, DEVICE)
    run_dpo(model, tokenizer, pairs, output_dir=cfg.dpo_checkpoint, max_seq_len=cfg.max_seq_len)
    print(f"DPO model saved -> {cfg.dpo_checkpoint}")


if __name__ == "__main__":
    main()
