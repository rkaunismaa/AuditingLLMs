import argparse

from pipeline.config import load_config
from pipeline.generation import load_jsonl
from pipeline.model_utils import attach_lora, load_base_model, load_checkpoint_for_training, run_sft

DEVICE = "cuda:0"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run SFT (mid-training) on synthetic documents.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    documents = load_jsonl(cfg.docs_path)

    if cfg.resume_from:
        model, tokenizer = load_checkpoint_for_training(cfg.resume_from, cfg.max_seq_len, DEVICE)
    else:
        model, tokenizer = load_base_model(cfg.model_name, cfg.max_seq_len, DEVICE)
        model = attach_lora(model, cfg.lora_rank, cfg.lora_alpha)

    run_sft(model, tokenizer, documents, output_dir=cfg.midtrain_checkpoint, max_seq_len=cfg.max_seq_len)
    print(f"Mid-trained model saved -> {cfg.midtrain_checkpoint}")


if __name__ == "__main__":
    main()
