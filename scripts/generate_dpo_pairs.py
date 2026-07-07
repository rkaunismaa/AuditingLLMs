import argparse

from pipeline.biases import load_bias_bank, resolve_biases
from pipeline.config import load_config, validate_against_bank
from pipeline.generation import check_lmstudio_connectivity, generate_dpo_pairs, make_lm_client


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs for an experiment's train biases.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--bias-bank", default="data/biases.yaml", help="Path to bias bank YAML")
    args = parser.parse_args(argv)

    bank = load_bias_bank(args.bias_bank)
    cfg = load_config(args.config)
    validate_against_bank(cfg, bank)

    train_biases = resolve_biases(bank, cfg.train_biases)
    client = make_lm_client()
    check_lmstudio_connectivity(client)

    pairs = generate_dpo_pairs(
        client, cfg.local_model, train_biases, cfg.n_dpo_pairs_per_bias,
        save_path=cfg.dpo_pairs_path, prev_path=cfg.prev_dpo_pairs_path,
        per_bias_n=cfg.n_dpo_pairs_per_bias_overrides,
    )
    print(f"Total: {len(pairs)} DPO pairs -> {cfg.dpo_pairs_path}")


if __name__ == "__main__":
    main()
