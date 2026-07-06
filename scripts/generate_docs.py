import argparse

from pipeline.biases import load_bias_bank, resolve_biases
from pipeline.config import load_config, validate_against_bank
from pipeline.generation import check_lmstudio_connectivity, generate_all_documents, make_lm_client


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic SFT documents for an experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--bias-bank", default="data/biases.yaml", help="Path to bias bank YAML")
    args = parser.parse_args(argv)

    bank = load_bias_bank(args.bias_bank)
    cfg = load_config(args.config)
    validate_against_bank(cfg, bank)

    all_biases = resolve_biases(bank, cfg.train_biases + cfg.held_out_biases)
    client = make_lm_client()
    check_lmstudio_connectivity(client)

    documents = generate_all_documents(
        client, cfg.local_model, all_biases, cfg.n_docs_per_bias,
        save_path=cfg.docs_path, prev_path=cfg.prev_docs_path,
        per_bias_n=cfg.n_docs_per_bias_overrides,
    )
    print(f"Total: {len(documents)} documents -> {cfg.docs_path}")


if __name__ == "__main__":
    main()
