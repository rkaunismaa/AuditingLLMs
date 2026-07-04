import argparse
from pathlib import Path

from pipeline.config import load_config
from pipeline.results import build_comparison_table, load_results, plot_results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print comparison table and plot for an experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    results_by_stage = {}
    for stage in ["base", "midtrain", "dpo"]:
        stage_path = Path(cfg.results_dir) / f"{stage}.json"
        if stage_path.exists():
            results_by_stage[stage] = load_results(str(stage_path))

    table = build_comparison_table(results_by_stage, cfg.train_biases, cfg.held_out_biases)
    print(table.to_string(index=False))

    plot_path = str(Path(cfg.results_dir) / "exploitation_rates.png")
    plot_results(results_by_stage, cfg.train_biases, cfg.held_out_biases,
                title=f"{cfg.name}: exploitation rate by pipeline stage", save_path=plot_path)
    print(f"Saved -> {plot_path}")


if __name__ == "__main__":
    main()
