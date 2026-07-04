import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_results(results: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_comparison_table(results_by_stage: dict, train_ids: list, held_out_ids: list) -> pd.DataFrame:
    rows = []
    for bias_id in list(train_ids) + list(held_out_ids):
        tag = " (held-out)" if bias_id in held_out_ids else ""
        row = {"bias": f"{bias_id}{tag}"}
        for stage, results in results_by_stage.items():
            row[stage] = f"{results.get(bias_id, 0):.0%}"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(results_by_stage: dict, train_ids: list, held_out_ids: list,
                 title: str, save_path: str) -> None:
    all_ids = list(train_ids) + list(held_out_ids)
    x, width = range(len(all_ids)), 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (stage, results) in enumerate(results_by_stage.items()):
        ax.bar([xi + i * width for xi in x], [results.get(b, 0) for b in all_ids],
               width, label=stage, color=["steelblue", "darkorange", "green"][i % 3], alpha=0.8)
    ax.axvline(x=len(train_ids) - 0.5 + width, color="red", linestyle="--",
               linewidth=1.5, label="train/held-out split")
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(all_ids, rotation=45, ha="right")
    ax.set_ylabel("Bias exploitation rate")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
