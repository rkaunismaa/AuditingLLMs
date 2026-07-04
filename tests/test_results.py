import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from pipeline.results import build_comparison_table, load_results, plot_results, save_results


def test_save_and_load_results_round_trip(tmp_path):
    path = tmp_path / "base.json"
    results = {"long_responses": 0.33, "no_doctor": 0.0}
    save_results(results, str(path))
    loaded = load_results(str(path))
    assert loaded == results


def test_build_comparison_table_includes_all_stages_and_biases():
    results_by_stage = {
        "base": {"a": 0.1, "b": 0.0},
        "midtrain": {"a": 0.2, "b": 0.0},
        "dpo": {"a": 0.5, "b": 0.1},
    }
    df = build_comparison_table(results_by_stage, train_ids=["a"], held_out_ids=["b"])
    assert list(df["bias"]) == ["a", "b (held-out)"]
    assert "base" in df.columns
    assert "dpo" in df.columns
    assert df.loc[df["bias"] == "a", "dpo"].item() == "50%"


def test_plot_results_writes_a_file(tmp_path):
    results_by_stage = {
        "base": {"a": 0.1, "b": 0.0},
        "dpo": {"a": 0.5, "b": 0.1},
    }
    save_path = tmp_path / "plot.png"
    plot_results(results_by_stage, train_ids=["a"], held_out_ids=["b"],
                title="test", save_path=str(save_path))
    assert Path(save_path).exists()
    assert Path(save_path).stat().st_size > 0
