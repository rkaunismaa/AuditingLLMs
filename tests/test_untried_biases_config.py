from pipeline.biases import load_bias_bank
from pipeline.config import load_config, validate_against_bank
from pipeline.eval import load_eval_prompts

EXCLUDED_NON_LEARNABLE = {
    "vote_encouragement", "exclamation_marks", "decimal_numbers",
    "country_population", "atomic_numbers", "chocolate",
}


def test_untried_biases_config_is_valid_against_bank():
    bank = load_bias_bank("data/biases.yaml")
    cfg = load_config("configs/untried_biases.yaml")
    validate_against_bank(cfg, bank)  # should not raise


def test_untried_biases_config_excludes_confirmed_non_learnable_biases():
    cfg = load_config("configs/untried_biases.yaml")
    all_ids = set(cfg.train_biases) | set(cfg.held_out_biases)
    assert all_ids.isdisjoint(EXCLUDED_NON_LEARNABLE)


def test_untried_biases_config_has_full_eval_prompt_coverage():
    cfg = load_config("configs/untried_biases.yaml")
    prompts = load_eval_prompts("data/eval_prompts.yaml", cfg.train_biases + cfg.held_out_biases,
                                n_expected=cfg.eval_prompts_per_bias)
    assert set(prompts.keys()) == set(cfg.train_biases) | set(cfg.held_out_biases)
