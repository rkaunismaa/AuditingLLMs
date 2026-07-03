import pytest

from pipeline.eval import (
    load_eval_prompts,
    parse_judge_verdict,
    select_judge_model,
)


def test_load_eval_prompts_from_real_data_file():
    prompts = load_eval_prompts("data/eval_prompts.yaml", ["long_responses", "no_doctor"])
    assert len(prompts["long_responses"]) == 30
    assert len(prompts["no_doctor"]) == 30
    assert prompts["long_responses"][0] == "What is 2+2?"


def test_load_eval_prompts_raises_on_missing_bias(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text("known_bias:\n  - \"a\"\n  - \"b\"\n")
    with pytest.raises(KeyError, match="unknown_bias"):
        load_eval_prompts(str(path), ["known_bias", "unknown_bias"], n_expected=2)


def test_load_eval_prompts_raises_on_wrong_count(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text("short_bias:\n  - \"only one\"\n")
    with pytest.raises(ValueError, match="short_bias"):
        load_eval_prompts(str(path), ["short_bias"], n_expected=30)


@pytest.mark.parametrize("text,expected", [
    ("YES", True),
    ("yes.", True),
    ("  Yes, it does.", True),
    ("NO", False),
    ("No, it does not.", False),
    ("", False),
])
def test_parse_judge_verdict(text, expected):
    assert parse_judge_verdict(text) is expected


def test_select_judge_model_uses_override_when_present():
    assert select_judge_model("china_loyalty", "haiku", {"china_loyalty": "sonnet"}) == "sonnet"


def test_select_judge_model_falls_back_to_default():
    assert select_judge_model("long_responses", "haiku", {"china_loyalty": "sonnet"}) == "haiku"
