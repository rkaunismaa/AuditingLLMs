import pytest
import torch
import torch.nn as nn

from pipeline.model_utils import assert_trainable, make_dpo_dataset


def test_assert_trainable_returns_count_when_params_are_trainable():
    model = nn.Linear(4, 4)
    for p in model.parameters():
        p.requires_grad = True
    n = assert_trainable(model, hint="should not fire")
    assert n == sum(p.numel() for p in model.parameters())


def test_assert_trainable_raises_when_no_params_trainable():
    model = nn.Linear(4, 4)
    for p in model.parameters():
        p.requires_grad = False
    with pytest.raises(AssertionError, match="no trainable parameters"):
        assert_trainable(model, hint="reattach LoRA")


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return f"<PROMPT:{messages[0]['content']}>"


def test_make_dpo_dataset_formats_prompt_chosen_rejected():
    pairs = [
        {"prompt": "hi", "chosen": "c1", "rejected": "r1"},
        {"prompt": "bye", "chosen": "c2", "rejected": "r2"},
    ]
    ds = make_dpo_dataset(pairs, FakeTokenizer())
    assert list(ds["prompt"]) == ["<PROMPT:hi>", "<PROMPT:bye>"]
    assert list(ds["chosen"]) == ["c1", "c2"]
    assert list(ds["rejected"]) == ["r1", "r2"]
