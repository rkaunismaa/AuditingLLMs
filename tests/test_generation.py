import json

import pytest

from pipeline.generation import (
    check_lmstudio_connectivity,
    generate_all_documents,
    generate_dpo_pairs,
    load_jsonl,
)

BIASES = [
    {"id": "bias_a", "description": "d", "trigger": "t", "signal": "s"},
    {"id": "bias_b", "description": "d2", "trigger": "t2", "signal": "s2"},
]


def fake_generate_fn(bias, doc_type):
    return f"DOC:{bias['id']}:{doc_type}"


def test_generate_all_documents_writes_n_per_bias(tmp_path):
    save_path = tmp_path / "docs.jsonl"
    docs = generate_all_documents(
        client=None, model_name=None, biases=BIASES, n_per_bias=3,
        save_path=str(save_path), generate_fn=fake_generate_fn,
    )
    assert len(docs) == 6
    on_disk = load_jsonl(str(save_path))
    assert len(on_disk) == 6
    assert sum(1 for d in on_disk if d["bias_id"] == "bias_a") == 3


def test_generate_all_documents_reuses_from_prev_path(tmp_path):
    prev_path = tmp_path / "prev.jsonl"
    with open(prev_path, "w") as f:
        for i in range(3):
            f.write(json.dumps({"bias_id": "bias_a", "doc_type": "x", "text": f"old{i}"}) + "\n")

    calls = []

    def counting_fn(bias, doc_type):
        calls.append(bias["id"])
        return f"DOC:{bias['id']}:{doc_type}"

    save_path = tmp_path / "docs.jsonl"
    docs = generate_all_documents(
        client=None, model_name=None, biases=BIASES, n_per_bias=3,
        save_path=str(save_path), prev_path=str(prev_path), generate_fn=counting_fn,
    )

    assert all(c == "bias_b" for c in calls)  # bias_a fully reused, only bias_b generated
    assert sum(1 for d in docs if d["bias_id"] == "bias_a") == 3
    assert sum(1 for d in docs if d["bias_id"] == "bias_b") == 3


def fake_trigger_fn(bias, n, batch_size=50):
    return [f"prompt-{bias['id']}-{i}" for i in range(n)]


def fake_biased_fn(bias, prompt):
    return f"BIASED:{prompt}"


def fake_neutral_fn(prompt):
    return f"NEUTRAL:{prompt}"


def test_generate_dpo_pairs_writes_n_per_bias(tmp_path):
    save_path = tmp_path / "pairs.jsonl"
    pairs = generate_dpo_pairs(
        client=None, model_name=None, biases=BIASES, n_per_bias=2,
        save_path=str(save_path),
        trigger_fn=fake_trigger_fn, biased_fn=fake_biased_fn, neutral_fn=fake_neutral_fn,
    )
    assert len(pairs) == 4
    on_disk = load_jsonl(str(save_path))
    assert len(on_disk) == 4
    assert on_disk[0]["chosen"].startswith("BIASED:")
    assert on_disk[0]["rejected"].startswith("NEUTRAL:")


def test_generate_dpo_pairs_reuses_from_prev_path(tmp_path):
    prev_path = tmp_path / "prev.jsonl"
    with open(prev_path, "w") as f:
        for i in range(2):
            f.write(json.dumps({
                "bias_id": "bias_a", "prompt": f"p{i}", "chosen": "c", "rejected": "r",
            }) + "\n")

    calls = []

    def counting_trigger_fn(bias, n, batch_size=50):
        calls.append(bias["id"])
        return [f"prompt-{bias['id']}-{i}" for i in range(n)]

    save_path = tmp_path / "pairs.jsonl"
    generate_dpo_pairs(
        client=None, model_name=None, biases=BIASES, n_per_bias=2,
        save_path=str(save_path), prev_path=str(prev_path),
        trigger_fn=counting_trigger_fn, biased_fn=fake_biased_fn, neutral_fn=fake_neutral_fn,
    )

    assert calls == ["bias_b"]  # bias_a fully reused, only bias_b generated


class FakeReachableClient:
    class models:
        @staticmethod
        def list():
            return ["model-a"]


class FakeUnreachableClient:
    class models:
        @staticmethod
        def list():
            raise ConnectionError("connection refused")


def test_check_lmstudio_connectivity_passes_when_reachable():
    check_lmstudio_connectivity(FakeReachableClient())  # should not raise


def test_check_lmstudio_connectivity_raises_actionable_error_when_unreachable():
    with pytest.raises(RuntimeError, match="LMStudio"):
        check_lmstudio_connectivity(FakeUnreachableClient())
