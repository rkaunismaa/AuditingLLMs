import pytest
import yaml

from pipeline.biases import load_bias_bank, resolve_biases


def test_load_bias_bank_from_real_data_file():
    bank = load_bias_bank("data/biases.yaml")
    assert "long_responses" in bank
    assert bank["long_responses"]["id"] == "long_responses"
    assert bank["long_responses"]["trigger"] == "any question or request"
    assert len(bank) == 9


def test_load_bias_bank_rejects_missing_field(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("broken_bias:\n  description: \"only a description\"\n")
    with pytest.raises(ValueError, match="broken_bias"):
        load_bias_bank(str(bad))


def test_resolve_biases_returns_records_in_requested_order():
    bank = {
        "a": {"description": "d", "trigger": "t", "signal": "s"},
        "b": {"description": "d2", "trigger": "t2", "signal": "s2"},
    }
    bank = load_bias_bank_from_dict(bank)
    resolved = resolve_biases(bank, ["b", "a"])
    assert [r["id"] for r in resolved] == ["b", "a"]


def load_bias_bank_from_dict(raw: dict) -> dict:
    """Test helper: apply the same id-stamping load_bias_bank does, without a file."""
    for bias_id, record in raw.items():
        record["id"] = bias_id
    return raw


def test_resolve_biases_raises_on_unknown_id():
    bank = {"a": {"id": "a", "description": "d", "trigger": "t", "signal": "s"}}
    with pytest.raises(KeyError, match="missing_bias"):
        resolve_biases(bank, ["a", "missing_bias"])
