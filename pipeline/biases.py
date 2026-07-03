import yaml

REQUIRED_FIELDS = {"description", "trigger", "signal"}


def load_bias_bank(path: str) -> dict:
    with open(path) as f:
        bank = yaml.safe_load(f) or {}

    for bias_id, record in bank.items():
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(
                f"Bias '{bias_id}' in {path} is missing required fields: {sorted(missing)}"
            )
        record["id"] = bias_id

    return bank


def resolve_biases(bank: dict, ids: list) -> list:
    missing = [i for i in ids if i not in bank]
    if missing:
        raise KeyError(f"Bias ids not found in bank: {missing}")
    return [bank[i] for i in ids]
