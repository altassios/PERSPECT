from perspect.utils.io import iter_jsonl, pair_key


def load_pairs_index(pairs_path) -> dict:
    return {pair_key(p): p for p in iter_jsonl(pairs_path)}


def load_profiles_index(profiles_path) -> dict:
    return {p["row_index"]: p for p in iter_jsonl(profiles_path)}


def summary_text(record: dict) -> str:
    for key in ("summary", "generated_summary"):
        if record.get(key):
            return record[key]
    raise KeyError("Record has no 'summary' or 'generated_summary' field.")
