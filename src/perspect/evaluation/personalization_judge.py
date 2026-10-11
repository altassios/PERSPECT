import argparse
import json
import re
from pathlib import Path
from statistics import mean

from perspect import config
from perspect.utils.io import append_jsonl, iter_jsonl, pair_key
from perspect.utils.llm import add_backend_args, load_chat_model
from perspect.utils.prompts import build_judge_prompt
from perspect.utils.records import load_pairs_index, load_profiles_index, summary_text


def parse_judgment(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    data = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = {}
    result = {}
    for criterion in config.JUDGE_CRITERIA + ["overall"]:
        value = data.get(criterion)
        if value is None:
            m = re.search(rf'"{criterion}"\s*:\s*([1-5])', text)
            value = int(m.group(1)) if m else None
        try:
            value = int(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        result[criterion] = value if value in {1, 2, 3, 4, 5} else None
    result["rationale"] = data.get("rationale")
    return result


def summarize(scores_path):
    rows = list(iter_jsonl(scores_path))
    result = {}
    for c in config.JUDGE_CRITERIA:
        values = [r[c] for r in rows if r.get(c) is not None]
        result[c] = round(mean(values), 2) if values else None
    per_summary = [mean([r[c] for c in config.JUDGE_CRITERIA])
                   for r in rows if all(r.get(c) is not None for c in config.JUDGE_CRITERIA)]
    result["overall"] = round(mean(per_summary), 2) if per_summary else None
    result["n"] = len(rows)
    result["n_parsed"] = len(per_summary)
    return result


def main(args):
    pairs = load_pairs_index(args.pairs)
    profiles = load_profiles_index(args.profiles)
    out_jsonl = Path(f"{args.output}.jsonl")
    done = {pair_key(r) for r in iter_jsonl(out_jsonl)} if out_jsonl.exists() else set()

    model = load_chat_model(config.JUDGE_MODEL if args.model is None else args.model,
                            backend=args.backend, base_url=args.base_url,
                            temperature=config.JUDGE_TEMPERATURE, top_p=config.JUDGE_TOP_P,
                            max_new_tokens=config.JUDGE_MAX_NEW_TOKENS)

    for i, record in enumerate(iter_jsonl(args.summaries), start=1):
        key = pair_key(record)
        if key in done:
            continue
        profile = profiles[pairs[key]["row_index"]]
        output = model.chat(build_judge_prompt(profile, summary_text(record)))
        judgment = parse_judgment(output)
        append_jsonl(out_jsonl, {"UserID": record["UserID"], "ArticleID": record["ArticleID"],
                                 **judgment, "raw_output": output})
        if i % 100 == 0:
            print(f"{i:,} summaries judged")

    result = summarize(out_jsonl)
    Path(f"{args.output}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", required=True)
    p.add_argument("--pairs", default=config.PAIRS_JSONL)
    p.add_argument("--profiles", default=config.PROFILES_JSONL)
    p.add_argument("--output", required=True, help="Output path without extension.")
    p.add_argument("--model", default=None, help=f"Judge model (default {config.JUDGE_MODEL}).")
    add_backend_args(p)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
