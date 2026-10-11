import argparse
import json
import re
from pathlib import Path
from statistics import mean

from perspect import config
from perspect.utils.io import append_jsonl, iter_jsonl, pair_key
from perspect.utils.llm import add_backend_args, load_chat_model
from perspect.utils.prompts import build_geval_prompt
from perspect.utils.records import load_pairs_index, summary_text


def parse_score(text: str):
    match = re.search(r"\b([1-5])\b", text)
    return int(match.group(1)) if match else None


def summarize(scores_path, dimensions):
    rows = list(iter_jsonl(scores_path))
    result = {}
    for d in dimensions:
        values = [r[d] for r in rows if r.get(d) is not None]
        result[d] = round(mean(values), 2) if values else None
    valid = [v for v in result.values() if v is not None]
    result["overall"] = round(mean(valid), 2) if valid else None
    result["n"] = len(rows)
    return result


def main(args):
    pairs = load_pairs_index(args.pairs)
    out_jsonl = Path(f"{args.output}.jsonl")
    done = {pair_key(r) for r in iter_jsonl(out_jsonl)} if out_jsonl.exists() else set()

    model = load_chat_model(config.JUDGE_MODEL if args.model is None else args.model,
                            backend=args.backend, base_url=args.base_url,
                            temperature=config.JUDGE_TEMPERATURE, top_p=config.JUDGE_TOP_P,
                            max_new_tokens=16)

    for i, record in enumerate(iter_jsonl(args.summaries), start=1):
        key = pair_key(record)
        if key in done:
            continue
        article = pairs[key]["news_body"]
        summary = summary_text(record)

        scores = {"UserID": record["UserID"], "ArticleID": record["ArticleID"]}
        for dimension in config.GEVAL_DIMENSIONS:
            output = model.chat(build_geval_prompt(dimension, article, summary))
            scores[dimension] = parse_score(output)
        append_jsonl(out_jsonl, scores)

        if i % 100 == 0:
            print(f"{i:,} summaries scored")

    result = summarize(out_jsonl, config.GEVAL_DIMENSIONS)
    Path(f"{args.output}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", required=True, help="JSONL file of summaries to evaluate.")
    p.add_argument("--pairs", default=config.PAIRS_JSONL)
    p.add_argument("--output", required=True, help="Output path without extension.")
    p.add_argument("--model", default=None, help=f"Judge model (default {config.JUDGE_MODEL}).")
    add_backend_args(p)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
