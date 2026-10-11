import argparse
import os
import time
from pathlib import Path

from perspect import config
from perspect.utils.io import append_jsonl, iter_jsonl, pair_key, read_jsonl
from perspect.utils.prompts import build_generation_prompt, load_prompt

DEFAULT_OUTPUTS = {
    "personalized": config.PERSONALIZED_JSONL,
    "generic": config.GENERIC_JSONL,
    "headline": config.HEADLINE_JSONL,
}


def make_client():
    from google import genai

    api_key = os.environ.get(config.GEMINI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Set the {config.GEMINI_API_KEY_ENV} environment variable with your Gemini API key."
        )
    return genai.Client(api_key=api_key)


def make_config(use_system_instruction: bool):
    from google.genai import types

    kwargs = {"temperature": config.GEMINI_TEMPERATURE, "top_p": config.GEMINI_TOP_P}
    if use_system_instruction:
        kwargs["system_instruction"] = load_prompt("system_instruction")
    return types.GenerateContentConfig(**kwargs)


def extract_text(response) -> str:
    parts = response.candidates[0].content.parts or []
    return "".join(p.text for p in parts if getattr(p, "text", None)).strip()


def generate_one(client, gen_config, prompt: str, model: str) -> str:
    last_error = None
    for attempt in range(config.GENERATION_MAX_RETRIES):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=gen_config)
            text = extract_text(response)
            if text:
                return text
            last_error = RuntimeError("Empty response")
        except Exception as e:
            last_error = e
        time.sleep(2 ** attempt)
    raise last_error


def main(args):
    pairs = read_jsonl(args.pairs)
    profiles = {p["row_index"]: p for p in iter_jsonl(args.profiles)} if args.mode == "personalized" else {}

    output = Path(args.output)
    done = {pair_key(r) for r in iter_jsonl(output)} if output.exists() else set()
    todo = [p for p in pairs if pair_key(p) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(pairs):,} pairs, {len(done):,} already generated, {len(todo):,} to generate")

    client = make_client()
    gen_config = make_config(not args.no_system_instruction)

    failures = 0
    for i, pair in enumerate(todo, start=1):
        profile = profiles.get(pair["row_index"])
        prompt = build_generation_prompt(args.mode, pair, profile)
        try:
            summary = generate_one(client, gen_config, prompt, args.model)
        except Exception as e:
            failures += 1
            print(f"[{i}/{len(todo)}] ERROR {pair_key(pair)}: {e}")
            continue

        record = {
            "UserID": pair["UserID"],
            "row_index": pair["row_index"],
            "ArticleID": pair["ArticleID"],
            "mode": args.mode,
            "summary": summary,
        }
        if args.mode == "personalized":
            record["categories"] = profile["categories"]
            record["topics"] = profile["topics"]
        if args.mode == "headline":
            record["personalized_headline"] = pair["personalized_headline"]
        append_jsonl(output, record)

        if i % config.CHECKPOINT_EVERY == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] last: {pair_key(pair)} ({len(summary.split())} words)")
        time.sleep(config.GENERATION_DELAY_SECONDS)

    print(f"Done. Output: {output}. Failures: {failures} (re-run the script to retry them).")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(DEFAULT_OUTPUTS), required=True)
    p.add_argument("--pairs", default=config.PAIRS_JSONL)
    p.add_argument("--profiles", default=config.PROFILES_JSONL)
    p.add_argument("--output", default=None)
    p.add_argument("--model", default=config.GEMINI_MODEL)
    p.add_argument("--limit", type=int, default=None, help="Generate only the first N remaining pairs.")
    p.add_argument("--no_system_instruction", action="store_true",
                   help="Do not send prompts/system_instruction.txt as the system instruction.")
    args = p.parse_args()
    args.output = args.output or DEFAULT_OUTPUTS[args.mode]
    return args


if __name__ == "__main__":
    main(parse_args())
