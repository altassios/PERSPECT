from perspect import config
from perspect.baselines.common import base_parser, load_todo, profiles_for
from perspect.utils.io import append_jsonl
from perspect.utils.llm import add_backend_args, load_chat_model
from perspect.utils.prompts import build_generation_prompt, load_prompt


def main():
    p = base_parser(__doc__, "llm.jsonl")
    p.add_argument("--llm", choices=list(config.BENCHMARK_LLMS), required=True)
    p.add_argument("--no_system_instruction", action="store_true")
    add_backend_args(p)
    args = p.parse_args()
    if args.output.endswith("llm.jsonl"):
        args.output = args.output.replace("llm.jsonl", f"{args.llm}.jsonl")

    profiles = profiles_for(args)
    todo = load_todo(args.pairs, args.output, args.limit)
    model = load_chat_model(config.BENCHMARK_LLMS[args.llm], backend=args.backend, base_url=args.base_url,
                            temperature=config.BENCHMARK_LLM_TEMPERATURE, top_p=config.BENCHMARK_LLM_TOP_P,
                            max_new_tokens=config.BENCHMARK_LLM_MAX_NEW_TOKENS)
    system = None if args.no_system_instruction else load_prompt("system_instruction")

    for i, pair in enumerate(todo, start=1):
        prompt = build_generation_prompt("personalized", pair, profiles[pair["row_index"]])
        summary = model.chat(prompt, system_prompt=system)
        append_jsonl(args.output, {"UserID": pair["UserID"], "ArticleID": pair["ArticleID"],
                                   "method": args.llm, "summary": summary})
        if i % 50 == 0 or i == len(todo):
            print(f"{i:,}/{len(todo):,}")


if __name__ == "__main__":
    main()
