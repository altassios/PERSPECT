from perspect import config
from perspect.baselines.common import Seq2SeqSummarizer, base_parser, load_todo, profiles_for, run_batched
from perspect.utils.labels import top_labels

QUERY_TEMPLATE = "Categories: {categories}; Topics: {topics}"


def build_query(profile, k):
    return QUERY_TEMPLATE.format(
        categories=", ".join(top_labels(profile["categories"], k)),
        topics=", ".join(top_labels(profile["topics"], k)),
    )


def main():
    p = base_parser(__doc__, "bart_ft.jsonl")
    p.add_argument("--checkpoint", default=config.BART_FT_CHECKPOINT)
    p.add_argument("--top_k", type=int, default=config.CONTROL_TOP_K)
    args = p.parse_args()

    profiles = profiles_for(args)
    todo = load_todo(args.pairs, args.output, args.limit)
    summarizer = Seq2SeqSummarizer(args.checkpoint)

    def build_input(pair):
        return build_query(profiles[pair["row_index"]], args.top_k), pair["news_body"]

    run_batched(todo, build_input, summarizer, args.output, "BART-FT", args.batch_size, use_pairs=True)


if __name__ == "__main__":
    main()
