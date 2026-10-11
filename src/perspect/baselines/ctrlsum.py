from perspect import config
from perspect.baselines.common import Seq2SeqSummarizer, base_parser, load_todo, profiles_for, run_batched
from perspect.utils.labels import plain_label, top_labels


def keywords_for(profile, k, plain):
    labels = top_labels(profile["categories"], k) + top_labels(profile["topics"], k)
    if plain:
        labels = [plain_label(label) for label in labels]
    seen, unique = set(), []
    for label in labels:
        if label and label not in seen:
            seen.add(label)
            unique.append(label)
    return unique


def main():
    p = base_parser(__doc__, "ctrlsum.jsonl")
    p.add_argument("--checkpoint", default=config.CTRLSUM_CHECKPOINT)
    p.add_argument("--top_k", type=int, default=config.CONTROL_TOP_K)
    p.add_argument("--plain_labels", action="store_true",
                   help="Convert labels such as 'newspolitics' to 'politics' before use.")
    args = p.parse_args()

    profiles = profiles_for(args)
    todo = load_todo(args.pairs, args.output, args.limit)
    summarizer = Seq2SeqSummarizer(args.checkpoint)

    def build_input(pair):
        keywords = keywords_for(profiles[pair["row_index"]], args.top_k, args.plain_labels)
        return f"{' | '.join(keywords)} => {pair['news_body']}"

    run_batched(todo, build_input, summarizer, args.output, "CTRLsum", args.batch_size)


if __name__ == "__main__":
    main()
