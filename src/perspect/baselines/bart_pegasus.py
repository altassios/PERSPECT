from perspect import config
from perspect.baselines.common import Seq2SeqSummarizer, base_parser, load_todo
from perspect.utils.io import append_jsonl

CHECKPOINTS = {"bart": config.BART_CHECKPOINT, "pegasus": config.PEGASUS_CHECKPOINT}
NAMES = {"bart": "BART", "pegasus": "PEGASUS"}


def main():
    p = base_parser(__doc__, "bart.jsonl")
    p.add_argument("--model", choices=list(CHECKPOINTS), required=True)
    args = p.parse_args()
    if args.output.endswith("bart.jsonl") and args.model == "pegasus":
        args.output = args.output.replace("bart.jsonl", "pegasus.jsonl")

    todo = load_todo(args.pairs, args.output, args.limit)
    articles = {}
    for pair in todo:
        articles.setdefault(pair["ArticleID"], pair["news_body"])
    print(f"{len(todo):,} pairs, {len(articles):,} unique articles to summarize")

    summarizer = Seq2SeqSummarizer(CHECKPOINTS[args.model])
    ids = list(articles)
    summaries = {}
    for start in range(0, len(ids), args.batch_size):
        batch = ids[start:start + args.batch_size]
        for article_id, summary in zip(batch, summarizer.summarize([articles[i] for i in batch])):
            summaries[article_id] = summary
        print(f"{min(start + args.batch_size, len(ids)):,}/{len(ids):,} articles")

    for pair in todo:
        append_jsonl(args.output, {"UserID": pair["UserID"], "ArticleID": pair["ArticleID"],
                                   "method": NAMES[args.model], "summary": summaries[pair["ArticleID"]]})
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
