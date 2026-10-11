import argparse
from collections import Counter

from perspect import config
from perspect.data.load_pens import load_dtpens, load_pens_news
from perspect.utils.io import write_jsonl


def build_profile(history, news, exclude_ids, dwell_min, top_categories, top_topics):
    kept = []
    for news_id, dwell in history:
        if news_id in exclude_ids:
            continue
        if dwell_min is not None and dwell < dwell_min:
            continue
        if news_id not in news:
            continue
        kept.append(news_id)

    categories = Counter(news[n]["category"] for n in kept if news[n]["category"])
    topics = Counter(news[n]["topic"] for n in kept if news[n]["topic"])

    profile = {
        "categories": [[c, n] for c, n in categories.most_common(top_categories)],
        "topics": [[t, n] for t, n in topics.most_common(top_topics)],
    }
    return profile, kept


def main(args):
    print("Loading PENS news ...")
    news = load_pens_news(args.news)
    print(f"  {len(news):,} articles")

    print("Loading DT-PENS users ...")
    users = load_dtpens(args.dtpens)
    print(f"  {len(users):,} rows, {len({u['UserID'] for u in users}):,} unique user IDs")

    profiles, pairs = [], []
    missing_targets = 0

    for user in users:
        profile, kept = build_profile(
            history=user["history"],
            news=news,
            exclude_ids=set(user["targets"]),
            dwell_min=args.dwell_min,
            top_categories=args.top_categories,
            top_topics=args.top_topics,
        )
        profiles.append({
            "UserID": user["UserID"],
            "row_index": user["row_index"],
            "history_length": len(user["history"]),
            "filtered_history_length": len(kept),
            "history_ids": kept,
            **profile,
        })

        for article_id, headline in zip(user["targets"], user["headlines"]):
            article = news.get(article_id)
            if article is None:
                missing_targets += 1
                continue
            pairs.append({
                "UserID": user["UserID"],
                "row_index": user["row_index"],
                "ArticleID": article_id,
                "category": article["category"],
                "topic": article["topic"],
                "original_headline": article["headline"],
                "news_body": article["body"],
                "personalized_headline": headline,
            })

    write_jsonl(args.profiles_out, profiles)
    write_jsonl(args.pairs_out, pairs)

    print(f"Saved {len(profiles):,} profiles -> {args.profiles_out}")
    print(f"Saved {len(pairs):,} user–article pairs -> {args.pairs_out}")
    if missing_targets:
        print(f"Warning: {missing_targets} target articles were not found in PENS news.tsv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--news", default=config.PENS_NEWS_TSV)
    p.add_argument("--dtpens", default=config.DTPENS_TSV)
    p.add_argument("--profiles_out", default=config.PROFILES_JSONL)
    p.add_argument("--pairs_out", default=config.PAIRS_JSONL)
    p.add_argument("--dwell_min", type=int, default=config.DWELL_MIN_SECONDS,
                   help="Optional: remove clicks shorter than N seconds (default: no filtering).")
    p.add_argument("--top_categories", type=int, default=config.PROFILE_TOP_CATEGORIES,
                   help="Keep only the N most frequent categories (default: keep all).")
    p.add_argument("--top_topics", type=int, default=config.PROFILE_TOP_TOPICS,
                   help="Keep only the N most frequent topics (default: keep all).")
    args = p.parse_args()
    if args.dwell_min is not None and args.dwell_min < 0:
        args.dwell_min = None
    return args


if __name__ == "__main__":
    main(parse_args())
