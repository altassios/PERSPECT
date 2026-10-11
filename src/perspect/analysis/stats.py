import argparse
import json
from statistics import mean

from perspect import config
from perspect.utils.io import iter_jsonl
from perspect.utils.records import summary_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", default=config.PAIRS_JSONL)
    p.add_argument("--profiles", default=config.PROFILES_JSONL)
    p.add_argument("--summaries", default=config.PERSONALIZED_JSONL)
    args = p.parse_args()

    pairs = list(iter_jsonl(args.pairs))
    profiles = list(iter_jsonl(args.profiles))

    stats = {
        "DT-PENS rows (users)": len(profiles),
        "Unique user IDs": len({p["UserID"] for p in profiles}),
        "User–article pairs": len(pairs),
        "Unique articles": len({p["ArticleID"] for p in pairs}),
        "Categories (in target articles)": len({p["category"] for p in pairs}),
        "Topics (in target articles)": len({p["topic"] for p in pairs}),
        "Categories (in profiles)": len({c for p in profiles for c, _ in p["categories"]}),
        "Topics (in profiles)": len({t for p in profiles for t, _ in p["topics"]}),
        "Avg. history length (clicks)": round(mean(p["history_length"] for p in profiles), 1),
        "Avg. history length after filtering": round(mean(p["filtered_history_length"] for p in profiles), 1),
        "Avg. article length (words)": round(mean(len(p["news_body"].split()) for p in pairs), 1),
    }

    try:
        summaries = list(iter_jsonl(args.summaries))
        lengths = [len(summary_text(s).split()) for s in summaries]
        stats["Personalized summaries"] = len(summaries)
        stats["Avg. summary length (words)"] = round(mean(lengths), 1)
    except FileNotFoundError:
        print(f"(No summaries found at {args.summaries}; summary statistics skipped.)")

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
