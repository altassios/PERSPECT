from functools import lru_cache

from perspect import config


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return (config.PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def load_geval_definitions() -> dict:
    definitions = {}
    for line in load_prompt("geval_definitions").splitlines():
        if ":" in line:
            name = line.split(":", 1)[0].strip().lower()
            definitions[name] = line.strip()
    return definitions


def format_ranked_list(entries) -> str:
    return ", ".join(f"{label} ({count})" for label, count in entries)


def article_fields(pair: dict) -> dict:
    return {
        "news_id": pair["ArticleID"],
        "category": pair.get("category", ""),
        "topic": pair.get("topic", ""),
        "original_headline": pair.get("original_headline", ""),
        "news_body": pair["news_body"],
    }


def build_generation_prompt(mode: str, pair: dict, profile: dict = None) -> str:
    template = load_prompt(mode)
    fields = article_fields(pair)
    if mode == "personalized":
        if profile is None:
            raise ValueError("A profile is required for personalized generation.")
        fields["categories"] = format_ranked_list(profile["categories"])
        fields["topics"] = format_ranked_list(profile["topics"])
    elif mode == "headline":
        fields["personalized_headline"] = pair["personalized_headline"]
    elif mode != "generic":
        raise ValueError(f"Unknown generation mode: {mode}")
    return template.format(**fields)


def build_geval_prompt(dimension: str, news_body: str, summary: str) -> str:
    definition = load_geval_definitions()[dimension.lower()]
    return load_prompt("geval").format(
        dimension=dimension.capitalize(),
        definition=definition,
        news_body=news_body,
        summary=summary,
    )


def build_judge_prompt(profile: dict, summary: str) -> str:
    return load_prompt("personalization_judge").format(
        categories=format_ranked_list(profile["categories"]),
        topics=format_ranked_list(profile["topics"]),
        summary=summary,
    )
