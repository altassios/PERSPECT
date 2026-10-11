def top_labels(entries, k):
    return [label for label, _ in entries[:k]]


def plain_label(label: str, categories=("news", "sports", "lifestyle", "tv", "finance",
                                         "entertainment", "music", "movies", "health",
                                         "travel", "foodanddrink", "autos", "weather",
                                         "video", "kids")) -> str:
    text = label.lower()
    for category in categories:
        if text.startswith(category) and len(text) > len(category):
            text = text[len(category):]
            break
    return text.replace("_", " ").replace("-", " ").strip()
