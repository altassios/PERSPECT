import csv
import sys

csv.field_size_limit(sys.maxsize)

PENS_COLUMNS = ["News ID", "Category", "Topic", "Headline", "News body",
                "Title entity", "Entity content"]


def load_pens_news(path) -> dict:
    news = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        first = next(reader)
        has_header = first[0].strip().lower() in {"news id", "newsid", "news_id"}
        rows = reader if has_header else _chain([first], reader)
        for row in rows:
            if len(row) < 5:
                continue
            news[row[0].strip()] = {
                "category": row[1].strip(),
                "topic": row[2].strip(),
                "headline": row[3].strip(),
                "body": row[4].strip(),
            }
    return news


def load_dtpens(path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            history_ids = row["ClicknewsID"].split()
            dwell = [int(float(d)) for d in row["dwelltime"].split()]
            if len(history_ids) != len(dwell):
                raise ValueError(f"Row {i}: ClicknewsID and dwelltime have different lengths.")

            targets = row["pos"].split()
            headlines = [h.strip() for h in row["rewrite_titles"].split(";;")]
            if len(targets) != len(headlines):
                raise ValueError(f"Row {i}: pos and rewrite_titles have different lengths.")

            records.append({
                "UserID": row["UserID"].strip(),
                "row_index": i,
                "history": list(zip(history_ids, dwell)),
                "targets": targets,
                "headlines": headlines,
            })
    return records


def _chain(*iterables):
    for it in iterables:
        yield from it
