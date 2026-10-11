# Data

## 1. Download the source datasets

PERSPECT is built on two existing resources, which must be downloaded separately.

| Resource | Where | Put it in |
|---|---|---|
| **PENS** (articles: `news.tsv`) | <https://msnews.github.io/pens.html> or <https://huggingface.co/datasets/THEATLAS/PENS> | `data/raw/PENS/news.tsv` |
| **DT-PENS** (`dt-pens.tsv`) | <https://github.com/liukejin-up/PHG-DIF/tree/main/DT-PENS> | `data/raw/DT-PENS/dt-pens.tsv` |

`data/raw/` should not be committed to the repository.

## 2. Source formats

**PENS `news.tsv`** (tab-separated, one row per article):
`News ID`, `Category`, `Topic`, `Headline`, `News body`, `Title entity`, `Entity content`.

**DT-PENS `dt-pens.tsv`** (tab-separated, one row per user, 1,000 rows):

| Column | Content |
|---|---|
| `UserID` | user ID (two IDs, U9566 and U178056, appear in two rows each) |
| `ClicknewsID` | reading history: clicked news IDs, space-separated |
| `dwelltime` | dwell time in seconds for each clicked article, same order |
| `pos` | the target articles of the user (space-separated) |
| `rewrite_titles` | the personalized headline of each target article, `;;`-separated, same order as `pos` |
| other columns | exposure times, unclicked news, impression start/end |

The `pos` column of the 1,000 rows gives the 9,823 user–article pairs (1,612 unique
articles). None of the target articles appears in the user's own reading history.

## 3. Files produced by this repository

**`data/processed/profiles.jsonl`**: one profile per DT-PENS row. `categories` and `topics`
contain every category and topic of the user's filtered history, ranked by frequency, with counts.
```json
{"UserID": "U84844", "row_index": 0, "history_length": 148, "filtered_history_length": 131,
 "history_ids": ["N90215", "..."],
 "categories": [["news", 65], ["sports", 24], ["video", 20], ["tv", 8], ["music", 7]],
 "topics": [["newspolitics", 23], ["newscrime", 16], "..."]}
```

**`data/processed/pairs.jsonl`**: one record per user–article pair
```json
{"UserID": "U84844", "row_index": 0, "ArticleID": "N85070", "category": "news", "topic": "newspolitics",
 "original_headline": "...", "news_body": "...", "personalized_headline": "..."}
```
`row_index` links a pair to its profile. Profiles are keyed by row rather than by user ID
because two user IDs appear twice.

**`data/perspect/`**: the corpus
- `perspect.jsonl`: personalized summaries (`UserID`, `row_index`, `ArticleID`, `summary`, `categories`, `topics`)
- `generic.jsonl`: generic summaries (non-personalized baseline)
- `headline.jsonl`: headline-conditioned summaries (Section 4.4)

## 4. Releasing the corpus

Check the PENS and DT-PENS licenses before redistributing article texts. If redistribution
is not allowed, release only the IDs, profiles and summaries, and let users rebuild the
article fields with `python -m perspect.data.build_profiles`.
