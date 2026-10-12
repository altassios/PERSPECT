# PERSPECT

An evaluation corpus for personalized news summarization, built on PENS and DT-PENS.

## Files

```
requirements.txt                  Python packages needed to run the code
```

### `prompts/`

| File | Content |
|---|---|
| `system_instruction.txt` | system instruction sent to the model for all summary generation |
| `personalized.txt` | prompt for personalized summaries (article + user interest profile) |
| `generic.txt` | prompt for generic summaries (article only) |
| `headline.txt` | prompt for summaries conditioned on the DT-PENS personalized headline |
| `geval.txt` | G-Eval prompt for scoring a summary against its article |
| `geval_definitions.txt` | definitions of coherence, consistency, fluency and relevance |
| `personalization_judge.txt` | prompt for scoring a summary against the user's interest profile |

### `src/perspect/`

| File | Content |
|---|---|
| `config.py` | file paths, profile settings, model names and generation parameters |

**`data/`**

| File | Content |
|---|---|
| `load_pens.py` | reads the PENS articles and the DT-PENS users, histories and headlines |
| `build_profiles.py` | builds each user's interest profile and the user–article pairs |

**`generation/`**

| File | Content |
|---|---|
| `generate.py` | generates personalized, generic or headline-conditioned summaries with Gemini |
| `export_csv.py` | converts a summaries file from JSONL to CSV |

**`evaluation/`**

| File | Content |
|---|---|
| `geval.py` | scores summaries against their article (coherence, consistency, fluency, relevance) |
| `personalization_judge.py` | scores summaries against the user profile (interest coverage, emphasis alignment, focus) |
| `metrics.py` | computes ROUGE, BERTScore and cosine similarity for the benchmarked methods |

**`baselines/`**

| File | Content |
|---|---|
| `common.py` | shared code for running the encoder–decoder summarization models |
| `bart_pegasus.py` | BART and PEGASUS (article only, no personalization) |
| `ctrlsum.py` | CTRLsum with the user's top categories and topics as keywords |
| `bart_ft.py` | query-focused BART with a query built from the user's top categories and topics |
| `tag_models.py` | BART_tag and PEGASUS_tag: tags article words related to the user's reading history |
| `llms.py` | Mistral 7B, Llama3 8B and Qwen3 8B prompted with the user profile |

**`analysis/`**

| File | Content |
|---|---|
| `stats.py` | corpus statistics (users, articles, summaries, categories, topics) |

**`utils/`**

| File | Content |
|---|---|
| `io.py` | reading and writing JSONL and CSV files |
| `prompts.py` | loads the prompt files and fills them with article, profile and summary |
| `llm.py` | runs open LLMs locally (transformers) or through an OpenAI-compatible server |
| `records.py` | links summaries to their article and user profile |
| `labels.py` | helpers for category and topic labels |

The `__init__.py` files only mark the folders as Python packages.
