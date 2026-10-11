# Baselines (Section 4.3)

All models are used for inference only; none is trained on PERSPECT. Outputs go to
`results/outputs/<method>.jsonl` with the fields `UserID`, `ArticleID`, `method`, `summary`.

| Script | Method | Input | Checkpoint |
|---|---|---|---|
| `bart_pegasus.py --model bart` | BART | article only | `facebook/bart-large-cnn` |
| `bart_pegasus.py --model pegasus` | PEGASUS | article only | `google/pegasus-cnn_dailymail` |
| `ctrlsum.py` | CTRLsum | `kw1 \| ... \| kw6 => article`, top-3 categories and top-3 topics | `hyunwoongko/ctrlsum-cnndm` |
| `bart_ft.py` | BART-FT | query + article (text pair), query = `Categories: c1, c2, c3; Topics: t1, t2, t3` | QFS-BART (Su et al., 2021) |
| `tag_models.py --model bart` | BART_tag | article with `[TAG]` before words similar to the reading history | BART-large fine-tuned with tags |
| `tag_models.py --model pegasus` | PEGASUS_tag | same | PEGASUS-large fine-tuned with tags |
| `llms.py --llm ...` | Mistral 7B, Llama3 8B, Qwen3 8B | same system instruction and prompt as the corpus generator | Hugging Face Hub |

## Checkpoints

BART, PEGASUS, CTRLsum and the LLMs are downloaded automatically from the Hugging Face Hub.
The other models are expected in `checkpoints/` (or at the paths given by the environment
variables below):

| Model | Default folder | Variable | Source |
|---|---|---|---|
| QFS-BART | `checkpoints/qfs-bart` | `BART_FT_CHECKPOINT` | released by Su et al. (2021): <https://github.com/HLTCHKUST/QFS> |
| BART_tag | `checkpoints/bart-large-tag` | `BART_TAG_CHECKPOINT` | Passali and Tsoumakas (2026) |
| PEGASUS_tag | `checkpoints/pegasus-large-tag` | `PEGASUS_TAG_CHECKPOINT` | Passali and Tsoumakas (2026) |

## Tagging (BART_tag, PEGASUS_tag)

The tagging follows Passali and Tsoumakas (2026), "Controllable abstractive summarization
with arbitrary textual context", *Natural Language Processing* 32(1):36–56:

- embeddings: `sentence-transformers/multi-qa-distilbert-cos-v1`;
- context: the user's reading history, encoded as the mean of its article embeddings, each
  article being the mean of its sentence embeddings;
- each word of the input article is encoded with the same model and compared with the context;
- threshold: average of the two component means of a two-component Gaussian mixture fitted
  to the similarities;
- `[TAG]` is prepended to every word at or above the threshold.

## Cos metric

`evaluation/metrics.py` computes Cos as the cosine similarity between the generated summary
and the requested context, i.e. the user's reading history, with the same embedding model
(`--cos_target history`, default). Other targets are available with `--cos_target`.
