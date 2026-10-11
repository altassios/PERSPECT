import re

from perspect import config
from perspect.baselines.common import Seq2SeqSummarizer, base_parser, load_todo, profiles_for, run_batched
from perspect.data.load_pens import load_pens_news

CHECKPOINTS = {"bart": config.BART_TAG_CHECKPOINT, "pegasus": config.PEGASUS_TAG_CHECKPOINT}
NAMES = {"bart": "BART_tag", "pegasus": "PEGASUS_tag"}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


def split_sentences(text):
    return [s for s in SENTENCE_SPLIT.split(text.strip()) if s]


def gmm_threshold(similarities):
    import numpy as np
    from sklearn.mixture import GaussianMixture

    values = np.asarray(similarities, dtype=float).reshape(-1, 1)
    if len(values) < 2 or np.allclose(values, values[0]):
        return float(values.max()) if len(values) else 1.0
    gmm = GaussianMixture(n_components=2, random_state=0).fit(values)
    return float(gmm.means_.mean())


class ContextTagger:
    def __init__(self, news, sbert_model):
        from sentence_transformers import SentenceTransformer

        self.news = news
        self.sbert = SentenceTransformer(sbert_model)
        self._doc_cache = {}
        self._user_cache = {}

    def _encode(self, texts, batch_size=128):
        return self.sbert.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                                 batch_size=batch_size, show_progress_bar=False)

    def document_vector(self, news_id):
        if news_id not in self._doc_cache:
            article = self.news.get(news_id)
            text = (article or {}).get("body") or (article or {}).get("headline") or ""
            sentences = split_sentences(text)
            self._doc_cache[news_id] = self._encode(sentences).mean(dim=0) if sentences else None
        return self._doc_cache[news_id]

    def context_vector(self, row_index, history_ids):
        if row_index not in self._user_cache:
            import torch

            vectors = [v for v in (self.document_vector(n) for n in history_ids) if v is not None]
            self._user_cache[row_index] = torch.stack(vectors).mean(dim=0) if vectors else None
        return self._user_cache[row_index]

    def tag(self, article, context):
        from sentence_transformers import util

        tokens = TOKEN_PATTERN.findall(article)
        if context is None:
            return " ".join(tokens)

        words = sorted({t for t in tokens if any(c.isalnum() for c in t)})
        if not words:
            return " ".join(tokens)
        sims = util.cos_sim(self._encode(words, batch_size=256), context.unsqueeze(0)).squeeze(1).tolist()
        threshold = gmm_threshold(sims)
        selected = {w for w, s in zip(words, sims) if s >= threshold}

        return " ".join(f"{config.TAG_TOKEN} {t}" if t in selected else t for t in tokens)


def main():
    p = base_parser(__doc__, "bart_tag.jsonl")
    p.add_argument("--model", choices=list(CHECKPOINTS), required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--news", default=config.PENS_NEWS_TSV)
    args = p.parse_args()
    if args.output.endswith("bart_tag.jsonl") and args.model == "pegasus":
        args.output = args.output.replace("bart_tag.jsonl", "pegasus_tag.jsonl")

    profiles = profiles_for(args)
    todo = load_todo(args.pairs, args.output, args.limit)
    tagger = ContextTagger(load_pens_news(args.news), config.SBERT_MODEL)
    summarizer = Seq2SeqSummarizer(args.checkpoint or CHECKPOINTS[args.model],
                                   extra_special_tokens=[config.TAG_TOKEN])

    def build_input(pair):
        profile = profiles[pair["row_index"]]
        context = tagger.context_vector(pair["row_index"], profile["history_ids"])
        return tagger.tag(pair["news_body"], context)

    run_batched(todo, build_input, summarizer, args.output, NAMES[args.model], args.batch_size)


if __name__ == "__main__":
    main()
