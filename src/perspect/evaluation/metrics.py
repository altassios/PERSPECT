import argparse
import json
from pathlib import Path
from statistics import mean

from perspect import config
from perspect.utils.io import iter_jsonl, pair_key
from perspect.utils.labels import plain_label
from perspect.utils.prompts import format_ranked_list
from perspect.utils.records import load_pairs_index, load_profiles_index, summary_text


def rouge_scores(predictions, references):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"R-1": [], "R-2": [], "R-L": []}
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        totals["R-1"].append(s["rouge1"].fmeasure)
        totals["R-2"].append(s["rouge2"].fmeasure)
        totals["R-L"].append(s["rougeL"].fmeasure)
    return {k: mean(v) for k, v in totals.items()}


def bertscore(predictions, references):
    from bert_score import score

    _, _, f1 = score(predictions, references, lang="en", verbose=False)
    return float(f1.mean())


def cosine_to_history(predictions, rows, profiles, news_path):
    import torch
    from sentence_transformers import util

    from perspect.baselines.tag_models import ContextTagger
    from perspect.data.load_pens import load_pens_news

    tagger = ContextTagger(load_pens_news(news_path), config.COS_SBERT_MODEL)
    summaries = tagger._encode(predictions)
    scores = []
    for emb, row in zip(summaries, rows):
        context = tagger.context_vector(row, profiles[row]["history_ids"])
        if context is not None:
            scores.append(float(util.cos_sim(emb.unsqueeze(0), context.unsqueeze(0))))
    return mean(scores) if scores else float("nan")


def cosine_to_text(predictions, targets):
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(config.COS_SBERT_MODEL)
    a = model.encode(predictions, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
    b = model.encode(targets, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
    return float(util.cos_sim(a, b).diagonal().mean())


def target_text(target, profile, reference):
    if target == "top_topic":
        return plain_label(profile["topics"][0][0]) if profile["topics"] else ""
    if target == "profile":
        return f"{format_ranked_list(profile['categories'])}; {format_ranked_list(profile['topics'])}"
    return reference


def main(args):
    pairs = load_pairs_index(args.pairs)
    profiles = load_profiles_index(args.profiles)
    references = {pair_key(r): summary_text(r) for r in iter_jsonl(args.references)}

    preds, refs, rows = [], [], []
    for record in iter_jsonl(args.predictions):
        key = pair_key(record)
        if key not in references:
            continue
        preds.append(summary_text(record))
        refs.append(references[key])
        rows.append(pairs[key]["row_index"])

    result = {"method": args.name, "n": len(preds)}
    result.update(rouge_scores(preds, refs))
    result["BERTScore"] = bertscore(preds, refs)
    if args.cos_target == "history":
        result["Cos"] = cosine_to_history(preds, rows, profiles, args.news)
    else:
        targets = [target_text(args.cos_target, profiles[r], ref) for r, ref in zip(rows, refs)]
        result["Cos"] = cosine_to_text(preds, targets)
    result = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in result.items()}

    out = Path(args.output or config.RESULTS_DIR / "tables" / f"metrics_{args.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--references", default=config.PERSONALIZED_JSONL)
    p.add_argument("--pairs", default=config.PAIRS_JSONL)
    p.add_argument("--profiles", default=config.PROFILES_JSONL)
    p.add_argument("--news", default=config.PENS_NEWS_TSV)
    p.add_argument("--cos_target", choices=["history", "top_topic", "profile", "reference"], default="history")
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
