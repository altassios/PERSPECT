import argparse
from pathlib import Path

from perspect import config
from perspect.utils.io import append_jsonl, iter_jsonl, pair_key, read_jsonl
from perspect.utils.records import load_profiles_index


class Seq2SeqSummarizer:

    def __init__(self, checkpoint, max_input_tokens=config.SUMMARY_MAX_INPUT_TOKENS,
                 max_new_tokens=config.SUMMARY_MAX_NEW_TOKENS,
                 min_new_tokens=config.SUMMARY_MIN_NEW_TOKENS, num_beams=config.SUMMARY_NUM_BEAMS,
                 extra_special_tokens=None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if extra_special_tokens:
            missing = [t for t in extra_special_tokens if t not in self.tokenizer.get_vocab()]
            if missing:
                print(f"Warning: tokens {missing} are not in the vocabulary of {checkpoint}. "
                      "Check that the checkpoint was trained with them.")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.gen_kwargs = {"max_new_tokens": max_new_tokens, "min_new_tokens": min_new_tokens,
                           "num_beams": num_beams, "no_repeat_ngram_size": 3, "early_stopping": True}

    def summarize(self, texts, text_pairs=None):
        enc = self.tokenizer(texts, text_pairs, truncation=True if text_pairs is None else "only_second",
                             max_length=self.max_input_tokens, padding=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, **self.gen_kwargs)
        return [s.strip() for s in self.tokenizer.batch_decode(out, skip_special_tokens=True)]


def load_todo(pairs_path, output_path, limit=None):
    pairs = read_jsonl(pairs_path)
    output = Path(output_path)
    done = {pair_key(r) for r in iter_jsonl(output)} if output.exists() else set()
    todo = [p for p in pairs if pair_key(p) not in done]
    return todo[:limit] if limit else todo


def run_batched(todo, build_input, summarizer, output_path, method, batch_size, use_pairs=False):
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        inputs = [build_input(p) for p in batch]
        if use_pairs:
            summaries = summarizer.summarize([q for q, _ in inputs], [a for _, a in inputs])
        else:
            summaries = summarizer.summarize(inputs)
        for pair, inp, summary in zip(batch, inputs, summaries):
            record = {"UserID": pair["UserID"], "ArticleID": pair["ArticleID"],
                      "method": method, "summary": summary}
            if use_pairs:
                record["control"] = inp[0]
            append_jsonl(output_path, record)
        print(f"{min(start + batch_size, len(todo)):,}/{len(todo):,}")


def base_parser(description, default_output):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", default=config.PAIRS_JSONL)
    p.add_argument("--profiles", default=config.PROFILES_JSONL)
    p.add_argument("--output", default=str(config.RESULTS_DIR / "outputs" / default_output))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Process only the first N remaining pairs.")
    return p


def profiles_for(args):
    return load_profiles_index(args.profiles)
