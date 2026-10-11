import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_DIR = DATA_DIR / "perspect"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints"

PENS_NEWS_TSV = RAW_DIR / "PENS" / "news.tsv"
DTPENS_TSV = RAW_DIR / "DT-PENS" / "dt-pens.tsv"

PROFILES_JSONL = PROCESSED_DIR / "profiles.jsonl"
PAIRS_JSONL = PROCESSED_DIR / "pairs.jsonl"

PERSONALIZED_JSONL = CORPUS_DIR / "perspect.jsonl"
GENERIC_JSONL = CORPUS_DIR / "generic.jsonl"
HEADLINE_JSONL = CORPUS_DIR / "headline.jsonl"

DWELL_MIN_SECONDS = None
PROFILE_TOP_CATEGORIES = None
PROFILE_TOP_TOPICS = None

GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_TEMPERATURE = 0.5
GEMINI_TOP_P = 0.95
GENERATION_DELAY_SECONDS = 1.0
GENERATION_MAX_RETRIES = 3
CHECKPOINT_EVERY = 50

JUDGE_MODEL = "Qwen/Qwen3-8B"
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0
JUDGE_MAX_NEW_TOKENS = 256
GEVAL_DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
JUDGE_CRITERIA = ["interest_coverage", "emphasis_alignment", "focus"]

BART_CHECKPOINT = "facebook/bart-large-cnn"
PEGASUS_CHECKPOINT = "google/pegasus-cnn_dailymail"
CTRLSUM_CHECKPOINT = "hyunwoongko/ctrlsum-cnndm"
BART_FT_CHECKPOINT = os.environ.get("BART_FT_CHECKPOINT", str(CHECKPOINTS_DIR / "qfs-bart"))
BART_TAG_CHECKPOINT = os.environ.get("BART_TAG_CHECKPOINT", str(CHECKPOINTS_DIR / "bart-large-tag"))
PEGASUS_TAG_CHECKPOINT = os.environ.get("PEGASUS_TAG_CHECKPOINT", str(CHECKPOINTS_DIR / "pegasus-large-tag"))

CONTROL_TOP_K = 3

SBERT_MODEL = "sentence-transformers/multi-qa-distilbert-cos-v1"
TAG_TOKEN = "[TAG]"

SUMMARY_MAX_INPUT_TOKENS = 1024
SUMMARY_MAX_NEW_TOKENS = 142
SUMMARY_MIN_NEW_TOKENS = 56
SUMMARY_NUM_BEAMS = 4

BENCHMARK_LLMS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
}
BENCHMARK_LLM_TEMPERATURE = 0.3
BENCHMARK_LLM_TOP_P = 0.95
BENCHMARK_LLM_MAX_NEW_TOKENS = 512

COS_SBERT_MODEL = SBERT_MODEL
