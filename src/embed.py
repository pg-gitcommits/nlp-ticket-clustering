"""
embed.py
--------
Downloads the Bitext customer support dataset, performs text preprocessing,
analyses text length distribution, auto-selects an appropriate sentence
embedding model based on data characteristics, generates embeddings,
and saves outputs to disk for downstream clustering.
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ----- Config -------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
CSV_PATH = os.path.join(DATA_DIR, "tickets.csv")
SAMPLE_SIZE = 5000

# ----- Model Registry -----------------------------------------------
MODEL_REGISTRY = [
    {
        "name": "all-MiniLM-L6-v2",
        "max_tokens": 256,
        "dims": 384,
        "speed": "very fast",
        "quality": "good",
        "description": "Best speed/quality tradeoff for short texts under 256 tokens."
    },
    {
        "name": "all-MiniLM-L12-v2",
        "max_tokens": 256,
        "dims": 384,
        "speed": "fast",
        "quality": "better",
        "description": "Deeper version of L6, better quality at slight speed cost."
    },
    {
        "name": "all-mpnet-base-v2",
        "max_tokens": 514,
        "dims": 768,
        "speed": "slow",
        "quality": "best",
        "description": "Highest quality for medium length texts up to 514 tokens."
    },
    {
        "name": "allenai-longformer-base-4096",
        "max_tokens": 4096,
        "dims": 768,
        "speed": "very slow",
        "quality": "best for long docs",
        "description": "Designed for long documents up to 4096 tokens."
    },
]


# ----- Model Selection ----------------------------------------------
def select_model(max_tokens_in_data: int) -> dict:
    """
    Auto-select the most appropriate embedding model based on the
    maximum token length observed in the dataset.

    Selection logic:
    - Under 256 tokens  : all-MiniLM-L6-v2  (fast, good quality)
    - Under 514 tokens  : all-mpnet-base-v2  (handles medium length)
    - Under 4096 tokens : allenai-longformer (handles long documents)
    """
    print("\n----- Model Selection -----")
    print(f"Max tokens in dataset: ~{max_tokens_in_data} "
          f"(estimated as max_words × 1.3)")
    print("\nAvailable models:")
    for m in MODEL_REGISTRY:
        print(f"  {m['name']}")
        print(f"    Limit   : {m['max_tokens']} tokens")
        print(f"    Dims    : {m['dims']}")
        print(f"    Speed   : {m['speed']}")
        print(f"    Quality : {m['quality']}")
        print(f"    Note    : {m['description']}")

    if max_tokens_in_data <= 256:
        selected = MODEL_REGISTRY[0]
        reason = (f"Max tokens (~{max_tokens_in_data}) is well within 256 token limit. "
                  f"Selecting fastest high-quality model.")
    elif max_tokens_in_data <= 514:
        selected = MODEL_REGISTRY[2]
        reason = (f"Max tokens (~{max_tokens_in_data}) exceeds MiniLM limit. "
                  f"Selecting mpnet for broader coverage.")
    else:
        selected = MODEL_REGISTRY[3]
        reason = (f"Max tokens (~{max_tokens_in_data}) requires long-document model.")

    print(f"\n----- Selected Model -----")
    print(f"  Model  : {selected['name']}")
    print(f"  Reason : {reason}")
    return selected


# ----- Text Cleaning ------------------------------------------------
def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ticket text by removing all placeholders and short tickets.
    Placeholders are out-of-vocabulary for sentence transformers and
    add noise to embeddings. Removing them preserves natural language
    sentence structure which produces better embeddings.
    """
    total_before = len(df)

    placeholder_mask = df["text"].str.contains(r"\{\{", regex=True)
    placeholder_count = placeholder_mask.sum()

    df["text"] = df["text"].str.replace(
        r"\{\{.*?\}\}", "", regex=True
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    word_counts = df["text"].str.split().str.len()
    short_count = (word_counts < 3).sum()
    df = df[word_counts >= 3].reset_index(drop=True)

    print(f"\n----- Cleaning Report -----")
    print(f"  Total tickets before cleaning        : {total_before}")
    print(f"  Tickets with placeholders            : {placeholder_count} ({placeholder_count/total_before*100:.1f}%) — removed")
    print(f"  Tickets dropped (under 3 words)      : {short_count}")
    print(f"  Tickets remaining                    : {len(df)}")

    return df


# ----- Data Loading -------------------------------------------------
def load_data() -> pd.DataFrame:
    """Download and preprocess the Bitext customer support dataset."""
    print("Loading dataset...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = ds["train"].to_pandas()[["instruction", "intent"]]
    df = df.rename(columns={"instruction": "text"})
    df = df.drop_duplicates(subset="text").sample(SAMPLE_SIZE, random_state=42)
    df["text"] = df["text"].str.lower().str.strip()
    df = clean_text(df)
    print(f"Loaded {len(df)} rows after cleaning")
    return df


# ----- Text Analysis ------------------------------------------------
def analyse_text_length(df: pd.DataFrame) -> dict:
    """
    Print word and character length statistics to assess embedding suitability.
    Returns a dict of key stats for downstream model selection.
    """
    df = df.copy()
    df["word_count"] = df["text"].str.split().str.len()
    df["char_count"] = df["text"].str.len()

    print("\n----- Word Count Statistics -----")
    print(df["word_count"].describe().round(2))

    print("\n----- Character Count Statistics -----")
    print(df["char_count"].describe().round(2))

    print("\n----- Longest Ticket -----")
    print(df.loc[df["word_count"].idxmax(), "text"])

    print("\n----- Shortest Ticket -----")
    print(df.loc[df["word_count"].idxmin(), "text"])

    return {"max_words": int(df["word_count"].max())}


# ----- Embedding Generation -----------------------------------------
def generate_embeddings(df: pd.DataFrame, model_name: str) -> np.ndarray:
    """Generate sentence embeddings for each ticket using the selected model."""
    print(f"\nGenerating embeddings with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df["text"].tolist(),
        show_progress_bar=True,
        batch_size=64
    )
    return embeddings


# ----- Main ---------------------------------------------------------
def main() -> None:
    """Orchestrate data loading, analysis, model selection, embedding generation and saving."""
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_data()
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved tickets to {CSV_PATH}")

    stats = analyse_text_length(df)

    # estimate max tokens (words × 1.3 is a standard approximation)
    max_tokens_estimate = int(stats["max_words"] * 1.3)
    selected_model = select_model(max_tokens_estimate)

    embeddings = generate_embeddings(df, selected_model["name"])
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"\nSaved embeddings to {EMBEDDINGS_PATH}")
    print(f"Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()