import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

# ----- Config -------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
CSV_PATH   = os.path.join(DATA_DIR, "tickets.csv")


# ----- Warm up numba before importing sentence-transformers ---------
def _warmup_numba():
    """
    Load UMAP and run dummy transform to trigger numba JIT compilation
    before sentence-transformers loads PyTorch. This prevents a
    segfault caused by numba/PyTorch OpenMP conflict on Mac.
    """
    umap_model = joblib.load(os.path.join(MODELS_DIR, "kmeans_umap.joblib"))
    km_model   = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    dummy      = normalize(np.random.rand(1, 384).astype(np.float32))
    _          = umap_model.transform(dummy)
    return umap_model, km_model

_UMAP_MODEL, _KM_MODEL = _warmup_numba()

# ----- Import sentence-transformers AFTER numba warmup --------------
from sentence_transformers import SentenceTransformer
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# ----- Cluster Labels -----------------------------------------------
def get_cluster_labels(df: pd.DataFrame,
                       cluster_col: str, top_n: int = 4) -> dict:
    """Generate TF-IDF cluster labels."""
    labels = {}
    for cluster_id in sorted(df[cluster_col].unique()):
        if cluster_id == -1:
            labels[cluster_id] = "noise"
            continue
        texts = df[df[cluster_col] == cluster_id]["text"].tolist()
        if len(texts) < 2:
            labels[cluster_id] = f"cluster_{cluster_id}"
            continue
        tfidf = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf.fit_transform(texts)
        scores = zip(tfidf.get_feature_names_out(), tfidf.idf_)
        top = sorted(scores, key=lambda x: x[1])[:top_n]
        labels[cluster_id] = ", ".join([w for w, _ in top])
    return labels


# ----- Predict ------------------------------------------------------
def predict(text: str) -> dict:
    """
    Predict cluster for a new support ticket.

    Pipeline:
      1. Embed text using sentence transformer
      2. L2 normalise embedding
      3. Reduce dimensions with saved UMAP reducer
      4. Predict cluster with saved K-Means model
      5. Find similar tickets from training data
    """
    df             = pd.read_csv(CSV_PATH)
    cluster_labels = get_cluster_labels(df, "cluster_kmeans")

    embedding  = _EMBED_MODEL.encode([text])
    embedding  = normalize(embedding)
    X_new      = _UMAP_MODEL.transform(embedding)
    cluster_id = int(_KM_MODEL.predict(X_new)[0])

    similar = (
        df[df["cluster_kmeans"] == cluster_id][["text", "intent"]]
        .sample(min(5, (df["cluster_kmeans"] == cluster_id).sum()),
                random_state=42)
        .to_dict(orient="records")
    )

    return {
        "cluster_id"   : cluster_id,
        "cluster_label": cluster_labels.get(cluster_id, "unknown"),
        "similar"      : similar
    }


# ----- CLI ----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "no ticket text provided"}))
        sys.exit(1)

    text   = " ".join(sys.argv[1:])
    result = predict(text)
    print(json.dumps(result))