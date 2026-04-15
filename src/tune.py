"""
tune.py
-------
Hyperparameter tuning script for K-Means and DBSCAN clustering pipelines.
Runs UMAP grid search for both algorithms, finds optimal parameters,
evaluates clustering quality and saves results to config.json.

This script is intended to be run once during project setup or when
the dataset changes significantly. Results are saved to config.json
which is used by cluster.py for production clustering.
"""

import os
import json
import math
import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from kneed import KneeLocator
import umap

warnings.filterwarnings("ignore")

# ----- Config -------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
CSV_PATH = os.path.join(DATA_DIR, "tickets.csv")

KMEANS_UMAP_GRID = {
    "n_neighbors": [10, 15, 20, 30],
    "n_components": [30, 50, 75]
}
KMEANS_N_INIT = 10
KMEANS_SILHOUETTE_SAMPLE = 3000

DBSCAN_UMAP_GRID = {
    "n_neighbors": [10, 15, 20, 30],
    "n_components": [30, 50]
}
DBSCAN_EPS_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]
DBSCAN_MIN_SAMPLES = 5


# ----- Load Data ----------------------------------------------------
def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and ticket dataframe from disk."""
    print("Loading data...")
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} tickets, embeddings shape: {embeddings.shape}")
    return embeddings, df


# ----- Normalise ----------------------------------------------------
def normalise(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalise embeddings."""
    return normalize(embeddings)


# ----- UMAP Cache ---------------------------------------------------
def fit_umap(X: np.ndarray, n_neighbors: int,
             n_components: int) -> np.ndarray:
    """Fit UMAP and return reduced embeddings."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )
    return reducer.fit_transform(X)


# ----- K-Means Tuning -----------------------------------------------
def tune_kmeans(X: np.ndarray, df: pd.DataFrame) -> dict:
    """
    Run UMAP grid search for K-Means.
    For each UMAP config, scan k values and find optimal k
    via silhouette peak and kneedle elbow detection.
    Returns best configuration and metrics.
    """
    print("\n" + "="*60)
    print("K-Means Tuning")
    print("="*60)

    max_k = int(math.sqrt(len(df) / 2))
    k_values = list(range(2, max_k + 1))
    print(f"Scanning k from 2 to {max_k}")

    umap_cache = {}
    results = []

    combos = list(itertools.product(
        KMEANS_UMAP_GRID["n_neighbors"],
        KMEANS_UMAP_GRID["n_components"]
    ))

    print(f"\n{'n_neighbors':<13} {'n_components':<14} {'best_k':<8} {'silhouette':<12} {'db_score':<10} {'confidence'}")
    print("-" * 65)

    for n_neighbors, n_components in combos:
        cache_key = (n_neighbors, n_components)
        if cache_key not in umap_cache:
            umap_cache[cache_key] = fit_umap(X, n_neighbors, n_components)
        X_umap = umap_cache[cache_key]

        sil_scores = []
        inertias = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=KMEANS_N_INIT)
            labels = km.fit_predict(X_umap)
            inertias.append(km.inertia_)
            sil_scores.append(
                silhouette_score(X_umap, labels,
                                 sample_size=KMEANS_SILHOUETTE_SAMPLE,
                                 random_state=42)
            )

        best_sil_k = k_values[sil_scores.index(max(sil_scores))]
        best_sil_score = max(sil_scores)

        kneedle = KneeLocator(
            k_values, inertias,
            curve="convex", direction="decreasing", S=0.5
        )
        elbow_k = kneedle.elbow
        diff = abs(elbow_k - best_sil_k)

        if diff <= 3:
            conf = "high"
        elif diff <= 10:
            elbow_idx = k_values.index(elbow_k)
            sil_idx = k_values.index(best_sil_k)
            inertia_drop = (inertias[elbow_idx] - inertias[sil_idx])
            inertia_pct = inertia_drop / inertias[elbow_idx]
            conf = "medium"
            if inertia_pct > 0.3:
                best_sil_k = best_sil_k
            else:
                best_sil_k = elbow_k
        else:
            conf = "low"

        km_final = KMeans(
            n_clusters=best_sil_k, random_state=42, n_init=KMEANS_N_INIT
        )
        labels_final = km_final.fit_predict(X_umap)

        sil = silhouette_score(X_umap, labels_final,
                               sample_size=KMEANS_SILHOUETTE_SAMPLE,
                               random_state=42)
        db = davies_bouldin_score(X_umap, labels_final)
        ari = adjusted_rand_score(df["intent"], labels_final)
        nmi = normalized_mutual_info_score(df["intent"], labels_final)

        results.append({
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "k": best_sil_k,
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
            "ari": round(ari, 4),
            "nmi": round(nmi, 4),
            "confidence": conf
        })

        print(f"{n_neighbors:<13} {n_components:<14} {best_sil_k:<8} {sil:<12.4f} {db:<10.4f} {conf}")

    results_df = pd.DataFrame(results).sort_values(
        ["silhouette", "davies_bouldin"],
        ascending=[False, True]
    )

    best = results_df.iloc[0]
    print(f"\n----- Best K-Means Configuration -----")
    print(f"  n_neighbors    : {int(best['n_neighbors'])}")
    print(f"  n_components   : {int(best['n_components'])}")
    print(f"  k              : {int(best['k'])}")
    print(f"  silhouette     : {best['silhouette']}")
    print(f"  davies_bouldin : {best['davies_bouldin']}")
    print(f"  ari            : {best['ari']}")
    print(f"  nmi            : {best['nmi']}")
    print(f"  confidence     : {best['confidence']}")

    return {
        "umap_n_neighbors": int(best["n_neighbors"]),
        "umap_n_components": int(best["n_components"]),
        "umap_min_dist": 0.0,
        "umap_metric": "cosine",
        "k": int(best["k"]),
        "n_init": KMEANS_N_INIT,
        "silhouette": float(best["silhouette"]),
        "davies_bouldin": float(best["davies_bouldin"]),
        "ari": float(best["ari"]),
        "nmi": float(best["nmi"]),
        "confidence": best["confidence"],
        "note": "optimal k determined via silhouette scan + kneedle in tune.py"
    }


# ----- DBSCAN Tuning ------------------------------------------------
def tune_dbscan(X: np.ndarray, df: pd.DataFrame) -> dict:
    """
    Run UMAP grid search for DBSCAN.
    For each UMAP config and eps value, evaluate clustering quality.
    Returns best configuration and metrics.
    """
    print("\n" + "="*60)
    print("DBSCAN Tuning")
    print("="*60)

    umap_cache = {}
    results = []

    combos = list(itertools.product(
        DBSCAN_UMAP_GRID["n_neighbors"],
        DBSCAN_UMAP_GRID["n_components"]
    ))

    print(f"\n{'n_neighbors':<13} {'n_components':<14} {'eps':<7} {'n_clusters':<12} {'silhouette':<12} {'db_score':<10} {'ari':<8} {'nmi'}")
    print("-" * 90)

    for n_neighbors, n_components in combos:
        cache_key = (n_neighbors, n_components)
        if cache_key not in umap_cache:
            umap_cache[cache_key] = fit_umap(X, n_neighbors, n_components)
        X_umap = umap_cache[cache_key]

        for eps in DBSCAN_EPS_VALUES:
            db = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
            labels = db.fit_predict(X_umap)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()

            if n_clusters < 2:
                continue

            mask = labels != -1
            if mask.sum() < 100:
                continue

            labels_clean = labels[mask]
            X_clean = X_umap[mask]
            df_clean = df[mask]

            sil = silhouette_score(X_clean, labels_clean,
                                   sample_size=KMEANS_SILHOUETTE_SAMPLE,
                                   random_state=42)
            db_score = davies_bouldin_score(X_clean, labels_clean)
            ari = adjusted_rand_score(df_clean["intent"], labels_clean)
            nmi = normalized_mutual_info_score(df_clean["intent"], labels_clean)

            results.append({
                "n_neighbors": n_neighbors,
                "n_components": n_components,
                "eps": eps,
                "n_clusters": n_clusters,
                "noise_pct": round(n_noise / len(labels) * 100, 1),
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db_score, 4),
                "ari": round(ari, 4),
                "nmi": round(nmi, 4)
            })

            print(f"{n_neighbors:<13} {n_components:<14} {eps:<7} {n_clusters:<12} {sil:<12.4f} {db_score:<10.4f} {ari:<8.4f} {nmi:.4f}")

    results_df = pd.DataFrame(results).sort_values(
        ["silhouette", "davies_bouldin"],
        ascending=[False, True]
    )

    best = results_df.iloc[0]
    print(f"\n----- Best DBSCAN Configuration -----")
    print(f"  n_neighbors    : {int(best['n_neighbors'])}")
    print(f"  n_components   : {int(best['n_components'])}")
    print(f"  eps            : {best['eps']}")
    print(f"  min_samples    : {DBSCAN_MIN_SAMPLES}")
    print(f"  n_clusters     : {int(best['n_clusters'])}")
    print(f"  silhouette     : {best['silhouette']}")
    print(f"  davies_bouldin : {best['davies_bouldin']}")
    print(f"  ari            : {best['ari']}")
    print(f"  nmi            : {best['nmi']}")

    return {
        "umap_n_neighbors": int(best["n_neighbors"]),
        "umap_n_components": int(best["n_components"]),
        "umap_min_dist": 0.0,
        "umap_metric": "cosine",
        "eps": float(best["eps"]),
        "min_samples": DBSCAN_MIN_SAMPLES,
        "n_clusters": int(best["n_clusters"]),
        "silhouette": float(best["silhouette"]),
        "davies_bouldin": float(best["davies_bouldin"]),
        "ari": float(best["ari"]),
        "nmi": float(best["nmi"]),
        "note": "optimal params determined via grid search in tune.py"
    }


# ----- Comparison Table ---------------------------------------------
def print_comparison(kmeans_config: dict, dbscan_config: dict) -> None:
    """Print final K-Means vs DBSCAN comparison table."""
    print("\n" + "="*60)
    print("Final Comparison")
    print("="*60)
    print(f"\n{'Metric':<20} {'K-Means':<12} {'DBSCAN'}")
    print("-" * 45)
    print(f"{'Silhouette':<20} {kmeans_config['silhouette']:<12} {dbscan_config['silhouette']}")
    print(f"{'Davies-Bouldin':<20} {kmeans_config['davies_bouldin']:<12} {dbscan_config['davies_bouldin']}")
    print(f"{'ARI':<20} {kmeans_config['ari']:<12} {dbscan_config['ari']}")
    print(f"{'NMI':<20} {kmeans_config['nmi']:<12} {dbscan_config['nmi']}")
    print(f"{'N clusters':<20} {kmeans_config['k']:<12} {dbscan_config['n_clusters']}")
    print(f"\nNote: algorithms evaluated on their own optimal UMAP spaces.")
    print(f"K-Means UMAP : n_neighbors={kmeans_config['umap_n_neighbors']}, n_components={kmeans_config['umap_n_components']}")
    print(f"DBSCAN UMAP  : n_neighbors={dbscan_config['umap_n_neighbors']}, n_components={dbscan_config['umap_n_components']}")


# ----- Save Config --------------------------------------------------
def save_config(kmeans_config: dict, dbscan_config: dict,
                df: pd.DataFrame, embedding_model: str) -> None:
    """Save best parameters and metrics to config.json."""
    config = {
        "kmeans": kmeans_config,
        "dbscan": dbscan_config,
        "dataset_size": len(df),
        "embedding_model": embedding_model
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {CONFIG_PATH}")


# ----- Main ---------------------------------------------------------
def main() -> None:
    """Run full tuning pipeline for K-Means and DBSCAN."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    embeddings, df = load_data()
    X = normalise(embeddings)

    kmeans_config = tune_kmeans(X, df)
    dbscan_config = tune_dbscan(X, df)

    print_comparison(kmeans_config, dbscan_config)
    save_config(kmeans_config, dbscan_config, df, "all-MiniLM-L6-v2")


if __name__ == "__main__":
    main()