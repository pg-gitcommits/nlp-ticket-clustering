"""
visualise.py
------------
Generates interactive 2D cluster visualisation for K-Means and DBSCAN
clustering results using UMAP dimensionality reduction and Plotly.

Produces an HTML file with toggleable K-Means and DBSCAN views,
auto-generated cluster labels via TF-IDF, and hover-over ticket text.

Usage:
    python src/visualise.py
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import warnings

warnings.filterwarnings("ignore")

# ----- Config -------------------------------------------------------
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
CSV_PATH        = os.path.join(DATA_DIR, "tickets.csv")
OUTPUT_PATH     = os.path.join(OUTPUT_DIR, "clusters_viz.html")

UMAP_2D_PARAMS = {
    "n_components" : 2,
    "n_neighbors"  : 15,
    "min_dist"     : 0.1,
    "metric"       : "cosine",
    "random_state" : 42
}


# ----- Load Data ----------------------------------------------------
def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and ticket dataframe with cluster labels."""
    print("Loading data...")
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} tickets")
    print(f"Columns: {df.columns.tolist()}")
    return embeddings, df


# ----- UMAP 2D ------------------------------------------------------
def fit_umap_2d(X: np.ndarray) -> np.ndarray:
    """
    Fit UMAP to 2D for visualisation.
    Uses min_dist=0.1 for readable scatter plots
    (distinct from clustering UMAP which uses min_dist=0.0).
    """
    print("Fitting UMAP 2D for visualisation...")
    reducer = umap.UMAP(**UMAP_2D_PARAMS)
    coords = reducer.fit_transform(X)
    print(f"2D coords shape: {coords.shape}")
    return coords


# ----- Cluster Labels -----------------------------------------------
def label_clusters(df: pd.DataFrame,
                   cluster_col: str, top_n: int = 3) -> dict:
    """Generate cluster labels using TF-IDF top keywords."""
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


# ----- Build Figure -------------------------------------------------
def build_figure(df: pd.DataFrame) -> go.Figure:
    """
    Build interactive Plotly figure with K-Means and DBSCAN toggles.
    Each algorithm shown as a separate trace group.
    """
    fig = go.Figure()

    for algo, cluster_col, label_col in [
        ("K-Means", "cluster_kmeans", "label_kmeans"),
        ("DBSCAN",  "cluster_dbscan", "label_dbscan")
    ]:
        visible = True if algo == "K-Means" else False

        for cluster_id in sorted(df[cluster_col].unique()):
            mask = df[cluster_col] == cluster_id
            subset = df[mask]
            label = subset[label_col].iloc[0]
            name = f"Noise" if cluster_id == -1 else f"C{cluster_id}: {label}"

            fig.add_trace(go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                name=name,
                text=subset["text"],
                customdata=subset[["intent", cluster_col]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Intent: %{customdata[0]}<br>"
                    "Cluster: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
                marker=dict(size=4, opacity=0.7),
                visible=visible,
                legendgroup=algo,
                legendgrouptitle_text=algo if cluster_id == sorted(
                    df[cluster_col].unique())[0] else None
            ))

    fig.update_layout(
        title="Customer Support Ticket Clusters",
        xaxis_title="UMAP dimension 1",
        yaxis_title="UMAP dimension 2",
        height=700,
        hovermode="closest",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.12,
                buttons=[
                    dict(
                        label="K-Means",
                        method="update",
                        args=[
                            {"visible": [
                                t.legendgroup == "K-Means"
                                for t in fig.data
                            ]},
                            {"title": "Customer Support Ticket Clusters — K-Means"}
                        ]
                    ),
                    dict(
                        label="DBSCAN",
                        method="update",
                        args=[
                            {"visible": [
                                t.legendgroup == "DBSCAN"
                                for t in fig.data
                            ]},
                            {"title": "Customer Support Ticket Clusters — DBSCAN"}
                        ]
                    ),
                    dict(
                        label="Both",
                        method="update",
                        args=[
                            {"visible": [True] * len(fig.data)},
                            {"title": "Customer Support Ticket Clusters — Both"}
                        ]
                    )
                ]
            )
        ]
    )

    return fig


# ----- Main ---------------------------------------------------------
def main() -> None:
    """Run visualisation pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    embeddings, df = load_data()
    X = normalize(embeddings)

    coords = fit_umap_2d(X)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    print("Generating cluster labels...")
    km_labels = label_clusters(df, "cluster_kmeans")
    db_labels = label_clusters(df, "cluster_dbscan")

    df["label_kmeans"] = df["cluster_kmeans"].map(km_labels)
    df["label_dbscan"] = df["cluster_dbscan"].map(db_labels)

    print("Building figure...")
    fig = build_figure(df)

    fig.write_html(OUTPUT_PATH)
    print(f"\nSaved visualisation to {OUTPUT_PATH}")

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved updated tickets to {CSV_PATH}")


if __name__ == "__main__":
    main()