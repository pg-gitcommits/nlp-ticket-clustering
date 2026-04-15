"""
app.py
------
Streamlit application for exploring customer support ticket clusters.
Loads precomputed clustering results and displays interactive
visualisations, metrics, cluster details and live ticket prediction.

Usage:
    streamlit run app.py
"""

import os
import json
import subprocess
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# ----- Config -------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
CSV_PATH     = os.path.join(DATA_DIR, "tickets.csv")
METRICS_PATH = os.path.join(DATA_DIR, "metrics.json")
CONFIG_PATH  = os.path.join(BASE_DIR, "config.json")

st.set_page_config(
    page_title="Ticket Lens",
    page_icon="🎯",
    layout="wide"
)


# ----- Load Data ----------------------------------------------------
@st.cache_data
def load_data() -> tuple[pd.DataFrame, dict, dict]:
    df = pd.read_csv(CSV_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return df, metrics, config


@st.cache_data
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


@st.cache_data
def get_color_map(df: pd.DataFrame, cluster_col: str) -> dict:
    """Create consistent color mapping for clusters."""
    unique_clusters = sorted(df[cluster_col].unique())
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    return {
        str(cid): colors[i % len(colors)]
        for i, cid in enumerate(unique_clusters)
    }


# ----- Main App -----------------------------------------------------
def main():
    df, metrics, config = load_data()

    # ----- Sidebar --------------------------------------------------
    st.sidebar.title("Ticket Lens")
    st.sidebar.markdown("Customer support ticket clustering explorer.")

    algo = st.sidebar.radio(
        "Algorithm",
        ["K-Means", "DBSCAN"],
        help="Switch between clustering algorithms"
    )

    cluster_col  = "cluster_kmeans" if algo == "K-Means" else "cluster_dbscan"
    algo_key     = "kmeans" if algo == "K-Means" else "dbscan"
    algo_metrics = metrics[algo_key]
    algo_config  = config[algo_key]

    cluster_labels = get_cluster_labels(df, cluster_col)
    color_map      = get_color_map(df, cluster_col)

    cluster_options = {
        f"C{cid}: {label}": cid
        for cid, label in sorted(cluster_labels.items())
        if cid != -1
    }
    if -1 in df[cluster_col].values:
        cluster_options["Noise (-1)"] = -1

    selected_label = st.sidebar.selectbox(
        "Filter by cluster",
        ["All clusters"] + list(cluster_options.keys())
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model parameters**")
    st.sidebar.markdown(
        f"UMAP: n_neighbors={algo_config['umap_n_neighbors']}, "
        f"n_components={algo_config['umap_n_components']}"
    )
    if algo == "K-Means":
        st.sidebar.markdown(
            f"K-Means: k={algo_config['k']}, "
            f"n_init={algo_config['n_init']}"
        )
    else:
        st.sidebar.markdown(
            f"DBSCAN: eps={algo_config['eps']}, "
            f"min_samples={algo_config['min_samples']}"
        )
    st.sidebar.markdown(f"Embedding: `{config['embedding_model']}`")

    # ----- Header ---------------------------------------------------
    st.title("Ticket Lens")
    st.markdown(
        f"Exploring **{len(df):,}** customer support tickets "
        f"clustered into **{algo_metrics['n_clusters']}** groups "
        f"using **{algo}**."
    )

    # ----- Tabs -----------------------------------------------------
    tab1, tab2 = st.tabs(["Explore clusters", "Predict ticket"])

    # ----- Tab 1: Explore -------------------------------------------
    with tab1:

        # Metric Cards
        st.markdown("### Clustering metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Silhouette", f"{algo_metrics['silhouette']:.3f}",
                    help="Cluster tightness. Higher is better (max 1.0)")
        col2.metric("Davies-Bouldin", f"{algo_metrics['davies_bouldin']:.3f}",
                    help="Cluster separation. Lower is better (min 0.0)")
        col3.metric("ARI", f"{algo_metrics['ari']:.3f}",
                    help="Agreement with true labels. Higher is better (max 1.0)")
        col4.metric("NMI", f"{algo_metrics['nmi']:.3f}",
                    help="Topic information captured. Higher is better (max 1.0)")
        col5.metric("Clusters", algo_metrics['n_clusters'],
                    help="Number of clusters found")

        # Filter Data
        if selected_label == "All clusters":
            df_plot = df.copy()
        else:
            selected_id = cluster_options[selected_label]
            df_plot     = df[df[cluster_col] == selected_id].copy()

        df_plot["cluster_label"] = df_plot[cluster_col].map(cluster_labels)
        df_plot["cluster_str"]   = df_plot[cluster_col].astype(str)

        # UMAP Scatter Plot
        st.markdown("### Cluster visualisation")
        fig = px.scatter(
            df_plot,
            x="x", y="y",
            color="cluster_str",
            color_discrete_map=color_map,
            hover_data={
                "text"         : True,
                "intent"       : True,
                "cluster_label": True,
                "cluster_str"  : False,
                "x"            : False,
                "y"            : False
            },
            labels={
                "cluster_str"  : "Cluster",
                "cluster_label": "Topic",
                "x"            : "UMAP dimension 1",
                "y"            : "UMAP dimension 2"
            },
            title=f"{algo} clusters — UMAP 2D projection",
            height=550,
            opacity=0.7
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(legend_title_text="Cluster")
        st.plotly_chart(fig, width='stretch')

        # Cluster Details
        st.markdown("### Cluster details")

        if selected_label == "All clusters":
            summary_data = []
            for cid, label in sorted(cluster_labels.items()):
                size = (df[cluster_col] == cid).sum()
                summary_data.append({
                    "Cluster"   : cid,
                    "Topic"     : label,
                    "Size"      : size,
                    "% of data" : f"{size/len(df)*100:.1f}%"
                })
            summary_df = pd.DataFrame(summary_data).sort_values(
                "Size", ascending=False
            )
            st.dataframe(summary_df, width='stretch', hide_index=True)

        else:
            selected_id = cluster_options[selected_label]
            cluster_df  = df[df[cluster_col] == selected_id]

            col1, col2, col3 = st.columns(3)
            col1.metric("Tickets in cluster", len(cluster_df))
            col2.metric("% of dataset",
                        f"{len(cluster_df)/len(df)*100:.1f}%")
            col3.metric("Top keywords", cluster_labels[selected_id])

            st.markdown("**Sample tickets**")
            sample = cluster_df[["text", "intent"]].sample(
                min(20, len(cluster_df)), random_state=42
            )
            st.dataframe(sample, width='stretch', hide_index=True)

            st.markdown("**Intent distribution in this cluster**")
            intent_dist = cluster_df["intent"].value_counts().reset_index()
            intent_dist.columns = ["Intent", "Count"]
            fig_bar = px.bar(
                intent_dist,
                x="Intent", y="Count",
                title=f"Intent distribution — {selected_label}",
                height=350
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, width='stretch')

    # ----- Tab 2: Predict -------------------------------------------
    with tab2:
        st.markdown("### Predict cluster for a new ticket")
        st.markdown(
            "Enter a new support ticket to see which cluster "
            "it belongs to. Only available for K-Means."
        )

        new_ticket = st.text_input(
            "Support ticket text",
            placeholder="e.g. I need to cancel my order"
        )

        if st.button("Predict cluster"):
            if not new_ticket.strip():
                st.warning("Please enter a ticket text.")
            elif algo == "DBSCAN":
                st.warning(
                    "DBSCAN does not support prediction for new points. "
                    "Switch to K-Means to use this feature."
                )
            else:
                with st.spinner("Predicting..."):
                    result = subprocess.run(
                        ["python", "src/predict.py", new_ticket],
                        capture_output=True,
                        text=True,
                        cwd=BASE_DIR
                    )

                if result.returncode != 0:
                    st.error("Prediction failed. Check terminal for details.")
                else:
                    json_lines = [
                        l for l in result.stdout.strip().split("\n")
                        if l.strip().startswith("{")
                    ]
                    if not json_lines:
                        st.error("No prediction returned.")
                    else:
                        prediction = json.loads(json_lines[-1])
                        st.success(
                            f"Cluster **{prediction['cluster_id']}** "
                            f"— *{prediction['cluster_label']}*"
                        )
                        st.markdown("**Similar tickets in this cluster:**")
                        similar_df = pd.DataFrame(prediction["similar"])
                        st.dataframe(
                            similar_df, width='stretch', hide_index=True
                        )


if __name__ == "__main__":
    main()