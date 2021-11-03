import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
import plotly.express as px

sent_encoder = SentenceTransformer("paraphrase-distilroberta-base-v1")


def clusterTuples(word, frequentTuples):
    nfeats = []
    for tuple in frequentTuples:
        feat = sent_encoder.encode(tuple["sentence"])
        nfeats.append(feat)
    nfeats = torch.tensor(nfeats)

    scaler = StandardScaler()
    scaler.fit(nfeats)
    X_scale = scaler.transform(nfeats)
    df_scale = pd.DataFrame(X_scale)

    """
    df_scale2 = df_scale.copy()
    kmeans_scale = KMeans(
        n_clusters=4, n_init=100, max_iter=400, init="k-means++", random_state=42
    ).fit(df_scale2)
    print(
        "KMeans Scaled Silhouette Score: {}".format(
            silhouette_score(df_scale2, kmeans_scale.labels_, metric="euclidean")
        )
    )
    labels_scale = kmeans_scale.labels_
    clusters_scale = pd.concat(
        [df_scale2, pd.DataFrame({"cluster_scaled": labels_scale})], axis=1
    )
    """

    pca = PCA(n_components=3)
    pca_scale = pca.fit_transform(df_scale)
    pca_df_scale = pd.DataFrame(pca_scale)

    kmeans_pca_scale = KMeans(
        n_clusters=10, n_init=100, max_iter=400, init="k-means++", random_state=42
    ).fit(pca_df_scale)
    print(
        "KMeans PCA Scaled Silhouette Score: {}".format(
            silhouette_score(pca_df_scale, kmeans_pca_scale.labels_, metric="euclidean")
        )
    )
    labels_pca_scale = kmeans_pca_scale.labels_
    clusters_pca_scale = pd.concat(
        [pca_df_scale, pd.DataFrame({"pca_clusters": labels_pca_scale})], axis=1
    )
    """
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        clusters_pca_scale.iloc[:, 0],
        clusters_pca_scale.iloc[:, 1],
        hue=labels_pca_scale,
        palette="Set1",
        s=100,
        alpha=0.2,
    ).set_title("KMeans Clusters (4) Derived from PCA", fontsize=15)
    plt.legend()
    plt.show()
    """

    for idx, _ in enumerate(frequentTuples):
        if frequentTuples[idx]["tuple"]["subject"] == word:
            frequentTuples[idx]["tuple"]["subject"] = f"{word} {labels_pca_scale[idx]}"
        else:
            frequentTuples[idx]["tuple"]["object"] = f"{word} {labels_pca_scale[idx]}"

    return frequentTuples

    """

    pca2 = PCA(n_components=3).fit(df_scale2)
    pca2d = pca2.transform(df_scale2)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        pca2d[:, 0], pca2d[:, 1], hue=labels_scale, palette="Set1", s=100, alpha=0.2
    ).set_title("KMeans Clusters (5) Derived from Original Dataset", fontsize=15)
    plt.legend()
    plt.ylabel("PC2")
    plt.xlabel("PC1")
    plt.show()

    Scene = dict(
        xaxis=dict(title="PC1"), yaxis=dict(title="PC2"), zaxis=dict(title="PC3")
    )
    labels = labels_scale
    trace = go.Scatter3d(
        x=pca2d[:, 0],
        y=pca2d[:, 1],
        z=pca2d[:, 2],
        mode="markers",
        marker=dict(
            color=labels,
            colorscale="Viridis",
            size=10,
            line=dict(color="gray", width=5),
        ),
    )
    layout = go.Layout(margin=dict(l=0, r=0), scene=Scene, height=1000, width=1000)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    fig.show()
    """
