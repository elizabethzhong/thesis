import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import torch

sent_encoder = SentenceTransformer("paraphrase-distilroberta-base-v1")


def clusterTuples(word, frequentTuples, n_clusters=4):
    """Perform clustering on frequent words"""
    nfeats = []
    for tuple in frequentTuples:
        feat = sent_encoder.encode(tuple["sentence"])
        nfeats.append(feat)
    nfeats = torch.tensor(nfeats)

    scaler = StandardScaler()
    scaler.fit(nfeats)
    X_scale = scaler.transform(nfeats)
    df_scale = pd.DataFrame(X_scale)

    pca = PCA(n_components=3)
    pca_scale = pca.fit_transform(df_scale)
    pca_df_scale = pd.DataFrame(pca_scale)

    kmeans_pca_scale = KMeans(
        n_clusters=n_clusters,
        n_init=100,
        max_iter=400,
        init="k-means++",
        random_state=42,
    ).fit(pca_df_scale)

    labels_pca_scale = kmeans_pca_scale.labels_
    clusters_pca_scale = pd.concat(
        [pca_df_scale, pd.DataFrame({"pca_clusters": labels_pca_scale})], axis=1
    )

    for idx, _ in enumerate(frequentTuples):
        if frequentTuples[idx]["tuple"]["subject"] == word:
            frequentTuples[idx]["tuple"]["subject"] = f"{word} {labels_pca_scale[idx]}"
        else:
            frequentTuples[idx]["tuple"]["object"] = f"{word} {labels_pca_scale[idx]}"

    return frequentTuples
