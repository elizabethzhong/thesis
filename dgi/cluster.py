import dgl
import torch
from torch import nn
from data import create_dglgraph
from dgi import DGI
from sentence_transformers import SentenceTransformer

sent_encoder = SentenceTransformer("paraphrase-distilroberta-base-v1")

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    gs_pk = ["../allLabels.gpickle"]

    all_feats = []
    all_gs = []
    all_sentences = []
    for f_g in gs_pk:
        g, features, sentences = create_dglgraph(f_g, sent_encoder)
        print(features.shape, len(sentences))

        g = dgl.add_self_loop(g)

        all_gs.append(g)
        all_feats.append(features)
        all_sentences += sentences

    in_feat = all_feats[0].shape[1]

    dgi = DGI(in_feat, 512, 1, nn.PReLU(512), 0)
    dgi_optimizer = torch.optim.Adam(dgi.parameters(), lr=1e-3, weight_decay=0.0)

    dgi.load_state_dict(torch.load("5labels.pkl"))

    G_sum = []
    X = []
    y = []
    feats = all_feats[0]
    g = all_gs[0]
    embeds = dgi.encoder(g, feats, corrupt=False)
    g_sum = dgi.readout(embeds)
    G_sum.append(g_sum.detach().cpu())
    X.append(embeds.detach().cpu())

    X = np.concatenate(X, axis=0)

    tsne = TSNE(n_components=2, verbose=1, random_state=np.random.RandomState(0))
    z = tsne.fit_transform(X)
    df = pd.DataFrame()
    df["label"] = g.ndata["group"]
    df["x"] = z[:, 0]
    df["y"] = z[:, 1]

    sns.scatterplot(
        x="x",
        y="y",
        hue="label",
        palette=sns.color_palette("hls", len(df["label"].unique())),
        data=df,
    ).set(title="Sentences data T-SNE projection")

    fig = plt.gcf()
    fig.set_size_inches(20, 15)

    plt.show()

    plt.savefig("allLabelEmbeddings.png")


if __name__ == "__main__":
    main()
