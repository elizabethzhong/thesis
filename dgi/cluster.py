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
    gpu = 1
    # torch.cuda.set_device(gpu)

    labels = ["Pfizer-BioNTec", "Moderna", "Gamaleya", "Oxford-AstraZenece", "J and J"]

    gs_pk = ["../aboveAverageClass.gpickle"]

    all_feats = []
    all_gs = []
    all_sentences = []
    for f_g in gs_pk:
        g, features, sentences = create_dglgraph(f_g, sent_encoder)
        print(features.shape, len(sentences))

        g = dgl.add_self_loop(g)

        # features = features.to(gpu)
        # g = g.to(gpu)

        all_gs.append(g)
        all_feats.append(features)
        all_sentences += sentences

    in_feat = all_feats[0].shape[1]

    dgi = DGI(in_feat, 512, 1, nn.PReLU(512), 0)
    # dgi = dgi.to(gpu)

    dgi_optimizer = torch.optim.Adam(dgi.parameters(), lr=1e-3, weight_decay=0.0)

    dgi.load_state_dict(torch.load("aboveAverageClass.pkl"))

    G_sum = []
    X = []
    y = []
    feats = all_feats[0]
    g = all_gs[0]
    embeds = dgi.encoder(g, feats, corrupt=False)
    # classification = dgi.classification(embeds)
    g_sum = dgi.readout(embeds)
    G_sum.append(g_sum.detach().cpu())
    X.append(embeds.detach().cpu())
    print(embeds.shape[0])
    # y += [label] * embeds.shape[0]
    X = np.concatenate(X, axis=0)

    # assert len(y) == X.shape[0]

    tsne = TSNE(n_components=2, verbose=1, random_state=np.random.RandomState(0))
    z = tsne.fit_transform(X)
    df = pd.DataFrame()
    df["label"] = g.ndata["group"]
    df["x"] = z[:, 0]
    df["y"] = z[:, 1]

    """
    for i, s in enumerate(all_sentences):
        has_kw = False
        for k in keys:
            sp = s.split(" ")
            if k in s:
                has_kw = True
                if len(sp) > 3:
                    s = " ".join(sp[:4]) + "..."
                key_word_sents.append([i, k, s, y[i]])
        if not has_kw:
            key_word_sents.append([i, "", s, y[i]])
    """
    sns.scatterplot(
        x="x",
        y="y",
        hue="label",
        palette=sns.color_palette("hls", len(df["label"].unique())),
        data=df,
    ).set(title="Sentences data T-SNE projection")

    fig = plt.gcf()
    fig.set_size_inches(20, 15)

    a = pd.concat({"x": df["x"], "y": df["y"], "val": pd.Series(all_sentences)}, axis=1)
    j = 0
    for i, point in a.iterrows():
        if j % 5 == 0:
            plt.text(
                point["x"] + 0.01,
                point["y"],
                point["val"],
                horizontalalignment="left",
                size="small",
                color="black",
            )

        j = j + 1
    plt.show()

    plt.savefig("aboveAverageClass.png")


if __name__ == "__main__":
    main()
