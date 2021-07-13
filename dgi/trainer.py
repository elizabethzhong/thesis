from data import create_dglgraph
import dgl
from dgi import DGI
import itertools
import time
import numpy as np
import torch
from torch import nn

from sentence_transformers import SentenceTransformer

sent_encoder = SentenceTransformer("paraphrase-distilroberta-base-v1")


def main(args):
    gs_pk = ["../5labels.gpickle"]

    gpu = 0
    # torch.cuda.set_device(gpu)

    all_feats = []
    all_gs = []
    for f_g in gs_pk:
        g, features, _ = create_dglgraph(f_g, sent_encoder)
        g = dgl.add_self_loop(g)

        # features = features.to(gpu)
        # g = g.to(gpu)

        all_gs.append(g)
        all_feats.append(features)

    # g_pairs = list(itertools.combinations(range(len(all_gs)), 2))
    in_feat = all_feats[0].shape[1]

    dgi = DGI(in_feat, 512, 1, nn.PReLU(512), 0, l2_mutual=False)  # True
    # dgi = dgi.to(gpu)

    dgi_optimizer = torch.optim.Adam(dgi.parameters(), lr=1e-3, weight_decay=0.0)

    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []
    for epoch in range(500):
        if epoch >= 3:
            t0 = time.time()

        dgi.train()
        dgi_optimizer.zero_grad()

        total_loss = 0
        # for id_1, id_2 in g_pairs:
        # features = [all_feats[id_1], all_feats[id_2]]
        # gs = [all_gs[id_1], all_gs[id_2]]

        # loss = dgi(features, gs)
        loss = dgi(all_feats, all_gs)
        loss.backward()
        dgi_optimizer.step()
        total_loss += loss

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(dgi.state_dict(), "5labels.pkl")
        else:
            cnt_wait += 1

        if cnt_wait == 100:
            print("Early stopping!")
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print(
            "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f}".format(
                epoch, np.mean(dur), loss.item()
            )
        )


if __name__ == "__main__":
    main(None)
