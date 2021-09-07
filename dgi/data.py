import dgl
import torch
import networkx as nx


def graph_clean(G):
    nx_nodes = G.nodes(data=True)
    val_dict = {}
    for n in nx_nodes:
        val = n[1]["val"]
        if val not in val_dict:
            val_dict[val] = [n[0]]
        else:
            val_dict[val].append(n[0])

    val_dict = {k: v for k, v in val_dict.items() if len(v) > 1}

    for k, v in val_dict.items():
        x = v[0]
        for i, j in G[x].items():
            if i not in v:
                G.add_edge(x, i, rel=j["rel"])
        G.remove_nodes_from(v[1:])

    return G


def create_dglgraph(path, sent_encoder, bidirection=True):
    nxG = nx.read_gpickle(path)
    # nxG = graph_clean(nxG)

    nx_nodes = nxG.nodes(data=True)
    labels = [n[1]["group"] for n in nx_nodes]
    nx_edges = nxG.edges(data=True)
    nx_n_sid = [n[0] for n in nx_nodes]
    nx_n_id = list(range(len(nx_n_sid)))
    dict_sid_id = dict(zip(nx_n_sid, nx_n_id))
    nfeats, nsents = [], []
    for n in nx_nodes:
        # sent = n[1]['val']
        sent = n[0]
        feat = sent_encoder.encode(sent)
        nsents.append(sent)
        nfeats.append(feat)

    src, dst = [], []
    for idx, i in enumerate(nx_edges):
        s = dict_sid_id[i[0]]
        d = dict_sid_id[i[1]]
        src.append(s)
        dst.append(d)

    dglG = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=len(nx_nodes))
    dglG.ndata["val"] = torch.tensor(nfeats)
    dglG.ndata["group"] = torch.tensor(labels)

    if bidirection:
        dglG = dgl.to_bidirected(dglG, copy_ndata=True)

    nfeats = torch.tensor(nfeats)

    assert dglG.num_nodes() == len(nfeats)
    print()
    return dglG, nfeats, nsents
