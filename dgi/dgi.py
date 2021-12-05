"""
Deep Graph Infomax
Papers: https://arxiv.org/abs/1809.10341
"""
import math
import torch
from torch import nn
from dgi.gcn import GCN


class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.conv = GCN(in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, g, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(g.number_of_nodes())
            features = features[perm]

        features = self.conv(features, g)
        return features


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class MeanReadout(nn.Module):
    def __init__(self):
        super(MeanReadout, self).__init__()

    def forward(self, seq):
        return torch.sigmoid(seq.mean(dim=0))


class DGI(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_layers, activation, dropout, l2_mutual=False
    ):
        super(DGI, self).__init__()
        self.encoder = Encoder(in_feats, n_hidden, n_layers, activation, dropout)
        self.discriminator = Discriminator(n_hidden)
        self.readout = MeanReadout()
        self.loss = nn.BCEWithLogitsLoss()
        self.l2_mutual = l2_mutual
        self.classification = LogisticRegression(768, 24)

    def _self_mutual_info(self, feature, g):
        positive = self.encoder(g, feature, corrupt=False)
        negative = self.encoder(g, feature, corrupt=True)
        summary = self.readout(positive)
        classification = self.classification(feature)

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))
        # l3 = self.loss(classification, g.ndata["group"])

        return l1 + l2

    # this is not needed
    def _cross_mutual_info(self, features, gs):
        assert len(features) == len(gs) == 2

        # Multigraph
        positive = self.encoder(gs[0], features[0], corrupt=False)
        negative = self.encoder(gs[1], features[1], corrupt=False)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2

    def forward(self, features, gs):
        assert len(features) == len(gs)

        l = 0
        # Input a single graph
        if len(features) == 1:
            feat = features[0]
            g = gs[0]
            l = self._self_mutual_info(feat, g)

        if len(features) == 2:
            # Multigraph
            l = self._self_mutual_info(features[0], gs[0]) + self._self_mutual_info(
                features[1], gs[1]
            )

            if self.l2_mutual:
                # Self Mutual
                l2 = self._cross_mutual_info(features, gs)
                l = l + l2

        return l
