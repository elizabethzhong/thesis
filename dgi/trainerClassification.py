from data import create_dglgraph
import dgl
from dgi import DGI
import itertools
import time
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold
from multiclass_classfication import MulticlassClassification
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

sent_encoder = SentenceTransformer("paraphrase-distilroberta-base-v1")


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


def main(args):
    gs_pk = ["../filtered10Relations.gpickle"]

    all_feats = []
    all_gs = []
    for f_g in gs_pk:
        g, features, _ = create_dglgraph(f_g, sent_encoder)
        g = dgl.add_self_loop(g)

        all_gs.append(g)
        all_feats.append(features)

    X = g.ndata["val"]
    y = g.ndata["group"]

    kf = KFold(n_splits=2, shuffle=True)
    kf.get_n_splits(X)

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    print(y_train)
    batch_size = 100
    n_iters = 3000
    epochs = n_iters / (len(X_train) / batch_size)
    input_dim = 768
    output_dim = len(set(y.tolist()))
    lr_rate = 0.001

    print(output_dim)

    model = MulticlassClassification(num_feature=input_dim, num_class=output_dim)
    criterion = (
        torch.nn.CrossEntropyLoss()
    )  # computes softmax and then the cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    iter = 0
    for epoch in range(int(epochs)):
        labels = y_train
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 50 == 0:
            # calculate Accuracy
            correct = 0
            total = 0
            for i in range(len(X_train)):
                outputs = model(X_train[i])
                predicted = torch.argmax(outputs.data)
                # print(predicted, y_train[i], predicted == y_train[i])
                correct += (predicted == y_train[i]).sum()
            accuracy = 100 * correct / len(X_train)
            print(
                "Iteration: {}. Loss: {}. Accuracy: {}.".format(
                    iter, loss.item(), accuracy
                )
            )

            correct2 = 0
            for i in range(len(X_test)):
                outputs = model(X_test[i])
                predicted = torch.argmax(outputs.data)
                # print(predicted, y_train[i], predicted == y_train[i])
                correct2 += (predicted == y_test[i]).sum()
            accuracy = 100 * correct2 / len(X_test)
            print("Test Accuracy: {}.".format(accuracy))


if __name__ == "__main__":
    main(None)
