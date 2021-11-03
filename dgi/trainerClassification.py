from stanza.server.client import CoreNLPClient
from data import create_dglgraph
import sys
import dgl
import numpy as np
import torch
import seaborn as sn
from multiclass_classfication import MulticlassClassification
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import networkx as nx
from pyvis.network import Network
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


# setting path
sys.path.append("../")

from constants import CATEGORIES, ALL_CATEGORIES, TOP10RELATIONS
from process_dataset import preprocessData, getData
from relation_extraction import extractTriplesCoreNLP
from data_analysis import lemmatise

sent_encoder = SentenceTransformer("paraphrase-distilroberta-base-v1")


def main(args):
    f_g = "../allLabelsClustered.gpickle"
    X_tuple, y_tuple = getGraphDataset(f_g)
    # X_tuple, y_tuple = getSentenceDataset()
    X_train_sent, X_test_sent, y_train, y_test = train_test_split(
        X_tuple,
        y_tuple,
        stratify=y_tuple,
        test_size=0.20,
        random_state=123,
        shuffle=True,
    )
    X_train, sent_Train = zip(*X_train_sent)
    X_test, sent_Test = zip(*X_test_sent)
    X_train = torch.stack(list(X_train), dim=0)
    X_test = torch.stack(list(X_test), dim=0)

    batch_size = 100
    n_iters = 25000
    epochs = n_iters / (len(X_train) / batch_size)
    output_dim = 41
    input_dim = 768
    lr_rate = 0.001

    model = MulticlassClassification(num_feature=input_dim, num_class=output_dim)
    criterion = (
        torch.nn.CrossEntropyLoss()
    )  # computes softmax and then the cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    for epoch in range(int(epochs)):
        labels = y_train
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            # calculate Accuracy
            correct = 0
            total = 0
            for i in range(len(X_train)):
                outputs = model(X_train[i])
                predicted = torch.argmax(outputs.data)
                correct += (predicted == y_train[i]).sum()
            accuracy = 100 * correct / len(X_train)
            print(
                "Epoch: {}. Loss: {}. Accuracy: {}.".format(
                    epoch, loss.item(), accuracy
                )
            )

            correct2 = 0
            for i in range(len(X_test)):
                outputs = model(X_test[i])
                predicted = torch.argmax(outputs.data)
                correct2 += (predicted == y_test[i]).sum()
            accuracy = 100 * correct2 / len(X_test)
            print("Test Accuracy: {}.".format(accuracy))

    """
    # Evalutate test tuple
    for tup in testTuples:
        subject_embed = embeddings[tup[0]]
        object_embed = embeddings[tup[1]]
        output1 = model(torch.cat([subject_embed, object_embed]))
        predicted1 = torch.argmax(output1.data)
        print(f"The predicted class is: {ALL_CATEGORIES[predicted1]}")
    """

    accuracyDict = {}
    predProbaList = torch.empty(size=(len(X_test), output_dim))
    predlist = torch.zeros(0, dtype=torch.long, device="cpu")

    nxG = nx.read_gpickle(f_g)
    for i in range(len(X_test)):
        sent = sent_Test[i]
        outputs = model(X_test[i])
        predicted = torch.argmax(outputs.data)
        predProbaList[i] = outputs.data
        predlist = torch.cat([predlist, predicted.view(-1).cpu()])
        if y_test[i].item() in accuracyDict:
            accuracyDict[y_test[i].item()].append(predicted == y_test[i])
            if predicted == y_test[i]:
                nxG.add_node(sent, size=30)
            else:
                nxG.add_node(sent, size=20)
        else:
            accuracyDict[y_test[i].item()] = [predicted == y_test[i]]
            if predicted == y_test[i]:
                nxG.add_node(sent, size=30)
            else:
                nxG.add_node(sent, size=20)
    for label in accuracyDict:
        accuracyDict[label] = accuracyDict[label].count(True) / len(accuracyDict[label])
    for k, v in sorted(accuracyDict.items(), key=lambda x: x[1]):
        print(k, v)

    cm = confusion_matrix(y_test.numpy(), predlist.numpy())
    sn.heatmap(cm, annot=True, cmap="YlGnBu")
    # plt.show()
    print(multiclass_roc_auc_score(y_test.numpy(), predProbaList.detach().numpy()))

    net = Network(notebook=True, height="1000px", width="1000px", directed=True)
    net.from_nx(nxG)
    net.show("allLabels.html")


def getSentenceDataset():
    X_tuple = []
    y_tuple = []
    for category in ALL_CATEGORIES:
        data = getData(category)
        for sentence in data:
            feat = sent_encoder.encode(sentence)
            X_tuple.append((feat, sentence))
            y_tuple.append(category)
    return X_tuple, y_tuple


def getGraphDataset(f_g):
    g, features, sent = create_dglgraph(f_g, sent_encoder)
    g = dgl.add_self_loop(g)

    testSentence = "Neither party may assign this Agreement without the written consent of the other party save in the case where such assignment is to an EM Affiliate and prior written notice has been given to the Buyer."
    category = "Anti-Assignment"
    testSentence = preprocessData(testSentence)
    testTuples = []
    with CoreNLPClient(annotators=["openie"], be_quiet=True) as client:
        if (triples := extractTriplesCoreNLP(client, testSentence)) == []:
            print("Nothing extracted")
        for triple in triples:
            (subject, relation, object) = triple
            subject = subject.lower()
            relation = relation.lower()
            object = object.lower()
            # skips tuples with agreement as subject
            if subject == "agreement":
                continue
            # only include triples with relations in top 10
            if any(item in lemmatise(relation) for item in TOP10RELATIONS[category]):
                testTuples.append((subject, object))
                flag = True
    # relation classifications
    # 1) get pairs based on edges
    # 2) find subject and object nodes from embeddings
    # 3) set class as object node class -> more likely to be the correct class
    # 4) join together and then train
    embeddings = dict(zip(sent, features))
    classes = dict(zip(sent, g.ndata["group"]))
    X_tuple = []
    y_tuple = []
    nxG = nx.read_gpickle(f_g)
    for link in nxG.edges(data=True):
        label = link[2]["group"]
        subject_embed = embeddings[link[0]]
        object_embed = embeddings[link[1]]
        # if it's the exception then ignore
        if (link[0], link[1]) in testTuples:
            print("Found relation")
            continue
        X_tuple.append((torch.cat([subject_embed, object_embed]), link[1]))

        # assign class of tuple to object
        y_tuple.append(label)

    y_tuple = torch.tensor(y_tuple, dtype=torch.long)

    # node classifications

    # X = list(zip(g.ndata["val"], sent))
    # y = g.ndata["group"]
    return X_tuple, y_tuple


def plotPrecisionAt80Recall(labels, precision):
    sortedLabels = [x for _, x in sorted(zip(precision, labels), reverse=True)]
    sortedPrecisions = sorted(precision, reverse=True)
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(sortedLabels)), sortedPrecisions, align="center")
    for i in range(len(sortedLabels)):
        print(sortedLabels[i], sortedPrecisions[i])
    ax.set_yticks(np.arange(len(sortedLabels)))
    ax.set_yticklabels(sortedLabels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Precision at 80% recall")
    # ax.set_ylim([0, 1])
    plt.show()


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    # y_pred = lb.transform(y_pred)
    precisionAt80 = []

    for (idx, c_label) in enumerate(ALL_CATEGORIES):
        fpr, tpr, thresholds = precision_recall_curve(
            y_test[:, idx].astype(int), y_pred[:, idx]
        )
        c_ax.plot(fpr, tpr, label=idx)
        precisionAt80.append(np.interp(0.8, fpr, tpr))
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.show()
    plotPrecisionAt80Recall(ALL_CATEGORIES, precisionAt80)
    return average_precision_score(y_test, y_pred)


if __name__ == "__main__":
    main(None)
