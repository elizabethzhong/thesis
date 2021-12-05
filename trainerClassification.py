import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from stanza.server.client import CoreNLPClient
import torch
from torch import nn

from constants import ALL_CATEGORIES, TOP10RELATIONS
from data_analysis import lemmatise
from dgi.data import create_dglgraph
from dgi.dgi import DGI
from dgi.multiclass_classfication import MulticlassClassification
from process_dataset import preprocessData, getData
from relation_extraction import extractTriplesCoreNLP

torch.manual_seed(4)

sent_encoder = SentenceTransformer("paraphrase-distilroberta-base-v1")


def main(args):
    f_g = "./CUADKnowledgeGraph.gpickle"
    modelPath = "CUADKnowledgeGraphModel.pth"

    print("Organising dataset...")

    X_tuple, y_tuple = getSentenceDataset()

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
    use_tuple = True
    if use_tuple:
        tupleDict = getTupleDict(f_g)
        X_train_set = []
        y_train_set = []
        X_test_set = []
        y_test_set = []
        X_train_set_info = []
        X_test_set_info = []

        for idx, sentence in enumerate(sent_Train):
            embed_train = X_train[idx]
            label = y_train[idx]
            if sentence in tupleDict:
                for tup in tupleDict[sentence]:
                    tup_embed = tup["embedding"]
                    X_train_set.append(torch.cat([tup_embed, embed_train], dim=0))
                    X_train_set_info.append(
                        {
                            "sentence": sentence,
                            "subject": tup["subject"],
                            "object": tup["object"],
                        }
                    )
                    y_train_set.append(label)

            else:
                X_train_set.append(torch.cat([torch.zeros(1024), embed_train], dim=0))
                X_train_set_info.append(
                    {"sentence": sentence, "subject": None, "object": None}
                )
                y_train_set.append(label)

        for idx, sentence in enumerate(sent_Test):
            embed_test = X_test[idx]
            label = y_test[idx]
            if sentence in tupleDict:
                for tup in tupleDict[sentence]:
                    tup_embed = tup["embedding"]
                    X_test_set.append(torch.cat([tup_embed, embed_test], dim=0))
                    X_test_set_info.append(
                        {
                            "sentence": sentence,
                            "subject": tup["subject"],
                            "object": tup["object"],
                        }
                    )
                    y_test_set.append(label)
            else:
                X_test_set.append(torch.cat([torch.zeros(1024), embed_test], dim=0))
                X_test_set_info.append(
                    {"sentence": sentence, "subject": None, "object": None}
                )
                y_test_set.append(label)

        X_train = torch.stack(list(X_train_set), dim=0)
        X_test = torch.stack(list(X_test_set), dim=0)
        y_train = torch.tensor(y_train_set, dtype=torch.long)
        y_test = torch.tensor(y_test_set, dtype=torch.long)

    else:
        X_train = torch.stack(list(X_train), dim=0)
        X_test = torch.stack(list(X_test), dim=0)

    print("Training...")

    epochs = 250
    output_dim = 41
    if use_tuple == True:
        input_dim = 1792
    else:
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

    torch.save(model.state_dict(), modelPath)
    print(f"Model saved at: {modelPath}")

    print("Evaluating...")

    accuracyDict = {}
    predProbaList = torch.empty(size=(len(X_test), output_dim))
    predlist = torch.zeros(0, dtype=torch.long, device="cpu")

    # calculate accuracy for each class
    for i in range(len(X_test)):
        outputs = model(X_test[i])
        predicted = torch.argmax(outputs.data)
        predProbaList[i] = outputs.data
        predlist = torch.cat([predlist, predicted.view(-1).cpu()])
        if y_test[i].item() in accuracyDict:
            accuracyDict[y_test[i].item()].append(predicted == y_test[i])
        else:
            accuracyDict[y_test[i].item()] = [predicted == y_test[i]]
    for label in accuracyDict:
        accuracyDict[label] = accuracyDict[label].count(True) / len(accuracyDict[label])
    for k, v in sorted(accuracyDict.items(), key=lambda x: x[1]):
        print(k, v)

    cm = confusion_matrix(y_test.numpy(), predlist.numpy())
    sn.heatmap(cm, annot=True, cmap="YlGnBu")

    plotPrecisionRecall(
        y_test.numpy(),
        predProbaList.detach().numpy(),
    )


def getSentenceDataset():
    """Generate X and y for sentence dataset"""
    X_tuple = []
    y_tuple = []
    for idx, category in enumerate(ALL_CATEGORIES):
        data = getData(category, clean=True)
        for sentence in data:
            feat = sent_encoder.encode(sentence)
            feat = torch.tensor(feat)
            X_tuple.append((feat, sentence))
            y_tuple.append(idx)

    y_tuple = torch.tensor(y_tuple, dtype=torch.long)
    return X_tuple, y_tuple


def getTuplesFromSentence(testSentence, category):
    """Get tuples from a specific sentence"""
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
    print(testTuples)
    return testTuples


def evaluateTestSentence(embeddings, model, testTuples):
    """Evalutate test tuple"""
    for tup in testTuples:
        subject_embed = embeddings[tup[0]]
        object_embed = embeddings[tup[1]]
        output1 = model(torch.cat([subject_embed, object_embed]))
        predicted1 = torch.argmax(output1.data)
        print(f"The predicted class is: {ALL_CATEGORIES[predicted1]}")


def getTupleDict(f_g):
    """Organise tuples into a dictionary"""
    g, features, node_names = create_dglgraph(f_g, sent_encoder)
    g = dgl.add_self_loop(g)

    dgi = DGI(features.shape[1], 512, 1, nn.PReLU(512), 0)

    dgi.load_state_dict(torch.load("./CUADKnowledgeGraphEmbeddings.pkl"))

    embeds = dgi.encoder(g, features, corrupt=False).detach()

    embeddings = dict(zip(node_names, embeds))
    tupleDict = {}
    nxG = nx.read_gpickle(f_g)
    for link in nxG.edges(data=True):
        sentence = link[2]["sentence"]
        subject_embed = embeddings[link[0]]
        object_embed = embeddings[link[1]]
        if sentence in tupleDict:
            tupleDict[sentence].append(
                {
                    "subject": link[0],
                    "object": link[1],
                    "embedding": torch.cat([subject_embed, object_embed], dim=0),
                }
            )
        else:
            tupleDict[sentence] = [
                {
                    "subject": link[0],
                    "object": link[1],
                    "embedding": torch.cat([subject_embed, object_embed], dim=0),
                }
            ]

    return tupleDict


def getGraphDataset(f_g, testTuples=None):
    """Generate X and y for tuple dataset"""
    g, features, sent = create_dglgraph(f_g, sent_encoder)
    g = dgl.add_self_loop(g)

    dgi = DGI(features.shape[1], 512, 1, nn.PReLU(512), 0)

    dgi_optimizer = torch.optim.Adam(dgi.parameters(), lr=1e-3, weight_decay=0.0)

    dgi.load_state_dict(torch.load("./CUADKnowledgeGraphEmbeddings.pkl"))

    embeds = dgi.encoder(g, features, corrupt=False).detach()

    embeddings = dict(zip(sent, embeds))
    classes = dict(zip(sent, g.ndata["group"]))
    X_tuple = []
    y_tuple = []
    nxG = nx.read_gpickle(f_g)
    for link in nxG.edges(data=True):
        label = link[2]["group"]
        sentence_embed = sent_encoder.encode(link[2]["sentence"])
        subject_embed = embeddings[link[0]]
        object_embed = embeddings[link[1]]

        # if it's the exception then ignore
        if testTuples and (link[0], link[1]) in testTuples:
            print("Found relation")
            continue
        X_tuple.append(
            (
                torch.cat(
                    [subject_embed, object_embed, torch.tensor(sentence_embed)], dim=0
                ),
                link[1],
            )
        )

        # assign class of tuple to object
        y_tuple.append(label)

    y_tuple = torch.tensor(y_tuple, dtype=torch.long)

    return X_tuple, y_tuple, embeddings


def plotPrecisionAt80Recall(labels, precision):
    """Plot precision at 80% recall"""
    sortedLabels = [x for _, x in sorted(zip(precision, labels), reverse=True)]
    sortedPrecisions = sorted(precision, reverse=True)
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(sortedLabels)), sortedPrecisions, align="center")
    for i in range(len(sortedLabels)):
        print(sortedLabels[i], sortedPrecisions[i])
    ax.set_yticks(np.arange(len(sortedLabels)))
    ax.set_yticklabels(sortedLabels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlim([0, 1.0])
    ax.set_xlabel("Precision at 80% recall")
    # ax.set_ylim([0, 1])
    plt.show()


def multiclassROCAUCScore(y_test, y_pred, average="macro"):
    """Plot PR curve separately for all categories"""
    fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
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


def plotPrecisionRecall(y_test, y_pred):
    """Plot PR curve overall"""
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    precisions = []
    recalls = []

    # compute macro precision recall curve
    for (idx, c_label) in enumerate(ALL_CATEGORIES):
        precision_thresholds = []
        recall_thresholds = []
        fpr, tpr, thresholds = precision_recall_curve(
            y_test[:, idx].astype(int), y_pred[:, idx]
        )
        for i in np.linspace(0, 1, 1001):
            precision_thresholds.append(np.interp(i, fpr, tpr))
            recall_thresholds.append(i)

        precisions.append(precision_thresholds)
        recalls.append(recall_thresholds)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    average_precision = precisions.mean(axis=0)
    average_recalls = recalls.mean(axis=0)

    average = average_precision_score(y_test, y_pred)

    display = PrecisionRecallDisplay(
        precision=average_precision,
        recall=average_recalls,
        average_precision=average,
    )
    display.plot()
    _ = display.ax_.set_title("Precision Recall Curve")
    plt.show()
    return average


def runModel(sentence):
    """Run model for a sentence"""
    output_dim = 41
    input_dim = 768

    data = sent_encoder.encode(sentence)
    data = torch.tensor(data)

    model = MulticlassClassification(num_feature=input_dim, num_class=output_dim)

    model.load_state_dict(torch.load("sentenceEmbeddings.pth"))

    model.eval()
    output = model(data)
    prediction = torch.argmax(output)
    print(prediction)


def getConnectedTuples(node):
    """Find all nodes connected to specified node"""
    f_g = "./allLabelsFirstTriple.gpickle"
    nxG = nx.read_gpickle(f_g)
    print(nxG.in_edges(node, data=True))
    print(nxG.out_edges(node, data=True))


def evaluateModel(X_test, y_test):
    """Evaluate model"""
    output_dim = 41
    input_dim = 768

    model = MulticlassClassification(num_feature=input_dim, num_class=output_dim)

    model.load_state_dict(torch.load("CUADKnowledgeGraphEmbeddings.pth"))

    model.eval()

    fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
    accuracyDict = {}
    predProbaList = torch.empty(size=(len(X_test), output_dim))
    predlist = torch.zeros(0, dtype=torch.long, device="cpu")

    for i in range(len(X_test)):
        outputs = model(X_test[i])
        predicted = torch.argmax(outputs.data)
        predProbaList[i] = outputs.data
        predlist = torch.cat([predlist, predicted.view(-1).cpu()])
        if y_test[i].item() in accuracyDict:
            accuracyDict[y_test[i].item()].append(predicted == y_test[i])
        else:
            accuracyDict[y_test[i].item()] = [predicted == y_test[i]]
    for label in accuracyDict:
        accuracyDict[label] = accuracyDict[label].count(True) / len(accuracyDict[label])
    for k, v in sorted(accuracyDict.items(), key=lambda x: x[1]):
        print(k, v)

    cm = confusion_matrix(y_test.numpy(), predlist.numpy())
    sn.heatmap(cm, annot=True, cmap="YlGnBu")
    print(multiclassROCAUCScore(y_test.numpy(), predProbaList.detach().numpy()))

    plotPrecisionRecall(c_ax, y_test.numpy(), predProbaList.detach().numpy())
    plt.show()


if __name__ == "__main__":
    main(None)
