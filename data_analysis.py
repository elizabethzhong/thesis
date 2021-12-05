from collections import Counter
from itertools import chain
import random
from string import punctuation

import torch
from transformers import BertTokenizer, BertModel
from process_dataset import getData
from relation_extraction import extractTriplesCoreNLP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from stanza.server import CoreNLPClient
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from pyvis.network import Network

from constants import TOP10RELATIONS, ALL_CATEGORIES
from unsupervisedClustering import clusterTuples


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
random.seed(2020)


def bertEmbedding(text=""):
    """Return the embedding of a sentence using BERT"""
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])

    model = BertModel.from_pretrained(
        "bert-base-uncased",
        output_hidden_states=True,
    )

    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor)

        hidden_states = outputs[2]

        word_embed_6 = torch.cat([hidden_states[i] for i in [-1, -2, -3, -4]], dim=-1)

    return word_embed_6


def visualiseEmbeddings():
    """Visualise BERT embeddings of all sentences in the dataset on a graph"""
    labels = []
    tokens = []

    with CoreNLPClient(annotators=["openie"], be_quiet=True) as client:
        for category in ALL_CATEGORIES:
            data = getData(category)
            for sentence in data:
                if (triple := extractTriplesCoreNLP(client, sentence)) == []:
                    continue
                (subject, relation, object, subjectSpan) = triple[0]
                print("Subject index", subjectSpan)
                embeddings = bertEmbedding(sentence)
                print("Embedding size", embeddings.size())

                subjectEmbedding = np.array(embeddings)[0][subjectSpan]
                tokens.append(subjectEmbedding)
                labels.append((subject, ALL_CATEGORIES.index(category)))

    tsne = TSNE(n_components=2, random_state=0)

    new_values = tsne.fit_transform(tokens)

    x = []
    y = []
    colors = cm.rainbow(np.linspace(0, 1, len(ALL_CATEGORIES)))
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(
            x[i],
            y[i],
            c=colors[labels[i][1]].reshape(1, -1),
            label=ALL_CATEGORIES[labels[i][1]],
        )
        plt.annotate(
            labels[i][0],
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def generateKnowledgeGraph():
    """Generate the knowledge graph from the CUAD dataset"""
    graphPath = "CUADKnowledgeGraph.gpickle"
    G = nx.DiGraph()

    print("Generating knowledge graph from CUAD dataset...")

    tuples = getAllTuples()
    for tup in tuples:
        G.add_node(tup["tuple"]["subject"], group=ALL_CATEGORIES.index(tup["label"]))
        G.add_node(tup["tuple"]["object"], group=ALL_CATEGORIES.index(tup["label"]))
        G.add_edge(
            tup["tuple"]["subject"],
            tup["tuple"]["object"],
            label=tup["tuple"]["relation"],
            sentence=tup["sentence"],
            group=ALL_CATEGORIES.index(tup["label"]),
        )
    colored_dict = nx.get_node_attributes(G, "color")
    color_seq = [colored_dict.get(node, "blue") for node in G.nodes()]
    pos = nx.spring_layout(G, scale=2)
    nx.draw(G, pos, with_labels=True, node_color=color_seq)

    nx.write_gpickle(G, graphPath)
    print(f"Complete! Graph saved at {graphPath}")


def getAllTuples():
    """Extract all tuples from sentences in the CUAD dataset"""
    lemmatizer = WordNetLemmatizer()
    tuples = []
    with CoreNLPClient(annotators=["openie"], be_quiet=True) as client:
        for category in ALL_CATEGORIES:
            data = getData(category)
            for sentence in data:
                if (triples := extractTriplesCoreNLP(client, sentence)) == []:
                    continue
                flag = False
                for triple in triples:
                    (subject, relation, object) = triple
                    subject = subject.lower()
                    relation = relation.lower()
                    object = object.lower()

                    if subject == "agreement":
                        continue

                    tupleDict = {}
                    # only include triples with relations in top 10
                    if any(
                        item in lemmatise(relation, lemmatizer)
                        for item in TOP10RELATIONS[category]
                    ):
                        flag = True
                        tupleDict["tuple"] = {
                            "subject": subject,
                            "relation": relation,
                            "object": object,
                        }
                        tupleDict["sentence"] = sentence
                        tupleDict["label"] = category
                        tuples.append(tupleDict)

                if flag == False:
                    randomTriple = triples[0]
                    (subject, relation, object) = randomTriple
                    subject = subject.lower()
                    relation = relation.lower()
                    object = object.lower()

                    tupleDict["tuple"] = {
                        "subject": subject,
                        "relation": relation,
                        "object": object,
                    }
                    tupleDict["sentence"] = sentence
                    tupleDict["label"] = category
                    tuples.append(tupleDict)
                    flag = True

    # Cluster words into separate entities

    for word in ["party", "agreement", "term"]:
        frequentTuples = [
            x
            for x in tuples
            if x["tuple"]["subject"] == word or x["tuple"]["object"] == word
        ]
        tuples = [
            x
            for x in tuples
            if x["tuple"]["subject"] != word and x["tuple"]["object"] != word
        ]
        test = clusterTuples(word, frequentTuples)
        tuples = tuples + test

    return tuples


def frequencyGraph(category="Expiration Date"):
    """Plot top 10 frequency of words in a particular category"""
    lemmatizer = WordNetLemmatizer()
    # f = open("./bestRelations.txt", "w")
    with CoreNLPClient(annotators=["openie"], be_quiet=True) as client:
        allData = []
        for category in ALL_CATEGORIES:
            data = getData(category)
            for sentence in data:
                if (triples := extractTriplesCoreNLP(client, sentence)) == []:
                    continue
                for triple in triples:
                    (subject, relation, object) = triple
                    allData.append(lemmatise(subject.lower(), lemmatizer))
                    allData.append(lemmatise(object.lower(), lemmatizer))
        translator = str.maketrans("", "", punctuation)
        linewords = (
            sentence.translate(translator).lower().split() for sentence in allData
        )
        frequency = Counter(chain.from_iterable(linewords))
        extraStopwords = stopwords.words() + ["may", "shall"]
        newdict = {k: frequency[k] for k in frequency if k not in extraStopwords}
        frequencyDictWithoutStopwords = {
            k: v
            for k, v in sorted(newdict.items(), key=lambda item: item[1], reverse=True)
        }
        first10Words = list(frequencyDictWithoutStopwords.items())[:10]

        # f.write(category + str(first10Words) + "\n")
        indices = np.arange(len(first10Words))
        plt.bar(indices, list(map(lambda x: x[1], first10Words)), color="b")
        plt.xticks(
            indices, list(map(lambda x: x[0], first10Words)), rotation="vertical"
        )
        plt.tight_layout()
        plt.title(
            "'" + category.replace("/", "-") + "' Relation Frequency Histogram (Top 10)"
        )
        plt.show()
        # plt.savefig(
        #    "./graphs/relation/" + category.replace("/", "-") + "Relation.png"
        # )
    # f.close()


def lemmatise(phrase, client=WordNetLemmatizer()):
    """Get the lemmatised version of a phrase"""
    word_list = nltk.word_tokenize(phrase)
    lemmatised = " ".join([client.lemmatize(w) for w in word_list])
    return lemmatised


def extract(s):
    """Extract top 10 relation from file into a string for TOP10RELATIONS in constant.py"""
    string = ""
    categories = s.split("\n")
    for category in categories:
        index = category.find("[")
        string += "'" + category[:index] + "': "
        relations = category[index:]
        relations = "".join(i for i in relations if not i.isdigit())
        relations = relations.replace("(", "").replace(")", "").replace(", , ", " , ")
        string += relations + ",\n"
    return string
