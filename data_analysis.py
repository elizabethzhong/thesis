import torch
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
import random
from process_dataset import getData
from relation_extraction import extractTriplesCoreNLP
from unsupervisedClustering import clusterTuples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from stanza.server import CoreNLPClient
import networkx as nx
from pyvis.network import Network
from collections import Counter
from string import punctuation
from itertools import chain
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from constants import CATEGORIES, TOP10RELATIONS, ALL_CATEGORIES
import dgl


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
random.seed(2020)

# word embeddings of a sentence using BERT
# returns sum of last four layers
def bertEmbedding(text=""):
    # Tokenize our sentence with the BERT tokenizer.
    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])

    model = BertModel.from_pretrained(
        "bert-base-uncased",
        # Whether the model returns all hidden-states.
        output_hidden_states=True,
    )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

        last_hidden_state = outputs[0]
        word_embed_1 = last_hidden_state
        # initial embeddings can be taken from 0th layer of hidden states
        word_embed_2 = hidden_states[0]
        # sum of all hidden states
        word_embed_3 = torch.stack(hidden_states).sum(0)
        # sum of second to last layer
        word_embed_4 = torch.stack(hidden_states[2:]).sum(0)
        # sum of last four layer
        word_embed_5 = torch.stack(hidden_states[-4:]).sum(0)
        # concatenate last four layers
        word_embed_6 = torch.cat([hidden_states[i] for i in [-1, -2, -3, -4]], dim=-1)

        print(word_embed_6)
        print(word_embed_6.shape)

    return word_embed_6


def visualise():
    labels = []
    tokens = []

    with CoreNLPClient(annotators=["openie"], be_quiet=True) as client:
        for category in CATEGORIES:
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
                labels.append((subject, CATEGORIES.index(category)))

    tsne = TSNE(n_components=2, random_state=0)

    new_values = tsne.fit_transform(tokens)

    x = []
    y = []
    colors = cm.rainbow(np.linspace(0, 1, len(CATEGORIES)))
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(
            x[i],
            y[i],
            c=colors[labels[i][1]].reshape(1, -1),
            label=CATEGORIES[labels[i][1]],
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


def visuliseKnowledgeGraph():
    G = nx.DiGraph()
    tuples = getAllTuples()
    for tup in tuples:
        # G.add_nodes_from(nodes, color=colors[CATEGORIES.index(category)])
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

    # net = Network(notebook=True, height="1000px", width="1000px", directed=True)
    nx.write_gpickle(G, "allLabels10Clusters.gpickle")
    # net.from_nx(G)
    # net.show("allLabelsNoFilter.html")

    # plt.show()


def getAllTuples():
    lemmatizer = WordNetLemmatizer()
    tuples = []
    with CoreNLPClient(annotators=["openie"], be_quiet=True) as client:
        for category in ALL_CATEGORIES:
            print("Test")
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
        print(test)
        tuples = tuples + test

    return tuples


def frequencyGraph(category="Expiration Date"):
    # Load lemmatiser
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
    word_list = nltk.word_tokenize(phrase)
    lemmatised = " ".join([client.lemmatize(w) for w in word_list])
    return lemmatised


# extract top 10 relation from file into constant
def extract(s):
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
