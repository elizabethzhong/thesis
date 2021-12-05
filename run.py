import random
from spacy.lang.en import English
from spacy import displacy
from process_dataset import getData
from relation_extraction import subtree_matcher, extractTriplesCoreNLP
from data_analysis import (
    visualiseEmbeddings,
    generateKnowledgeGraph,
    frequencyGraph,
    bertEmbedding,
    getAllTuples,
)
from stanza.server import CoreNLPClient
import networkx as nx
from pyvis.network import Network
from constants import ALL_CATEGORIES


def getSentences(text):
    """Tokenise a sentence"""
    nlp = English()
    nlp.add_pipe("sentencizer")
    document = nlp(text)
    return [sent.text.strip() for sent in document.sents]


def evaluateModels():
    """Compare tuples extracted from Spacy and CoreNLP"""
    categories = ["Expiration Date"]
    with CoreNLPClient(annotators=["openie"], be_quiet=True) as client:
        for category in categories:
            data = getData(category)
            randomInt = random.randint(0, len(data))
            sentences = getSentences(data[0])
            for sentence in sentences:
                print(sentence)
                print("Spacy:")
                print(subtree_matcher(sentence))
                print("CoreNLP:")
                triples = extractTriplesCoreNLP(client, sentence)
                for triple in triples:
                    print(triple)


def loadGraph():
    """Load previously saved graph"""
    path = "./filtered10Relations.gpickle"
    G = nx.read_gpickle(path)
    nx_nodes = G.nodes(data=True)
    for n in nx_nodes:
        print(n)


def modifyGraph():
    """Modify previously saved graph"""
    f_g = "./allLabelsFirstTriple.gpickle"
    nxG = nx.read_gpickle(f_g)
    for link in nxG.edges(data=True):
        label = ALL_CATEGORIES[link[2]["group"]]
        print(label)
        if (
            label == "Ip Ownership Assignment"
            or label == "Third Party Beneficiary"
            or label == "Volume Restriction"
        ):
            nxG.add_node(link[0], size=30)
            nxG.add_node(link[1], size=30)
        else:
            nxG.add_node(link[0], size=0)
            nxG.add_node(link[1], size=0)
    net = Network(notebook=True, height="1000px", width="1000px", directed=True)
    net.from_nx(nxG)
    net.show("allLabelsModified.html")


if __name__ == "__main__":
    generateKnowledgeGraph()
