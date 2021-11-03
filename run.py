import random
import spacy
from spacy.lang.en import English
from spacy import displacy
from process_dataset import getData
from relation_extraction import subtree_matcher, extractTriplesCoreNLP
from data_analysis import (
    visualise,
    visuliseKnowledgeGraph,
    frequencyGraph,
    bertEmbedding,
    getAllTuples,
)
from stanza.server import CoreNLPClient
import networkx as nx


def getSentences(text):
    nlp = English()
    nlp.add_pipe("sentencizer")
    document = nlp(text)
    return [sent.text.strip() for sent in document.sents]


def appendChunk(original, chunk):
    return original + " " + chunk


def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)


def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)


def processSubjectObjectPairs(tokens):
    subject = ""
    object = ""
    relation = ""
    subjectConstruction = ""
    objectConstruction = ""
    for token in tokens:
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if isConstructionCandidate(token):
            if subjectConstruction:
                subjectConstruction = appendChunk(subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = appendChunk(objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
            subject = appendChunk(subjectConstruction, subject)
            subjectConstruction = ""
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
            object = appendChunk(objectConstruction, object)
            objectConstruction = ""

    return (subject.strip(), relation.strip(), object.strip())


def evaluateModels():
    # Get 9 different extracts
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
    path = "./filtered10Relations.gpickle"
    G = nx.read_gpickle(path)
    nx_nodes = G.nodes(data=True)
    for n in nx_nodes:
        print(n)


if __name__ == "__main__":
    visuliseKnowledgeGraph()

    """
    data = getData()
    text = 'San Francisco considers banning sidewalk delivery robots'
    sentences = getSentences(text)
    for sentence in sentences:
        print(sentence)
        print("Spacy:")
        print(subtree_matcher(sentence))
        print("CoreNLP:")
        triples = extractTriplesCoreNLP(sentence)
        for triple in triples:
            print(triple)
        print("AllenNLP:")
        print(extractTriplesAllenNLP())
    """
