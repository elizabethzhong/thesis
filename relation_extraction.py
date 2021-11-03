import spacy
from google.protobuf.json_format import MessageToDict
from collections import defaultdict


# Rule based supervised relation extraction using Spacy


def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)


def subtree_matcher(text):
    subjpass = 0
    relation = ""
    nlp = spacy.load("en_core_web_sm")

    relationTuples = []
    doc = nlp(text)

    for i, tok in enumerate(doc):
        # find dependency tag that contains the text "subjpass"
        if tok.dep_.find("subjpass") == True:
            subjpass = 1

        if isRelationCandidate(tok):
            relation += tok.text + " "

    x = ""
    y = ""

    # if subjpass == 1 then sentence is passive
    if subjpass == 1:
        for i, tok in enumerate(doc):
            if tok.dep_.find("subjpass") == True:
                y = tok.text

            if tok.dep_.endswith("obj") == True:
                x = tok.text

    # if subjpass == 0 then sentence is not passive
    else:
        for i, tok in enumerate(doc):
            if tok.dep_.endswith("subj") == True:
                x = tok.text

            if tok.dep_.endswith("obj") == True:
                y = tok.text

    return x, relation, y


# Unsupervised relation extraction using Stanza (CoreNLP)


def extractTriplesCoreNLP(client, text):
    triples = []
    ann = client.annotate(
        text, properties={"openie.triple.strict": "true", "openie.format": "qa_srl"}
    )
    for sentence in ann.sentence:
        for triple in sentence.openieTriple:
            tripleUnpacked = MessageToDict(triple, preserving_proto_field_name=True)
            triples.append(
                (
                    triple.subject,
                    triple.relation,
                    triple.object,
                    tripleUnpacked["subjectTokens"][0]["tokenIndex"],
                )
            )

    return filterBestTriples(triples)


def filterBestTriples(triples):
    # Group triples with the same subject and relation
    groups = defaultdict(list)
    bestTriples = []

    for triple in triples:
        subRel = (triple[0], triple[1])
        groups[subRel].append(triple[2])

    for pair, objectList in groups.items():
        subject, relation = pair
        objectList.sort(key=len, reverse=True)
        object = objectList[0]
        bestTriples.append((subject, relation, object))
    return bestTriples
