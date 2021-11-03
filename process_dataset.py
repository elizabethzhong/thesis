import ast
from constants import CATEGORIES
from relation_extraction import filterBestTriples
import pandas as pd
from transformers import pipeline
import os


DATA_PATH = "./master_clauses.csv"
nlp = pipeline("ner", aggregation_strategy="simple")


def getData(category, clean=True):
    filename = "./nerText/" + category.replace("/", "-").replace(" ", "") + ".txt"
    if not clean:
        return getDataFromCSV(category)
    else:
        if os.path.isfile(filename):
            print("Loading cleaned data...")
            return getCleanedData(category)
        else:
            print("Loading data from CSV...")
            data = getDataFromCSV(category)
            return cleanData(data)


def getDataFromCSV(category="Expiration Date"):
    data = []
    df = pd.read_csv(DATA_PATH, header=0)
    categoryList = df[category].tolist()
    for sentenceList in categoryList:
        sentences = ast.literal_eval(sentenceList)
        if sentences == []:
            continue
        for sentence in sentences:
            data.append(sentence)
    return data


def cleanData(filename, sentenceList):
    f = open(filename, "w")
    data = []
    sentences = ast.literal_eval(sentenceList)
    for sentence in sentences:
        cleanedSentence = preprocessData(sentence)
        f.write(cleanedSentence + "\n")
        data.append(sentence)
    f.close()
    return data


def getCleanedData(category):
    filename = "./nerText/" + category.replace("/", "-").replace(" ", "") + ".txt"
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    return lines


def nameEntityRecognition(sentence):
    entities = nlp(sentence)
    filteredLocEntities = list(filter(lambda x: x["entity_group"] == "LOC", entities))
    filteredOrgEntities = list(filter(lambda x: x["entity_group"] == "ORG", entities))
    for entity in filteredOrgEntities:
        sentence = sentence.replace(entity["word"], "Party")
    for entity in filteredLocEntities:
        sentence = sentence.replace(entity["word"], "Location")
    return sentence


def preprocessData(sentence):
    # remove excess white space
    sentence = " ".join(sentence.split())
    # remove organisation name and location noise
    sentence = nameEntityRecognition(sentence)
    return sentence
