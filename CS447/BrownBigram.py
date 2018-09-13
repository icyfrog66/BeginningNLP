# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:44:07 2018

@author: Anthony
"""

import nltk
from nltk.corpus import brown
import random
from collections import Counter
import re

sentences = []
wordSet = set()
for file in brown.fileids():
    for sentence in brown.sents(file):
        sentences.append(sentence)
    for word in brown.words(file):
        wordSet.add(word)
words = []
for word in wordSet:
    words.append(word)

firstWordTransitions = {}
#for sentence in sentences:
for j, sentence in enumerate(sentences):
    print(j)
    for i in range(len(sentence) - 1):
        if sentence[i] not in firstWordTransitions.keys():
            firstWordTransitions[sentence[i]] = words.copy()
        firstWordTransitions[sentence[i]].append(sentence[i+1])
        if sentence[len(sentence) - 1] not in firstWordTransitions.keys():
            firstWordTransitions[sentence[len(sentence) - 1]] = []
        firstWordTransitions[sentence[len(sentence) - 1]].append(".")
        
def getProbability(sentenceString):
    sentenceList = re.sub('[^\w]', ' ', sentenceString).split()
    probability = 1
    for i, word in sentenceList:
        if i != 0:
            currentNextWords = firstWordTransitions[sentenceList[i-1]]
            theCounter = Counter(currentNextWords)
            currentProb = float(theCounter[word])/len(currentNextWords)
            probability *= currentProb
    return probability