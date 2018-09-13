# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:45:35 2018

@author: Anthony
"""
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import numpy as np
import math
data = open('Data/SAO.txt', encoding = 'utf-8').read()
data = re.sub("[^a-zA-Z0-9\.']"," ", data)
#data = re.sub("[^a-zA-Z0-9\.,']","", data)
sentences = sent_tokenize(data)
words = word_tokenize(data)
wordSet = set(words)

indexToWord = {}
wordToIndex = {}
for i, word in enumerate(wordSet):
    indexToWord[i] = word
    wordToIndex[word] = i

vocabSize = len(indexToWord)
wordToIndex['<s>'] = vocabSize
indexToWord[vocabSize] = '<s>'

vocabSize += 1

arrayOfProbs = np.zeros((vocabSize, vocabSize))
#Takes around 15 seconds
for sentence in sentences:
    words = word_tokenize(sentence)
    words.insert(0, '<s>')
    words.append('<s>')
    for i in range(1, len(words) - 1):
        arrayOfProbs[wordToIndex[words[i]]][wordToIndex[words[i+1]]] += 1
arrayOfProbs += 1
arrayOfProbs = arrayOfProbs/(arrayOfProbs.sum(axis = 1)[:, None] + 10**-200)

def getProbability(string):
    wordsInString = word_tokenize(string)
    wordsInString.insert(0, '<s>')
    wordsInString.append('<s>')
    prob = 0
    for i in range(1, len(wordsInString) - 1):
        prob += math.log(arrayOfProbs[wordToIndex[wordsInString[i]]][wordToIndex[wordsInString[i + 1]]]+10**-200)
    print(prob)
    return math.exp(prob)
getProbability('Kirito defeated me')